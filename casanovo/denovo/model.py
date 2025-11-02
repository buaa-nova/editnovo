"""A de novo peptide sequencing model."""

from ast import Assert
import collections
import copy
import heapq
import logging
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch.nn.functional as F

from casanovo.depthcharge.components.levenstein_util import equal_ignore_tokens, is_a_loop, item

from ..depthcharge.masses import PeptideMass
import einops
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from ..depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder, Ranker

from . import evaluate
from .. import config
from ..data import ms_io

logger = logging.getLogger("casanovo")

DecoderOut = collections.namedtuple(
    "IterativeRefinementDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history"],
)

class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    max_length : int
        The maximum peptide length to decode.
    residues : Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams : int
        Number of beams used during beam search decoding.
    top_match : int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter : Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    train_label_smoothing : float
        Smoothing factor when calculating the training loss.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
    constant_lr_iters : int
        The number of iterations to keep constant learning rate in the constant phase scheduler.
    final_decay_iters : int
        The number of iterations for final decay from constant LR to minimum LR.
    min_lr_factor : float
        The minimum learning rate factor as a fraction of the base learning rate (default: 0.01).
    out_writer : Optional[ms_io.MztabWriter]
        The output writer for the prediction results.
    calculate_precision : bool
        Calculate the validation set precision during training.
        This is expensive.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter
        ] = None,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        constant_lr_iters: int = 20_000,
        final_decay_iters: int = 40_000,
        min_lr_factor: float = 0.001,
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        max_decoder_iters: int = 1,
        eos_penalty : int = 0,
        max_ratio: float = 0.5,
        dual_training_for_deletion: bool = False,
        no_share_discriminator: bool = False,
        dual_training_for_insertion: bool = False,
        sampling_model_gen: float = 0.3,
        glc_prob: float = 0.5,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build the model.
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
        )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
            dual_training_for_deletion=dual_training_for_deletion,
            no_share_discriminator=no_share_discriminator,
            dual_training_for_insertion=dual_training_for_insertion,
            sampling_model_gen=sampling_model_gen,
        )
        self.ranker = Ranker(dim_model=dim_model, n_head=n_head, dim_feedforward=dim_feedforward, n_layer=3, dropout_p=dropout)
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        self.constant_lr_iters = constant_lr_iters
        self.final_decay_iters = final_decay_iters
        self.min_lr_factor = min_lr_factor
        # `kwargs` will contain additional arguments as well as unrecognized
        # arguments, including deprecated ones. Remove the deprecated ones.
        for k in config._config_deprecated:
            kwargs.pop(k, None)
            warnings.warn(
                f"Deprecated hyperparameter '{k}' removed from the model.",
                DeprecationWarning,
            )
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.peptide_mass_calculator = PeptideMass(
            self.residues
        )
        self.stop_token = self.decoder._aa2idx["$"]
        self.max_iter = max_decoder_iters
        self.eos_penalty = eos_penalty
        self.max_ratio = max_ratio
        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = out_writer
        self.max_iter = max_decoder_iters
        self.extra_ignored_ids = [self.decoder.pad, self.decoder.bos, self.decoder.eos]
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=0.01
        )
        self.dual_training_for_deletion = dual_training_for_deletion
        self.dual_training_for_insertion = dual_training_for_insertion
        self.aa_mass = self.peptide_mass_calculator.mass_tensor
        self.DP_BIN_SIZE = 1
        self.MASS_C13_DIFF = 1.00335

    
    def freeze_encoder_decoder(self):
        """
        Freeze encoder and decoder parameters while keeping ranker trainable.
        This is useful for fine-tuning only the ranking component.
        """
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Freeze decoder parameters  
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        # Ensure ranker parameters remain trainable
        for param in self.ranker.parameters():
            param.requires_grad = True
            
        logger.info("Frozen encoder and decoder parameters. Only ranker is trainable.")
        
    def unfreeze_all(self):
        """
        Unfreeze all model parameters.
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all model parameters.")

    def beam_search_forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        memories, mem_masks = self.encoder(spectra)

        # Sizes.
        batch = spectra.shape[0]  # B
        
        beam = self.n_beams  # S
        # Initialize scores and tokens.
        prev_decoder_out = self.initialize_output_tokens(memories)
        
        # Prepare mass and charge
        masses = self.decoder.mass_encoder(precursors[:, None, 0])
        charges = self.decoder.charge_encoder(precursors[:, 1].int() - 1)
        precursors_embedding = masses + charges[:, None, :]
        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))
        spectra_index = torch.arange(batch, device=self.encoder.device)

        # Get the first prediction.
        cache_results = self.decoder.forward_beam_decoder(
                self.encoder, prev_decoder_out, precursors_embedding, memories, mem_masks, self.n_beams, spectra_index
            )
        # tokens shape: [BxS, L], scores shape: [BxS, L]
        tokens, scores, spectra_index = _select_top_beams(cache_results, self.n_beams, self.extra_ignored_ids,
                                                           self.decoder.pad, device=self.encoder.device)

        # Make all tensors the right shape for decoding.
        prev_decoder_out = prev_decoder_out._replace(
            output_tokens= einops.repeat(prev_decoder_out.output_tokens, "B L -> (B S) L", S=beam),  # [B, L]
            output_scores= einops.repeat(prev_decoder_out.output_scores, "B L -> (B S) L", S=beam),  # [B, L]
        ) 
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        precursors_embedding = einops.repeat(
            precursors_embedding, "B L D -> (B S) L D", S=beam
        )

        for step in range(1, self.max_iter + 1):
            # 1. check termination
            (
                finished_beams, #
                beam_fits_precursor,
                discarded_beams,
            ) = self._finish_beams(tokens, scores, prev_decoder_out, precursors, beam, step, self.extra_ignored_ids)

            # 2. collect finalized sentences
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            # Update the scores.[N_active, T, V]
            decoder_out = DecoderOut(
                output_tokens= tokens[~finished_beams, :],
                output_scores=scores[~finished_beams, :],
                attn=None,  
                step=step,
                max_step=self.max_iter + 1,
                history=None,
            )
            cache_results = self.decoder.forward_beam_decoder(
                self.encoder, 
                decoder_out,
                precursors_embedding[~finished_beams, :], 
                memories[~finished_beams, :],
                mem_masks[~finished_beams, :], 
                n_beam=beam,
                spectra_index=spectra_index[~finished_beams],
            )

            new_tokens, new_scores, spectra_index = _select_top_beams(
                cache_results, self.n_beams, self.extra_ignored_ids, self.decoder.pad, device=self.encoder.device
            )

            # Update the tokens and scores for the next step.
            prev_decoder_out = prev_decoder_out._replace(
                output_tokens=tokens,
                output_scores=scores,
            )
            tokens = new_tokens
            scores = new_scores

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return list(self._get_top_peptides(pred_cache, 1))

    def _cache_finished_beams(  
        self,
        tokens: torch.Tensor,  # [B*S, L]
        scores: torch.Tensor,  # [B*S, L]
        step: int,
        finished_beams: torch.Tensor,  # [B*S]
        beam_fits_precursor: torch.Tensor,  # [B*S]
        pred_cache: Dict[
            int, List[Tuple[bool, float, np.ndarray, str, torch.Tensor]]
        ],
    ):
        """
        将已终止的 beams 缓存到 pred_cache 中。
        Args:
            tokens:           [B*S, L] 当前 step 的所有 beam 输出
            scores:           [B*S, L] 当前 step 的所有 beam 输出分数
            step:             int       当前序列长度或位置
            finished_beams:   [B*S] BoolTensor,标记哪些 beam 该终止（循环不变）
            beam_fits_precursor: [B*S] BoolTensor,通过质量检查的 mask(示例全 True)
            pred_cache : Dict[
                    int, List[Tuple[bool, float, torch.Tensor, str]]
                ]
                Priority queue with finished beams for each spectrum, ordered by
                peptide score. For each finished beam, a tuple with the (negated)
                peptide score, a random tie-breaking float, the amino acid-level
                scores, and the predicted tokens is stored.
        """
        
        for i in range(len(finished_beams)):
            if not finished_beams[i]:
                continue
            spec_idx = i // self.n_beams  # batch index
            pred_peptide = tokens[i]  # [L]
            # Don't cache this peptide if it was already predicted previously.
            if any(
                equal_ignore_tokens(pred_cached[-1], pred_peptide, self.extra_ignored_ids)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            finalized_hypos = self.finalized_hypos(
                step, 
                pred_peptide,
                scores[i],
                ignore_ids=self.extra_ignored_ids
            )
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (
                    beam_fits_precursor[i], 
                    finalized_hypos["score"],
                    finalized_hypos["positional_scores"],
                    finalized_hypos["sequence"],
                    finalized_hypos["tokens"],
                ),
            )

    def equal_ignore_pad(
        self,
        a: Union[torch.Tensor, np.ndarray, list],
        b: Union[torch.Tensor, np.ndarray, list],
        pad_token: int
    ) -> bool:
        # 1) If one of them is a torch.Tensor, grab its device & dtype
        if isinstance(a, torch.Tensor):
            ref_device, ref_dtype = a.device, a.dtype
        elif isinstance(b, torch.Tensor):
            ref_device, ref_dtype = b.device, b.dtype
        else:
            # neither is tensor → default to CPU long
            ref_device, ref_dtype = torch.device('cpu'), torch.long

        # 2) Coerce both to torch.Tensor on (ref_device, ref_dtype)
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=ref_device, dtype=ref_dtype)
            else:
                # covers np.ndarray or list
                return torch.tensor(x, device=ref_device, dtype=ref_dtype)

        a_t = to_tensor(a)
        b_t = to_tensor(b)

        # 3) Strip trailing pads
        def strip(x: torch.Tensor) -> torch.Tensor:
            nonpad = (x != pad_token).nonzero(as_tuple=True)[0]
            if nonpad.numel() == 0:
                return x.new_empty((0,), dtype=x.dtype)
            last = nonpad.max().item()
            return x[: last + 1]

        a_str = strip(a_t)
        b_str = strip(b_t)

        # 4) Compare shapes & values
        return a_str.shape == b_str.shape and bool(torch.equal(a_str, b_str))
 
    def _finish_beams(
            self,
            tokens:   torch.Tensor,  # [B*S, L]
            scores:   torch.Tensor,  # [B*S, L]
            prev_decoder_out: torch.Tensor,  # [B*S, L]
            precursors:   torch.Tensor,  # [B*S, 3]  
            beam_size: int,
            step: int,
            ignore_ids: Optional[Iterable[int]] = None,
    ):
        """
        终止那些‘卡在循环’里的 beams。

        Args:
            tokens:        [B*S, L] 当前 step 的所有 beam 输出
            scores:        [B*S, L] 当前 step 的所有 beam 输出分数
            prev_tokens:   [B*S, L] 上一 step 的所有 beam 输出
            precursors:    [B*S, …] （可以用来做质量检查）
            step:          int       当前序列长度或位置
            beam_size:     int       每个样本的 beam 数 S

        Returns:
            finished_beams:      [B*S] BoolTensor,标记哪些 beam 该终止（循环不变）
            beam_fits_precursor: [B*S] BoolTensor,通过质量检查的 mask(示例全 True)
        """
        prev_out_token = prev_decoder_out.output_tokens  # [B*S, L]
        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.encoder.device)
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        discard = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        has_neginf = torch.isneginf(scores).any(dim=1)  # [B*S], True if any element is -inf

        # 2) mark those beams discarded
        discard[has_neginf] = True
        # 总行数 = B * S
        total_beams, L1 = tokens.size()
        # print(f"tokens size: {tokens.size()}")
        _,  L2  = prev_out_token.size()
        # print(f"prev_out_token size: {prev_out_token.size()}")
        S = beam_size
        # 计算 batch 大小 B
        B = total_beams // S

        # 恢复成 [B, S, L]
        toks = tokens.view(B, S, L1)
        # print(tokens.size())
        # print(f"prev_out_token size: {prev_out_token.size()}")
        prev = prev_out_token.view(B, S, L2)
        # 2) 在尾部补齐到同样长度 L_max
        L_max = max(L1, L2)
        if L1 < L_max:
            pad = toks.new_full((B, S, L_max - L1), self.decoder.pad)
            toks = torch.cat([toks, pad], dim=2)
        if L2 < L_max:
            pad = prev.new_full((B, S, L_max - L2), self.decoder.pad)
            prev = torch.cat([prev, pad], dim=2)
        # 3) 交叉比较：toks[b,i] vs prev[b,j]
        #    unsqueeze/广播后得 [B, S, S, L_max]
        eq4d = (toks.unsqueeze(2) == prev.unsqueeze(1))
        # 对最后一维全等 → [B, S, S]
        eq = eq4d.all(dim=-1)
        # 4) 只要同一 sample b 中，对应 s 任意 t 相等，就判循环
        loop_mask = eq.any(dim=2)  # [B, S]
        # 展平回 [B*S]
        finished_beams = loop_mask.view(-1).to(finished_beams.device)

        # 2. 质量检查
        # —— 1. 构造非 padding 掩码 —— 
        ignore_ids = set(ignore_ids) if ignore_ids is not None else set()
        if ignore_ids:
            pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for pid in ignore_ids:
                pad_mask |= tokens.eq(pid)
            pad_mask |= tokens.eq(self.decoder.unk)
            non_ignore = ~pad_mask
            non_pad = tokens.ne(self.decoder.pad)
            
        else:
            non_pad = torch.ones_like(tokens, dtype=torch.bool)

   
        for i in range(total_beams):
            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 2]
            # 提取第 i 行的有效 token 序列
            tok_row = tokens[i]        # [T]
            token_ids = tok_row[non_ignore[i]]     # 真实 token id 列表
            sequence = self.decoder.detokenize(token_ids)
            # 计算该序列的质量
            calc_mz = self.peptide_mass_calculator.mass(
                seq=sequence,
                charge=precursor_charge
            )
            delta_mass_ppm = [
                _calc_mass_error(
                    calc_mz,
                    precursor_mz,
                    precursor_charge,
                    isotope,
                )
                for isotope in range(
                    self.isotope_error_range[0],
                    self.isotope_error_range[1] + 1,
                )
            ]
            matches_precursor_mz = any(
                abs(d) < self.precursor_mass_tol for d in delta_mass_ppm
            )

            if matches_precursor_mz:
                beam_fits_precursor[i] = True
        
        # 如果一个样本的任意一条 beam 通过了前体质量检查，那么将该样本所有的 beam 都标记为已终止（finished_beams=True）
        fits = beam_fits_precursor.view(B, S)   # [B, S]
        # 3) 对每个样本 b，只要任意一条 beam fits 就标记该样本所有 beams
        #    构造一个 [B, S] 的 mask：对于符合条件的 b 行，整行都是 True
        batch_done = fits.any(dim=1)            # [B], 每个样本是否至少有一个 fits=True
        # expand 到 [B, S]
        batch_done_expanded = batch_done.unsqueeze(1).expand(-1, S)  # [B, S]
        batch_done_flat = batch_done_expanded.contiguous().view(-1)

        finished_beams |= batch_done_flat

        return finished_beams, beam_fits_precursor, discard

        
    # def replace_active_beams(
    #     tokens: torch.Tensor,           # [B*S, T] 原 tokens
    #     scores: torch.Tensor,           # [B*S, T] 原 scores 或聚合后[ B*S ]
    #     finished_beams: torch.BoolTensor,# [B*S] 已完成标记
    #     new_seqs: torch.LongTensor,     # [N_active, S, T]
    #     new_scores: torch.Tensor,       # [N_active, S] 或 [N_active, S, T]
    #     beam_size: int
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     用 new_seqs/new_scores 更新 tokens/scores 中所有 active beams，
    #     并保证对每个样本只保留 beam_size 条最优 beam。
    #     """
    #     B = tokens.size(0) // beam_size
    #     T = tokens.size(1)
    #     device = tokens.device

    #     # 1) 找到 active beams 的索引和对应样本 id
    #     active = ~finished_beams                # [B*S]
    #     all_indices = torch.arange(B * beam_size, device=device)
    #     active_idx = all_indices[active]        # [N_active]
    #     sample_id = active_idx // beam_size     # [N_active], 每个 active beam 属于哪个 sample

    #     # 2) 把 new_seqs/scores 从 [N_active, S, ...] 摊平为 [N_active*S, ...]
    #     flat_seqs   = new_seqs.reshape(-1, T)           # [N_active*S, T]
    #     flat_scores = new_scores.reshape(-1, new_scores.size(-1))  # [N_active*S, T] 或 [N_active*S]

    #     # 3) 为每个 sample 分组，并择优
    #     #    我们将收集每个 sample 的所有 (flat_seqs, flat_scores, flat_idx)
    #     best_seqs  = torch.empty_like(tokens)
    #     best_scores= torch.empty_like(scores)

    #     for b in range(B):
    #         # 找到属于 sample b 的所有扁平 candidates
    #         mask_b = (sample_id == b).nonzero(as_tuple=False).view(-1)
    #         if mask_b.numel() == 0:
    #             # 如果这个 sample 没 active beam，直接填原来那些 finished_beams
    #             start = b * beam_size
    #             best_seqs[start:start+beam_size]   = tokens[start:start+beam_size]
    #             best_scores[start:start+beam_size] = scores[start:start+beam_size]
    #             continue

    #         # mask_b 中每个 i 对应 flat_seqs[i*S:(i+1)*S]
    #         # 实际上 greedy_k_best_on_masks 已经按 beam_size 输出了最优 S 条，
    #         # 所以第 i 条 active beam 展开后 flat_seqs 对应的 [i*S : i*S+S] 就是它的 S 条候选。
    #         cand_seqs   = flat_seqs[mask_b.repeat_interleave(beam_size)*beam_size + torch.arange(beam_size, device=device)]
    #         cand_scores = flat_scores[mask_b.repeat_interleave(beam_size)*beam_size + torch.arange(beam_size, device=device)]

    #         # 4) 选出分数最高的 beam_size 条
    #         #    先把 cand_scores 聚合为总分（如果是 per-pos，需要 sum）
    #         if cand_scores.dim() == 2:
    #             # cand_scores [N_can, T] -> 总分
    #             total_scores = cand_scores.sum(dim=1)
    #         else:
    #             total_scores = cand_scores  # 已经是一维 [N_can]

    #         topk_scores, topk_idx = torch.topk(total_scores, beam_size, largest=True)
    #         chosen_seqs   = cand_seqs[topk_idx]
    #         chosen_scores = cand_scores[topk_idx]

    #         # 5) 写回 flat tokens/scores
    #         base = b * beam_size
    #         best_seqs[base:base+beam_size]   = chosen_seqs
    #         best_scores[base:base+beam_size] = chosen_scores

    #     return best_seqs, best_scores
    

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor, 
        prev_output_tokens: Optional[torch.Tensor] = None
    ) -> List[List[Dict]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        flag = prev_output_tokens is not None
        memory, memory_key_padding_mask = self.encoder(spectra)
        # Prepare mass and charge
        masses = self.decoder.mass_encoder(precursors[:, None, 0])
        charges = self.decoder.charge_encoder(precursors[:, 1].int() - 1)
        precursors_embedding = masses + charges[:, None, :]
        bsz = spectra.size(0)
        sent_idxs = torch.arange(bsz)
        if prev_output_tokens is None:
            prev_decoder_out = self.initialize_output_tokens(memory)
            prev_output_tokens = prev_decoder_out.output_tokens.clone()
        else:
            prev_decoder_out = self.initialize_output_tokens(memory)
            prev_decoder_out = prev_decoder_out._replace(
                output_tokens=prev_output_tokens,
                output_scores=torch.zeros_like(prev_output_tokens).float(),
            )

        finalized = [[] for _ in range(bsz)]
        for step in range(self.max_iter + 1):
            # decoder_options = {
            #     "eos_penalty": self.eos_penalty,
            #     "max_ratio": self.max_ratio,
            # }
            prev_decoder_out = prev_decoder_out._replace(
                step=step,
                max_step=self.max_iter + 1,
            )
            # if flag:
            #     print(f"Step {step}, prev_output_tokens: {prev_decoder_out.output_tokens}")
            #     print(f"Step {step}, prev_output_scores: {prev_decoder_out.output_scores}")

            decoder_out = self.decoder.forward_decoder(
                step, self.encoder, prev_decoder_out, precursors_embedding, memory, memory_key_padding_mask
            )

            # if step == 0:
            #     decoder_out.output_tokens[0, 1] = self.decoder.unk
            #     decoder_out.output_tokens[0, 2] = 8
            #     decoder_out = decoder_out._replace(
            #         output_tokens=decoder_out.output_tokens,  # [B, 1]
            #     )

            # terminate if there is a loop
            # terminated shape: (bsz, ) out_tokens shape: (bsz, L), out_scores shape: (bsz, L, V)
            terminated, out_tokens, out_scores, out_attn = is_a_loop(
                prev_output_tokens,
                decoder_out.output_tokens,
                decoder_out.output_scores,
                a = None,
                padding_idx= self.decoder.pad,
            )

            # force remove low confidence tokens if terminated, and turn terminated to False
            # terminated, out_tokens, out_scores = self.remove_low_confidence(
            #     terminated,
            #     out_tokens,
            #     out_scores,
            #     precursor_mz=precursors[:, 2],
            #     precursor_charge=precursors[:, 1],
            #     ignore_ids=self.extra_ignored_ids
            # )

            # for i in range(out_tokens.size(0)):
            #     pad_mask = torch.zeros_like(out_tokens[i], dtype=torch.bool)
                
            #     for pid in self.extra_ignored_ids:
            #         pad_mask |= out_tokens[i].eq(pid)
            #     non_ignore = ~pad_mask
            #     tokens = out_tokens[i][non_ignore]  # 真实 token id 列表
            #     seq = "".join(self.decoder.detokenize(tokens))
            #     real_out_scores = out_scores[i][non_ignore]  # 真实 token 分数列表
            #     print(f"Step {step}, len: {len(seq)}, terminated: {terminated[i].item()}, tokens: {seq}, scores: {real_out_scores}")
            
            decoder_out = decoder_out._replace(
                output_tokens=out_tokens,
                output_scores=out_scores,
                attn=out_attn,
            )

            if step == self.max_iter:  # reach last iteration, terminate
                terminated.fill_(1)

            # collect finalized sentences
            finalized_idxs = sent_idxs[terminated.to(sent_idxs.device)]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]

            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    self.finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        ignore_ids=self.extra_ignored_ids
                    )
                ]

            # check if all terminated
            if terminated.sum() == terminated.size(0):
                break
            
            # for next step
            not_terminated = ~terminated
            prev_decoder_out = decoder_out._replace(
                output_tokens=decoder_out.output_tokens[not_terminated],
                output_scores=decoder_out.output_scores[not_terminated],
            )

            precursors_embedding, memory, memory_key_padding_mask = self.encoder.reorder_encoder_out(
                precursors_embedding, memory, memory_key_padding_mask, not_terminated.nonzero(as_tuple=False).squeeze()
            )
            sent_idxs = sent_idxs[not_terminated.to(sent_idxs.device)]
            prev_output_tokens = prev_decoder_out.output_tokens.clone()

        return finalized
    
    def remove_low_confidence(
        self,
        terminated: torch.BoolTensor,        # [bs]
        prev_out_token: torch.LongTensor,    # [bs, T]
        prev_out_score: torch.FloatTensor,   # [bs, T]
        precursor_mz: torch.FloatTensor,     # [bs]
        precursor_charge,  # [bs] 或 int
        ignore_ids: Optional[Iterable[int]] = None
    ) -> Tuple[torch.BoolTensor, torch.LongTensor, torch.FloatTensor]:
        """
        对已 terminated 的样本：
        - 计算其 token 序列的质量 calc_mz;
        - 如果 |calc_mz - precursor_mz| > self.precursor_mass_tol,
            就在非 ignore_ids 的 token 中找到最低置信度那个,
            从序列中物理删除（左侧压缩、末尾 pad)并将 terminated 复位为 False
        - 否则该行不变。
        其他未 terminated 行不变。始终返回 [bs,T] 形状的 Tensor。
        """
        bs, T = prev_out_token.shape
        device = prev_out_token.device

        # —— 0. 早退 —— 
        if not terminated.any():
            return terminated, prev_out_token, prev_out_score

        # —— 1. 构造非 padding 掩码 —— 
        ignore_ids = set(ignore_ids) if ignore_ids is not None else set()
        if ignore_ids:
            pad_mask = torch.zeros_like(prev_out_token, dtype=torch.bool)
            for pid in ignore_ids:
                pad_mask |= prev_out_token.eq(pid)
            non_ignore = ~pad_mask
            non_pad = prev_out_token.ne(self.decoder.pad)
            
        else:
            non_pad = torch.ones_like(prev_out_token, dtype=torch.bool)

        # —— 2. 选出“要清理”的行 —— 
        # 条件：terminated 且 质量偏差超过 tol
        # 先克隆输出
        new_tokens = prev_out_token.clone()
        new_scores = prev_out_score.clone()
        new_terminated = terminated.clone()

        # 仅遍历那些 terminated 行
        idx = torch.nonzero(terminated, as_tuple=True)[0]
        for i in idx:
            # 提取第 i 行的有效 token 序列
            tok_row = prev_out_token[i]        # [T]
            valid = non_pad[i]                 # [T]
            tokens = tok_row[non_ignore[i]]     # 真实 token id 列表
            seq = "".join(self.decoder.detokenize(tokens))
            # 计算该序列的质量
            calc_mz = self.peptide_mass_calculator.mass(
                seq=seq,
                charge=precursor_charge[i]
            )
            delta_mass_ppm = [
                _calc_mass_error(
                    calc_mz,
                    precursor_mz[i],
                    precursor_charge[i],
                    isotope,
                )
                for isotope in range(
                    self.isotope_error_range[0],
                    self.isotope_error_range[1] + 1,
                )
            ]
            matches_precursor_mz = any(
                abs(d) < self.precursor_mass_tol for d in delta_mass_ppm
            )

            if matches_precursor_mz:
                continue  # 质量匹配,跳过该行

            # 否则：找到最低置信度的非 padding token 的 index
            sc_row = prev_out_score[i]  # [T]
            # 给非 valid 或 pad 的位置设置 +∞,保证不会被选为最小
            inf = torch.tensor(float("inf"), device=device)
            sc_for_min = torch.where(valid, sc_row, inf)  # [T]
            idx_min = torch.argmin(sc_for_min)            # 最低置信度位置
# "".join(self.decoder.detokenize(tokens))
            print(f"Remove low confidence token: {self.decoder.detokenize(tok_row[idx_min].view(1))}, score {sc_row[idx_min].item()}")
            # 物理删除该位置：压缩左侧,末尾 pad
            keep_mask = torch.ones(T, dtype=torch.bool, device=device)
            keep_mask[idx_min] = False                   # 标记要删的位置
            keep_mask &= valid                           # 只在原 valid 范围删

            kept_tokens = tok_row[keep_mask]              # [L_i]
            kept_scores = sc_row[keep_mask]               # [L_i]
            L = kept_tokens.size(0)

            # 重置整行
            new_tokens[i].fill_(self.decoder.pad)
            new_scores[i].zero_()
            # 写回保留部分
            new_tokens[i, :L] = kept_tokens
            new_scores[i, :L] = kept_scores

            # 复位 terminated
            new_terminated[i] = False

        return new_terminated, new_tokens, new_scores
    
    # def remove_low_confidence(
    #     self,
    #     terminated: torch.BoolTensor,        # [bs],标记哪些样本已经终止
    #     prev_out_token: torch.LongTensor,    # [bs, T],上轮生成的 token id
    #     prev_out_score: torch.FloatTensor,   # [bs, T],上轮生成的 token 对应的置信度（概率 / 分数）
    #     threshold: float = 0.1,              # 置信度阈值,低于它的 token 要被清理
    #     ignore_ids: Optional[Iterable[int]] = None  # 要当作“padding”跳过的 id 列表
    # ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
      
    #     """
    #         只对 avg_score >= threshold 且 terminated == True 的行
    #         删除低置信度 (< threshold 且非 ignore_ids) 的 token,
    #         并将剩余 token 压缩到左侧、末尾填 pad_id；score 同理、空位填 0。
    #         同时把这些样本的 terminated 复位为 False。
    #         如果 terminated 全 False,直接返回原始 Tensor。
    #     """
    
    #     # —— 0. 早退 —— 
    #     if not terminated.any():
    #         return terminated, prev_out_token, prev_out_score

    #     # —— 1. 非 padding 掩码 —— 
    #     ignore_ids = set(ignore_ids) if ignore_ids is not None else set()
    #     if ignore_ids:
    #         pad_mask = torch.zeros_like(prev_out_token, dtype=torch.bool)
    #         for pid in ignore_ids:
    #             pad_mask |= prev_out_token.eq(pid)
    #         non_pad = ~pad_mask
    #     else:
    #         non_pad = torch.ones_like(prev_out_token, dtype=torch.bool)

    #     # 2. 标记需清理的行：
    #     #    在 terminated 且 存在至少一个 非 padding 且 score >= threshold 的位置 时触发
    #     high_conf_exist = (prev_out_score < threshold) & non_pad        # [bs, T]
    #     do_clean = terminated & high_conf_exist.any(dim=1)              # [bs]

    #     if not do_clean.any():
    #         return terminated, prev_out_token, prev_out_score, 

    #     # —— 4. 复制并复位 terminated —— 
    #     new_tokens = prev_out_token.clone()
    #     new_scores = prev_out_score.clone()
    #     new_terminated = terminated.clone()
    #     new_terminated[do_clean] = False

    #     # —— 5. 仅对要清理的行做压缩＋填充 —— 
    #     idx = torch.nonzero(do_clean, as_tuple=True)[0]  # 需要清理的行索引
    #     for i in idx:
    #         tok_row = prev_out_token[i]    # [T]
    #         sc_row = prev_out_score[i]     # [T]
    #         valid = non_pad[i]             # [T]

    #         # 保留：非 padding 且 score > threshold
    #         keep_stay = valid & (sc_row > threshold)
    #         kept_tokens = tok_row[keep_stay]    # [L_i]
    #         kept_scores = sc_row[keep_stay]     # [L_i]
    #         L = kept_tokens.size(0)

    #         # 先把整行设成 pad / 0
    #         new_tokens[i].fill_(self.decoder.pad)
    #         new_scores[i].zero_()
    #         # 再把保留部分写回左侧
    #         new_tokens[i, :L] = kept_tokens
    #         new_scores[i, :L] = kept_scores

    #     return new_terminated, new_tokens, new_scores, 
                    



    def finalized_hypos(self, step, prev_out_token, prev_out_score, ignore_ids=None):

        if ignore_ids is None:
            # 默认只丢弃 pad
            ignore_ids = [self.decoder.pad]
        # 转成 tensor 放到同设备
        ignore = torch.tensor(ignore_ids, device=prev_out_token.device, dtype=prev_out_token.dtype)

        # 比较并聚合：先扩成 (L, len(ignore)),再 any(dim=1)
        mask = ~(prev_out_token.unsqueeze(-1) == ignore).any(dim=1)
        # mask[i]=True 表示保留该 token

        # 2) 用 mask 取出所有“有效”token
        tokens = prev_out_token[mask]

        # 3) 如果有分数,一起按照同样的 mask 过滤并计算平均分
        if prev_out_score is None or len(prev_out_score[mask]) == 0:
            scores, score = None, None
        else:
            scores = prev_out_score[mask]
            score  = scores.mean()

        sequence = "".join(self.decoder.detokenize(tokens))
        # 5) 把这些信息打包返回
        return {
            "steps":            step,       # 第几步/轮生成
            "tokens":           prev_out_token,     # token id 序列
            "sequence":         sequence,     # 去 pad 之后的 token id 序列
            "positional_scores": scores,    # 每个 token 的分数（可选）
            "score":            score,      # 平均分（可选）
        }    
    

    def initialize_output_tokens(self, memories: torch.Tensor):
        """
        初始化输出 token 和对应的 score 张量：
        - output_tokens: LongTensor[B, 2],第 0 列填 BOS,第 1 列填 EOS
        - output_scores: Tensor[B, 2],和 memories 同 dtype&device,全部置 0
        """
        bsz = memories.size(0)
        device = memories.device
        dtype = memories.dtype

        # 1) output_tokens: 长度为 2,dtype 用 Long（整型）,device 同 encoder 输出
        initial_output_tokens = torch.zeros(bsz, 2, dtype=torch.long, device=device)
        initial_output_tokens[:, 0] = self.decoder.bos  # BOS ID
        initial_output_tokens[:, 1] = self.decoder.eos  # EOS ID

        # 2) output_scores: shape 同 tokens,但 dtype&device 跟 memories 保持一致
        initial_output_scores = torch.zeros(bsz, 2, dtype=dtype, device=device)

    
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )


    def _get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.decoder.vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores[:, step, :, :], "B V S -> B (V S)"
        )

        # Find all still active beams by masking out terminated beams.
        active_mask = (
            ~finished_beams.reshape(batch, beam).repeat(1, vocab)
        ).float()
        # Mask out the index '0', i.e. padding token, by default.
        # FIXME: Set this to a very small, yet non-zero value, to only
        # get padding after stop token.
        active_mask[:, :beam] = 1e-8

        # Figure out the top K decodings.
        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, step, :] = torch.tensor(v_idx)
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return tokens, scores

    def _get_top_peptides(
        self,
        pred_cache: Dict[int, List[Tuple[bool, float, np.ndarray, str, torch.Tensor]]],
        top_match: int
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        从 pred_cache 中为每个谱图提取 top_match 条候选肽段：
        pred_cache: key -> list of (score, pos_score, aa_scores, pred_tokens)
        top_match: 每个谱图要返回的候选数量

        返回：
        A list of length = len(pred_cache)，其中每一项又是一个列表，
        列表项格式为 (pep_score, aa_scores_array, peptide_str)。
        """
        all_results: List[List[Tuple[float, np.ndarray, str]]] = []

        for peptides in pred_cache.values():
            # peptides 的类型是
            # List[ Tuple[
            #     float,      # pep_score
            #     float,      # positional_score
            #     torch.Tensor# pred_tokens 向量
            # ] ]

            if peptides:
                # 取出得分最高的 top_match 条（按第 1 项 pep_score 排序）
                topk = heapq.nlargest(top_match, peptides)
                formatted = []
                for _, pep_score, aa_scores, peptide_str, _ in topk:
                    formatted.append((pep_score, aa_scores, peptide_str))
                all_results.append(formatted)
            else:
                all_results.append([])

        return all_results

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        encoder, key_mask = self.encoder(spectra)
        step = self.trainer.global_step
        prob = 0.5 - 0.5 * (step / 287000)
        decoder_result = self.decoder(sequences, precursors, encoder, key_mask, prob=prob)
        word_ins = decoder_result.get("word_ins", None)
        target_enc = word_ins.get("word_enc", None)
        # rank_result = self.ranker(
        #     encoder, target_enc, key_mask
        # )
        # decoder_result["rank"] = {"logit": rank_result}
        return decoder_result

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        outputs = self._forward_step(*batch)
        # logits = outputs["word_ins"].get("out")
        # target_tokens = outputs["word_ins"].get("tgt")
        # edge_mask = outputs["word_ins"].get("mask", None)
        # # 将 logits 和真实目标 target_tokens 都 flatten
        # logits_flat = logits.view(-1, logits.size(-1))        # (B*T, V)
        # targets_flat = target_tokens.view(-1)                  # (B*T,)
        # mask_flat = edge_mask.view(-1)                         # (B*T,)

        # # 只挑出 edge_mask 为 True 的位置
        # selected_logits = logits_flat[mask_flat]               # (N_edge, V)
        # selected_targets = targets_flat[mask_flat]             # (N_edge,)

        # loss = self.celoss(selected_logits, selected_targets)
        # losses = [{"name": "word_ins-loss", "loss": loss, "factor": 1.0}]
        # rank = outputs.get("rank", None)
        # del outputs["rank"]

        losses, nll_loss = [], []
        for obj in outputs:
            _losses = self._compute_loss(
                outputs[obj].get("out"),
                outputs[obj].get("tgt"),
                outputs[obj].get("mask", None),
                outputs[obj].get("ls", 0.0),
                name=obj + "-loss",
                factor=outputs[obj].get("factor", 1.0),
                weight=outputs[obj].get("weight", None),
            )
            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        # for l in losses:
        #     if l["name"] == "word_ins-loss":
        #         seq_loss = l["seq_loss"]
        #         temp = 0.5
        #         sim_tgt = torch.exp(-seq_loss/ temp)
        #         sim_tgt = sim_tgt.to(rank["logit"].device).float()
        #         loss_rank = F.binary_cross_entropy_with_logits(rank["logit"], sim_tgt, reduction="mean")
                # loss_rank = F.binary_cross_entropy(rank["logit"], sim_tgt)


        loss = sum(l["loss"] for l in losses)
        # loss += 0.3 * loss_rank  # rank loss
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
    
        # logging_output = {
        #     "loss": loss.data,
        #     "nll_loss": nll_loss.data,
        # }

        # record mask-ins, word-ins, delete loss seperately
        # for l in losses:
        #     logging_output[l["name"]] = (
        #         item(l["loss"].data / l["factor"])
        #     )
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # record mask-ins, word-ins, delete loss seperately
        for l in losses:
            self.log(
                f"{mode}_{l['name']}",
                l["loss"].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # self.log(
        #     f"{mode}_rank_loss",
        #     loss_rank.detach(),
        #     on_step=False,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        return loss
        # pred, truth = self._forward_step(*batch)
        # pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
        # if mode == "train":
        #     loss = self.celoss(pred, truth.flatten())
        # else:
        #     loss = self.val_celoss(pred, truth.flatten())
        # self.log(
        #     f"{mode}_CELoss",
        #     loss.detach(),
        #     on_step=False,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        # return loss


    # def _compute_loss(
    #     self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0, weight=None
    # ):
    #     """
    #     outputs: batch x len x d_model
    #     targets: batch x len
    #     masks:   batch x len

    #     policy_logprob: if there is some policy
    #         depends on the likelihood score as rewards.
    #     """

    #     def mean_ds(x: torch.Tensor, dim=None) -> torch.Tensor:
    #         return (
    #             x.float().mean().type_as(x)
    #             if dim is None
    #             else x.float().mean(dim).type_as(x)
    #         )

    #     if masks is not None:
    #         outputs, targets = outputs[masks], targets[masks]

    #     if masks is not None and not masks.any():
    #         nll_loss = torch.tensor(0)
    #         loss = nll_loss
    #     else:
    #         logits = F.log_softmax(outputs, dim=-1)

    #         losses = F.nll_loss(logits, targets.to(logits.device), reduction="none", weight=weight)

    #         nll_loss = mean_ds(losses)
    #         if label_smoothing > 0:
    #             loss = (
    #                 nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
    #             )
    #         else:
    #             loss = nll_loss

    #     loss = loss * factor
    #     return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _compute_loss(
        self,
        outputs: torch.Tensor,                    # (B, L, V)
        targets: torch.Tensor,                    # (B, L)
        masks:   Optional[torch.BoolTensor] = None,  # (B, L), True=valid
        label_smoothing: float = 0.0,
        name:    str   = "loss",
        factor:  float = 1.0,
        weight:  Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute NLL loss per-token, aggregate to per-sequence and overall.
        Returns dict with:
        - 'loss':    overall scalar loss (applies label_smoothing & factor)
        - 'nll_loss': overall mean NLL across the batch
        - 'seq_loss': tensor of shape (B,) with per-sequence mean NLL
        - 'factor':  the factor multiplier
        """
        B, L, V = outputs.size()

        # 1) log-probs and flatten for per-token loss
        log_probs     = F.log_softmax(outputs, dim=-1)       # (B, L, V)
        log_probs_flat= log_probs.view(-1, V)                # (B*L, V)
        targets_flat  = targets.reshape(-1)                     # (B*L,)

        # 2) per-token NLL (no reduction)
        losses_flat = F.nll_loss(
            log_probs_flat,
            targets_flat.to(log_probs_flat.device),
            reduction="none",
            weight=weight
        )                                                    # (B*L,)

        # 3) reshape back to (B, L)
        losses = losses_flat.view(B, L)                      # (B, L)

        # 4) zero out PAD positions
        if masks is not None:
            losses = losses.masked_fill(~masks, 0.0)     # PAD→0

        # 5) per-sequence NLL: mean over valid tokens
        if masks is not None:
            valid_counts = (masks).sum(dim=1).clamp(min=1).float()  # (B,)
            seq_nll      = losses.sum(dim=1) / valid_counts        # (B,)
        else:
            seq_nll      = losses.mean(dim=1)                       # (B,)

        # 6) overall NLL is mean over all sequences
        overall_nll = seq_nll.mean()                         # scalar

        # 7) apply label-smoothing if needed
        if label_smoothing > 0:
            # smoothing term: mean of negative log-probs over all tokens
            smooth_term = -log_probs.mean()
            overall_loss = overall_nll * (1 - label_smoothing) + smooth_term * label_smoothing
        else:
            overall_loss = overall_nll

        # 8) apply scaling factor
        overall_loss = overall_loss * factor

        return {
            "name":     name,
            "loss":     overall_loss,
            "nll_loss": overall_nll,
            "seq_loss": seq_nll,
            "factor":   factor
        }

    def _is_mass_match(self, seq: str, obs_mz: float, charge: int) -> bool:
        """
        Check if the calculated mass matches the observed mass within the
        allowed tolerance.
        Parameters
        ----------
        seq : str
            The peptide sequence.
        obs_mz : float
            The observed m/z of the precursor.
        charge : int
            The charge state of the precursor.
        Returns
        -------
        bool
            True if the calculated mass matches the observed mass within the
            allowed tolerance, False otherwise.
        """
        calc_mz = self.peptide_mass_calculator.mass(seq=seq, charge=charge)
        delta_mass_ppm = [
            _calc_mass_error(
                calc_mz,
                obs_mz,
                charge,
                isotope,
            )
            for isotope in range(
                self.isotope_error_range[0],
                self.isotope_error_range[1] + 1,
            )
        ]
        return any(
            abs(d) < self.precursor_mass_tol for d in delta_mass_ppm
        )

    def _is_mass_match_list(self, seq_list: str, obs_mz: float, charge: int) -> list:
        return [self._is_mass_match(seq, obs_mz, charge) for seq in seq_list]
    
    def dp_path_search_topB(
        self,
        prob,         # (word_num, K)  本步每个候选的得分，与 aa_idxs 对齐
        aa_idxs,      # (word_num, K)  本步可选的全局AA索引（已保证合法）
        aa_mass,      # (V,) 或 (max_AA_id+1,) 每个AA的质量
        premass: float,
        grid_size: float,
        tol: float,
        B: int = 10,
        device: torch.device | str | None = None,
    ):
        """
        时间优先 DP（只新增一个AA，无ε/重复检查；整条路径直写）。
        升级为：每个 (l, t) 保留 Top-B 条路径。
        返回：
            dp:      (length, cell_num, B)  每格前B得分（-inf 表示空位）
            dp_mass: (length, cell_num, B)  对应质量（未命中为空位时内容无效）
            paths:   (length, cell_num, B, length)  对应整条AA索引（空位填-1）
        """
        device = device or prob.device
        word_num, K = prob.shape
        length = word_num + 1
        cell_num = int(premass / grid_size) + 1
        last_cell = cell_num - 1

        # 初始化
        dp      = torch.full((length, cell_num, B), float("-inf"), dtype=torch.float32, device=device)
        dp_mass = torch.zeros((length, cell_num, B), dtype=torch.float32, device=device)
        paths   = torch.full((length, cell_num, B, length), -1, dtype=torch.int32, device=device)

        # 起点：空序列位于 t=0，给它一个占位 beam（b=0）
        dp[0, 0, 0] = 0.0
        dp_mass[0, 0, 0] = 0.0
        # paths[0,0,0,:] 全为 -1，表示空序列

        # 一个小工具：把候选插入 (l,t) 的 top-B 中
        def try_insert(l, t, cand_score, cand_mass, prev_prefix, cur_idx_pos, cur_aa_id):
            """
            prev_prefix: tensor(int32, shape=(length,))，上一层前缀路径（已-1填充）
            cur_idx_pos: 当前要写入的位置（0..l-1）
            cur_aa_id:   本步选择的全局AA id
            """
            # 找出当前最差（最小）分数所在的 beam 槽位
            scores = dp[l, t]   # (B,)
            min_idx = torch.argmin(scores)
            min_val = scores[min_idx]

            if cand_score > min_val:
                # 覆盖写入
                dp[l, t, min_idx] = cand_score
                dp_mass[l, t, min_idx] = cand_mass

                # 复制前缀 + 追加当前AA
                # 注意 prev_prefix 是长度=length 的向量，只有 0..(l-2) 可能有效
                # 这里直接整条拷贝更简单；再把 l-1 写成当前AA
                paths[l, t, min_idx] = prev_prefix
                paths[l, t, min_idx, cur_idx_pos] = cur_aa_id

        # DP 主循环
        for l in range(1, length):
            ids_step  = aa_idxs[l - 1]      # (K,)
            prob_step = prob[l - 1]         # (K,)

            for t in range(1, cell_num):
                if l == 3 and t == 243:
                    print("debug")

                # 枚举该步所有候选AA
                for j in range(K):
                    aa_id = int(ids_step[j].item())
                    w_i = float(aa_mass[aa_id].item())  # 该AA质量（>0）

                    # 反推上一个 cell：tp 与 tp+1（两次尝试覆盖边界）
                    base = t * grid_size - w_i
                    tp = int(base / grid_size)

                    for _ in range(2):
                        if 0 <= tp < cell_num:
                            # 来自上一层的 top-B 都尝试扩展
                            prev_scores = dp[l - 1, tp]    # (B,)
                            # 只处理有限分的 beam
                            finite_mask = torch.isfinite(prev_scores)
                            if torch.any(finite_mask):
                                idxs = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)
                                for b_prev in idxs.tolist():
                                    prev_score = float(prev_scores[b_prev].item())
                                    prev_mass  = float(dp_mass[l - 1, tp, b_prev].item())
                                    new_mass   = prev_mass + w_i

                                    # 非末cell：命中该bin；末cell：premass±tol
                                    if t != last_cell:
                                        in_bin = (t * grid_size <= new_mass) and (new_mass < (t + 1) * grid_size)
                                    else:
                                        in_bin = (premass - tol <= new_mass <= premass + tol)

                                    if in_bin:
                                        cand = prev_score + float(prob_step[j].item())

                                        # 取出上一条前缀路径做复制（整条）
                                        prev_prefix = paths[l - 1, tp, b_prev]  # (length,)
                                        try_insert(
                                            l, t,
                                            cand_score=cand,
                                            cand_mass=new_mass,
                                            prev_prefix=prev_prefix,
                                            cur_idx_pos=l - 1,
                                            cur_aa_id=aa_id
                                        )
                        tp += 1  # 第二次尝试：tp+1

        # 为了方便下游直接拿到“最终 top-B”，这里把最后一格 (length-1, last_cell, :)
        # 的前B结果按分数降序排一下（可选）
        final_scores = dp[length - 1, last_cell]  # (B,)
        order = torch.argsort(final_scores, descending=True)
        dp[length - 1, last_cell]     = final_scores[order]
        dp_mass[length - 1, last_cell] = dp_mass[length - 1, last_cell][order]
        paths[length - 1, last_cell]   = paths[length - 1, last_cell][order]

        return dp, dp_mass, paths
    
    def dp_path_search(   
        self,
        prob,         # (word_num, K)  本步每个候选的得分，与 aa_idxs 对齐
        aa_idxs,      # (word_num, K)  本步可选的全局AA索引（已保证合法）
        aa_mass,    # mass list
        premass: float,
        grid_size: float,
        tol: float,
        device: torch.device | str | None = None,
    ):
        """
        时间优先 DP（仅“新增一个AA”，无ε/重复检查；直接存整条路径）。
        只在每步给定的候选集合 aa_idxs[l-1, :] 中搜索，不遍历全集。

        维度:
        length   = word_num + 1               # 第0列为空序列
        cell_num = floor(premass / grid_size) + 1  # 质量栅格数
        dp.shape      == (length, cell_num)
        dp_mass.shape == (length, cell_num)
        paths.shape   == (length, cell_num, length)
            -> paths[l, t, :l] 是到 (l, t) 的完整AA索引序列（其余为 -1）
        """

        word_num, K = prob.shape
        length = word_num + 1
        cell_num = int(premass / grid_size) + 1
        last_cell = cell_num - 1

        dp      = torch.full((length, cell_num), float("-inf"), dtype=torch.float32, device=device)
        dp_mass = torch.zeros((length, cell_num), dtype=torch.float32, device=device)
        paths   = torch.full((length, cell_num, length), -1, dtype=torch.int32, device=device)

        # 起点：空序列位于 t=0
        dp[0, 0] = 0.0
        dp_mass[0, 0] = 0.0

        # l = 1..length-1，每步必须从候选集合里选一个AA
        for l in range(1, length):
            ids_step  = aa_idxs[l - 1]      # (K,)
            prob_step = prob[l - 1]         # (K,)

            for t in range(1, cell_num):
                
                # 枚举该步所有候选
                for j in range(K):
                    i = int(ids_step[j].item())        # 全局AA ID
                    # if l == 3 and t == 243 and i == 6:
                    #     print("debug")
                    w_i = float(aa_mass[i].item())     # 该AA质量（已保证合法>0）

                    # 反推上一个 cell：tp 与 tp+1（两次尝试覆盖边界）
                    base = t * grid_size - w_i
                    tp = int(base / grid_size)         # 向零截断，等价 C 的 int
                    for _ in range(2):
                        if 0 <= tp < cell_num and torch.isfinite(dp[l - 1, tp]):
                            # if l == 3 and t == 243:
                            #     print("debug")
                            new_mass = float(dp_mass[l - 1, tp].item()) + w_i

                            # 非末cell：命中该bin；末cell：premass±tol
                            if t != last_cell:
                                in_bin = (t * grid_size <= new_mass) and (new_mass < (t + 1) * grid_size)
                            else:
                                in_bin = (premass - tol <= new_mass <= premass + tol)

                            if in_bin:
                                cand = float(dp[l - 1, tp].item()) + float(prob_step[j].item())
                                if cand > float(dp[l, t].item()):
                                    dp[l, t] = cand
                                    dp_mass[l, t] = new_mass
                                    # 直接写整条路径：复制前缀 + 追加当前AA
                                    if l - 1 > 0:
                                        paths[l, t, :l - 1] = paths[l - 1, tp, :l - 1]
                                    paths[l, t, l - 1] = i
                        tp += 1  # 第二次尝试：tp+1

        return dp, dp_mass, paths

    def find_possible_path(self, seqs: torch.Tensor, scores: torch.Tensor, true_idx: list, mz: float, charges: float):
        """
        Find possible peptide sequences by filling in the masked positions
        based on precursor mass constraints.
        
        Arguments:
        - seqs (torch.Tensor): The candidate sequences, shape (n, seq_len).
        - score (torch.Tensor): The score matrix for each possible position, shape (n, seq_len).
        - true_idx (list): The indices of the masked positions in the sequences.
        - mz (float): The target m/z for each candidate.
        - charges (float): The charge state for each peptide sequence.

        Returns:
        - possible_paths (list): A list of possible paths satisfying the mass constraint.
        - scores (torch.Tensor): The score for each path, shape (n, num_possible_paths).
        """
        n, seq_len = seqs.shape

        # --- 1. Calculate fixed-part mass (once) ---

        all_indices = set(range(seq_len))
        true_idx_set = set(int(i) for i in true_idx)
        fixed_idx = sorted(list(all_indices - true_idx_set))
        if self.aa_mass.device != seqs.device:
            self.aa_mass = self.aa_mass.to(seqs.device)

        if len(fixed_idx) > 0:
            fixed_masses = self.aa_mass[seqs[:, fixed_idx]].sum(dim=1) # Shape (n)
        else:
            fixed_masses = torch.zeros(n, device=self.device, dtype=torch.float32)
    
        # --- 2. Define Mass Targets & Binning ---
        target_peptide_mass = (mz * charges) - charges * self.peptide_mass_calculator.proton
        target_residue_mass = target_peptide_mass - self.peptide_mass_calculator.h2o
        # required_mass is the mass we need to fill with *optional* AAs
        required_mass = target_residue_mass - fixed_masses # Shape (n)
        aa_idxs = seqs.T[true_idx]  # (length,topk)
        # --- 4. Prepare and Run DP Table (0/1 Knapsack) ---
        dp, dp_mass, final_paths = self.dp_path_search_topB(scores[true_idx], aa_idxs, self.aa_mass, 
                            required_mass[0], self.DP_BIN_SIZE, tol=0.1, device=seqs.device)
        
                
        return dp, dp_mass, final_paths

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Record the loss.
        loss = self.training_step(batch, mode="valid")
        if not self.calculate_precision:
            return loss

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        
        peptides_pred, peptides_true = [], batch[2]
        steps = []
        positional_scores = []
        scores = []

        def _is_mass_match(real_mass, predict_mass):
            abs_delta = abs(real_mass - predict_mass)
            return abs_delta < 0.1

        n_spectra = 0
        for spectrum_preds in self.forward(batch[0], batch[1]):
            peptide1 = peptides_true[n_spectra]
            spectra_fill = False
            for pred_dict in spectrum_preds:
                precursors = batch[1]
                real_mass = precursors[n_spectra][0]
                predict_mass = self.peptide_mass_calculator.mass(pred_dict["sequence"])
                mask_position = pred_dict["positional_scores"] < -0.20
                if _is_mass_match(real_mass, predict_mass):
                    peptides_pred.append(pred_dict["sequence"])
                    steps.append(pred_dict["steps"])
                    if pred_dict["score"]:
                        scores.append(pred_dict["score"].cpu().detach().numpy())
                        positional_scores.append(pred_dict["positional_scores"].cpu().detach().numpy())
                    else:
                        scores.append(-100000)
                        positional_scores.append([-10000])
                    spectra_fill = True
                    break
                memory, memory_key_padding_mask = self.encoder(batch[0])
                peptide2 = pred_dict["sequence"]
                tokens = self.decoder.tokenize(peptide2)
                true_tokens = self.decoder.tokenize(peptide1)
                pad_false = torch.tensor([False], device=tokens.device, dtype=torch.bool)
                mask_position = torch.cat([pad_false, mask_position, pad_false], dim=0)
                # Prepare mass and charge
                masses = self.decoder.mass_encoder(batch[1][:, None, 0])
                charges = self.decoder.charge_encoder(batch[1][:, 1].int() - 1)
                precursors_embedding = masses + charges[:, None, :]

                orginal_predict = copy.deepcopy(pred_dict)
                # Try with insert postion filling
                mask_ins_score, _ = self.decoder.forward_mask_ins(
                    normalize=True,
                    precurosors=precursors_embedding[n_spectra].unsqueeze(0),
                    encoder_out=memory[n_spectra].unsqueeze(0),
                    encoder_out_mask=memory_key_padding_mask[n_spectra].unsqueeze(0),
                    prev_output_tokens=tokens.unsqueeze(0),
                )
                insert_possible = torch.exp(mask_ins_score.squeeze(0))
                sampled_length_indices = torch.multinomial(insert_possible, num_samples=self.n_beams, replacement=True).T
                current_len = torch.sum(tokens != self.decoder.pad).item()
                seq_len = current_len + sampled_length_indices.sum(dim=1)
                max_length = seq_len.max().item()
                output_tensor = torch.full((self.n_beams, max_length), self.decoder.unk, dtype=tokens.dtype, device=tokens.device)

                bos_tokens = torch.full((self.n_beams, 1), self.decoder.bos, dtype=tokens.dtype, device=tokens.device)
                bos_indices = torch.zeros((self.n_beams, 1), dtype=torch.long, device=tokens.device)
                output_tensor.scatter_(dim=1, index=bos_indices, src=bos_tokens)

                tokens[mask_position] = self.decoder.unk
                num_content_tokens = current_len - 2 
                # src_content_tokens = original_token_batch[:, 1 : 1 + num_content_tokens]
                base_indices = torch.arange(1, num_content_tokens + 1, device=tokens.device).expand(self.n_beams, -1)
                insert_offsets = base_indices + torch.cumsum(sampled_length_indices[:, :num_content_tokens], dim=1)
                output_tensor.scatter_(dim=1, index=insert_offsets, src=tokens[1:1 + num_content_tokens].expand(self.n_beams, -1))

                eos_indices = (seq_len - 1).unsqueeze(1)
                eos_tokens = torch.full((self.n_beams, 1), self.decoder.eos, dtype=tokens.dtype, device=tokens.device)
                output_tensor.scatter_(dim=1, index=eos_indices, src=eos_tokens)
                mask_indices = torch.arange(max_length, device=tokens.device).expand(self.n_beams, -1)
                padding_mask = mask_indices >= seq_len.unsqueeze(1)
                output_tensor[padding_mask] = self.decoder.pad
                padded_original = F.pad(tokens, (0, max_length - len(tokens)), mode='constant', value=self.decoder.pad)
                output_tensor = torch.cat((output_tensor, padded_original.unsqueeze(0)), dim=0)
                output_tensor = output_tensor.unique(dim=0)
                batch_size = output_tensor.size(0)

                word_ins_score, _ = self.decoder.forward_word_ins(
                    normalize=True,
                    precurosors=precursors_embedding[n_spectra].unsqueeze(0).expand(batch_size, -1, -1),
                    encoder_out=memory[n_spectra].unsqueeze(0).expand(batch_size, -1, -1),
                    encoder_out_mask=memory_key_padding_mask[n_spectra].unsqueeze(0).expand(batch_size, -1),
                    prev_output_tokens=output_tensor,
                )
                try:
                    topv, topi = torch.topk(word_ins_score, k=self.n_beams, dim=-1)  # topv/topi: [B, T, 2]
                except Exception as e:
                    print(f"n_spectra:{n_spectra}, word_ins_score.shape: {word_ins_score.shape}")

                mask_position = (output_tensor == self.decoder.unk)
                K = min(self.n_beams, topi.size(1))
                # 复制 K 份原序列 -> [K, T]
                seqs = output_tensor.unsqueeze(1).repeat(1, K, 1)
                sampled_indices = torch.randint(low=0, high=K, size=topi.shape, device=topi.device)
                new_tokens = torch.gather(topi, 2, sampled_indices)
                new_tokens_permuted = new_tokens.permute(0, 2, 1)
                mask = mask_position.unsqueeze(1)
                seqs = torch.where(mask, new_tokens_permuted, seqs)
                # 只在 true_idx 上回填，不改其他位置
                # topi[true_idx] -> [M, K]，转置后是 [K, M]，与 seqs[:, true_idx] 的形状对齐
                # seqs[:, true_idx] = topi[true_idx, :K].T
                seqs = seqs.reshape(-1, seqs.size(-1))  # [B*K, T]
                B = len(seqs)
                spec = batch[0][n_spectra]            # [n_peaks, 2]
                spec_rep = spec.unsqueeze(0).expand(B, -1, -1)  # [B, n_peaks, 2]           
                candidates = self.forward(spec_rep, batch[1][n_spectra].unsqueeze(0).expand(B, -1), prev_output_tokens=seqs)
                candidates = [pred_dict for preds in candidates for pred_dict in preds]
                candidate_tokens = [c["tokens"] for c in candidates]
                candidate_tokens = pad_sequence(candidate_tokens, batch_first=True, padding_value=0)
                candidate_tokens = torch.unique(candidate_tokens, dim=0)
                mz_matched_candidates = []
                for token in candidate_tokens:
                    seq = self.decoder.detokenize(token)
                    if _is_mass_match(real_mass, self.peptide_mass_calculator.mass(seq)):
                        mz_matched_candidates.append(token)
                if len(mz_matched_candidates) == 1:
                    peptide = self.decoder.detokenize(mz_matched_candidates[0])
                    peptide = "".join(item for item in peptide if item not in {"$", "&", ""})
                    peptides_pred.append(peptide)
                    steps.append(0)
                    scores.append(-200000)
                    positional_scores.append([-20000])
                    spectra_fill = True
                    break
                if len(mz_matched_candidates) > 1:
                    mz_matched_candidates = torch.stack(mz_matched_candidates, dim=0)
                    batch_size = mz_matched_candidates.size(0)
                    mask = (mz_matched_candidates != self.decoder.bos) & (mz_matched_candidates != self.decoder.eos) & (mz_matched_candidates != self.decoder.pad)
                    mask = mask.unsqueeze(-1)
                    predic_delete, _ = self.decoder.forward_word_del(
                        normalize=True,
                        precurosors=precursors_embedding[n_spectra].unsqueeze(0).expand(batch_size, -1, -1),
                        encoder_out=memory[n_spectra].unsqueeze(0).expand(batch_size, -1, -1),
                        encoder_out_mask=memory_key_padding_mask[n_spectra].unsqueeze(0).expand(batch_size, -1),
                        prev_output_tokens=mz_matched_candidates,
                    )
                    predic_delete = torch.where(mask, predic_delete, torch.nan)
                    result = torch.nanmean(predic_delete, dim=1, keepdim=True)
                    result = result[:, 0, 0]
                    max_score, max_idx = torch.max(result, dim=0)
                    peptide = self.decoder.detokenize(mz_matched_candidates[max_idx])
                    peptide = "".join(item for item in peptide if item not in {"$", "&", ""})
                    peptides_pred.append(peptide)
                    steps.append(0)
                    scores.append(max_score.cpu().detach().numpy())
                    positional_scores.append([-30000])
                    spectra_fill = True
                    break
                pass
                # path = self.find_possible_path(seqs, topv, true_idx, batch[1][n_spectra, 2], batch[1][n_spectra, 1])
                # if not torch.all(path[-1, -1, :len(true_idx)] == -1):
                #     aa_indices = path[-1, -1, :len(true_idx)]  # (length,)
                #     tokens[true_idx] = aa_indices.to(dtype=torch.int64)
                #     pred_dict["sequence"] = "".join(self.decoder.detokenize(tokens))
                #     pred_dict["sequence"] = pred_dict["sequence"][1:-1]  # remove <s> and </s>
                #     predic_delete = self.decoder.forward_word_del(
                #         normalize=True,
                #         precurosors=precursors_embedding[n_spectra].unsqueeze(0),
                #         encoder_out=memory[n_spectra].unsqueeze(0),
                #         encoder_out_mask=memory_key_padding_mask[n_spectra].unsqueeze(0),
                #         prev_output_tokens=tokens.unsqueeze(0),
                #     )
                #     predict_sum = torch.sum(predic_delete[0], dim=1, keepdim=True)
                #     true_delete = self.decoder.forward_word_del(
                #         normalize=True,
                #         precurosors=precursors_embedding[n_spectra].unsqueeze(0),
                #         encoder_out=memory[n_spectra].unsqueeze(0),
                #         encoder_out_mask=memory_key_padding_mask[n_spectra].unsqueeze(0),
                #         prev_output_tokens=true_tokens.unsqueeze(0),
                #     )
                #     true_delete_sum = torch.sum(true_delete[0], dim=1, keepdim=True)
                #     peptide =  pred_dict["sequence"]
                #     peptides_pred.append(peptide)
                #     steps.append(pred_dict["steps"])
                #     scores.append(-200000)
                #     positional_scores.append([-20000])
                #     spectra_fill = True
                #     break
            n_spectra += 1
            if not spectra_fill:
                peptides_pred.append(orginal_predict["sequence"])
                steps.append(orginal_predict["steps"])
                if pred_dict["score"]:
                    scores.append(orginal_predict["score"].cpu().detach().numpy())
                    positional_scores.append(orginal_predict["positional_scores"].cpu().detach().numpy())
                else:
                    scores.append(-100000)
                    positional_scores.append([-10000])

        # logger.info("Peptide_true | Peptide_predict |Steps |  Full_match | Match_count | n_aa1 | n_aa2")
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                logger,
                peptides_true,
                peptides_pred,
                self.decoder._peptide_mass.masses,
                scores=scores,
                positional_scores=positional_scores,
                steps=steps,
            )
        )
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "Peptide precision at coverage=1",
            pep_precision,
            **log_args,
        )
        self.log(
            "AA precision at coverage=1",
            aa_precision,
            **log_args,
        )
        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        predictions: List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]
            Model predictions for the given batch of spectra containing spectrum
            ids, precursor information, peptide sequences as well as peptide
            and amino acid-level confidence scores.
        """
        predictions = []
        for (
            precursor_charge,
            precursor_mz,
            spectrum_i,
            spectrum_preds,
        ) in zip(
            batch[1][:, 1].cpu().detach().numpy(),
            batch[1][:, 2].cpu().detach().numpy(),
            batch[2],
            self.forward(batch[0], batch[1]),
        ):
            # 遍历列表里的每个 dict
            for pred in spectrum_preds:
                peptide      = pred["sequence"] if len(pred["sequence"]) > 0 else ""
                peptide_score= pred["score"].cpu().detach().numpy() if pred["score"] is not None else 0.0
                aa_scores    = pred["positional_scores"].cpu().detach().numpy() if pred["positional_scores"] is not None else [0.0]
                predictions.append(
                    (
                        spectrum_i,
                        precursor_charge,
                        precursor_mz,
                        peptide,
                        peptide_score,
                        aa_scores,
                    )
                )

        return predictions

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["train_CELoss"].detach()
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss.item(),
            "mask-ins-train": self.trainer.callback_metrics[
                "train_mask_ins-loss"
            ].detach().item(),
            "word-ins-train": self.trainer.callback_metrics[
                "train_word_ins-loss"
            ].detach().item(),
            "word-del-train": self.trainer.callback_metrics[
                "train_word_del-loss"
            ].detach().item(),
            # "rank-train": self.trainer.callback_metrics[
            #     "train_rank_loss"
            # ].detach().item(),
        }
        if self.dual_training_for_deletion:
            metrics["dual-word-del-train"] = self.trainer.callback_metrics[
                "train_word_del_dual-loss"
            ].detach().item()
        if self.dual_training_for_insertion:
            metrics["dual-mask-ins-train"] = self.trainer.callback_metrics[
                "train_mask_ins_dual-loss"
            ].detach().item()
            metrics["dual-word-ins-train"] = self.trainer.callback_metrics[
                "train_word_ins_dual-loss"
            ].detach().item()
        self._history.append(metrics)
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "step": self.trainer.global_step,
            "valid": callback_metrics["valid_CELoss"].detach().item(),
            "mask-ins-valid": callback_metrics["valid_mask_ins-loss"].detach().item(),
            "word-ins-valid": callback_metrics["valid_word_ins-loss"].detach().item(),
            "word-del-valid": callback_metrics["valid_word_del-loss"].detach().item(),
        }
        if self.dual_training_for_deletion:
            metrics["dual-word-del-valid"] = (
                callback_metrics["valid_word_del_dual-loss"].detach().item()
            )
        if self.dual_training_for_insertion:
            metrics["dual-mask-ins-valid"] = (
                callback_metrics["valid_mask_ins_dual-loss"].detach().item()
            )
            metrics["dual-word-ins-valid"] = (
                callback_metrics["valid_word_ins_dual-loss"].detach().item()
            )

        if self.calculate_precision:
            metrics["valid_aa_precision"] = (
                callback_metrics["AA precision at coverage=1"].detach().item()
            )
            metrics["valid_pep_precision"] = (
                callback_metrics["Peptide precision at coverage=1"]
                .detach()
                .item()
            )
        self._history.append(metrics)
        self._log_history()

    def on_predict_batch_end(
        self,
        outputs: List[Tuple[np.ndarray, List[str], torch.Tensor]],
        *args,
    ) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """
        if self.out_writer is None:
            return
        # Triply nested lists: results -> batch -> step -> spectrum.
        for (
            spectrum_i,
            charge,
            precursor_mz,
            peptide,
            peptide_score,
            aa_scores,
        ) in outputs:
            if len(peptide) == 0:
                continue
            self.out_writer.psms.append(
                (
                    peptide,
                    tuple(spectrum_i),
                    peptide_score,
                    charge,
                    precursor_mz,
                    self.peptide_mass_calculator.mass(peptide, charge),
                    ",".join(list(map("{:.5f}".format, aa_scores))),
                ),
            )

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain loss\tMask-ins-loss\tWord-ins-loss\tWord-del-loss\tValid loss\t"
            if self.calculate_precision:
                header += "Peptide precision\tAA precision"

            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            if self.dual_training_for_deletion and self.dual_training_for_insertion:
                msg = "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
                vals = [
                    metrics["step"],
                    metrics.get("train", np.nan),
                    metrics.get("mask-ins-train", np.nan),
                    metrics.get("word-ins-train", np.nan),
                    metrics.get("word-del-train", np.nan),
                    metrics.get("dual-word-del-train", np.nan),
                    metrics.get("dual-mask-ins-train", np.nan),
                    metrics.get("dual-word-ins-train", np.nan),
                    metrics.get("valid", np.nan),
                    metrics.get("mask-ins-valid", np.nan),
                    metrics.get("word-ins-valid", np.nan),
                    metrics.get("word-del-valid", np.nan),
                    metrics.get("dual-word-del-valid", np.nan),
                    metrics.get("dual-mask-ins-valid", np.nan),
                    metrics.get("dual-word-ins-valid", np.nan),
                ]

            elif self.dual_training_for_insertion:
                msg = "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6ft%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
                vals = [
                    metrics["step"],
                    metrics.get("train", np.nan),
                    metrics.get("mask-ins-train", np.nan),
                    metrics.get("word-ins-train", np.nan),
                    metrics.get("word-del-train", np.nan),
                    # metrics.get("dual-word-del-train", np.nan),
                    metrics.get("dual-mask-ins-train", np.nan),
                    metrics.get("dual-word-ins-train", np.nan),
                    metrics.get("valid", np.nan),
                    metrics.get("mask-ins-valid", np.nan),
                    metrics.get("word-ins-valid", np.nan),
                    metrics.get("word-del-valid", np.nan),
                    # metrics.get("dual-word-del-valid", np.nan),
                    metrics.get("dual-mask-ins-valid", np.nan),
                    metrics.get("dual-word-ins-valid", np.nan),
                ]
            else:
                msg = "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
                vals = [
                    metrics["step"],
                    metrics.get("train", np.nan),
                    metrics.get("mask-ins-train", np.nan),
                    metrics.get("word-ins-train", np.nan),
                    metrics.get("word-del-train", np.nan),
                    metrics.get("dual-word-del-train", np.nan),
                    metrics.get("valid", np.nan),
                    metrics.get("mask-ins-valid", np.nan),
                    metrics.get("word-ins-valid", np.nan),
                    metrics.get("word-del-valid", np.nan),
                    metrics.get("dual-word-del-valid", np.nan),
                ]

            if self.calculate_precision:
                msg += "\t%.6f\t%.6f"
                vals += [
                    metrics.get("valid_pep_precision", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                ]

            logger.info(msg, *vals)
            if self.tb_summarywriter is not None:
                for descr, key in [
                    ("loss/train_crossentropy_loss", "train"),
                    ("loss/train_mask-ins_loss", "mask-ins-train"),
                    ("loss/train_word-ins_loss", "word-ins-train"),
                    ("loss/train_word-del_loss", "word-del-train"),
                    ("loss/train_word-del_dual_loss", "dual-word-del-train"),
                    ("loss/dual_mask-ins_loss", "dual-mask-ins-train"),
                    ("loss/dual_word-ins_loss", "dual-word-ins-train"),
                    ("loss/train_rank_loss", "rank-train"),
                    ("loss/val_crossentropy_loss", "valid"),
                    ("loss/val_mask-ins_loss", "mask-ins-valid"),
                    ("loss/val_word-ins_loss", "word-ins-valid"),
                    ("loss/val_word-del_loss", "word-del-valid"),
                    ("loss/val_word-del_dual_loss", "dual-word-del-valid"),
                    ("loss/val_dual_mask-ins_loss", "dual-mask-ins-valid"),
                    ("loss/val_dual_word-ins_loss", "dual-word-ins-valid"),
                    ("eval/val_pep_precision", "valid_pep_precision"),
                    ("eval/val_aa_precision", "valid_aa_precision"),
                ]:
                    metric_value = metrics.get(key, np.nan)
                    if not np.isnan(metric_value):
                        self.tb_summarywriter.add_scalar(
                            descr, metric_value, metrics["step"]
                        )

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized AdamW optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), **self.opt_kwargs, fused=True)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupConstantScheduler(
            optimizer, 
            self.warmup_iters, 
            self.cosine_schedule_period_iters, 
            self.constant_lr_iters,
            self.final_decay_iters,
            min_lr_factor=self.min_lr_factor
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup in first cycle only, followed by cosine decay cycles.
    First cycle: warmup + cosine decay (full peak LR)
    Subsequent cycles: cosine decay only (0.1 * peak LR, no warmup)

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate (only in first cycle).
    cosine_schedule_period_iters : int
        The number of iterations for the cosine decay period in the first cycle.
    min_lr_factor : float, optional
        The minimum learning rate factor as a fraction of the base learning rate. Default is 0.01 (1%).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
        min_lr_factor: float = 0.001,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """
        Calculate learning rate factor with multiple cycles:
        - Cycle 1: warmup + cosine decay (full peak LR)
        - Cycle 2+: cosine decay only (0.1 * peak LR, no warmup)
        
        Note: Only the first cycle has warmup. Subsequent cycles start directly at 0.1 * peak LR.
        The learning rate will not go below min_lr_factor.
        """
        current_epoch = epoch
        
        # First cycle: warmup + cosine decay
        first_cycle_total = self.warmup_iters + self.cosine_schedule_period_iters
        
        if current_epoch < first_cycle_total:
            # We're in the first cycle
            if current_epoch <= self.warmup_iters:
                # Warmup phase - only in first cycle
                peak_lr_factor = 1.0
                lr_factor = self.min_lr_factor + (peak_lr_factor - self.min_lr_factor) * (current_epoch / self.warmup_iters)
            else:
                # Cosine decay phase in first cycle
                decay_epoch = current_epoch - self.warmup_iters
                peak_lr_factor = 1.0
                cosine_factor = 0.5 * (1 + np.cos(np.pi * decay_epoch / self.cosine_schedule_period_iters))
                lr_factor = self.min_lr_factor + (peak_lr_factor - self.min_lr_factor) * cosine_factor
            return lr_factor
        
        # Subsequent cycles: only cosine decay, no warmup
        remaining_epoch = current_epoch - first_cycle_total
        cycle_in_subsequent = remaining_epoch // self.cosine_schedule_period_iters
        decay_epoch_in_cycle = remaining_epoch % self.cosine_schedule_period_iters
        
        # Peak LR is 0.1 of original for all subsequent cycles
        peak_lr_factor = 0.1
        
        # Cosine decay from peak_lr_factor to min_lr_factor
        cosine_factor = 0.5 * (1 + np.cos(np.pi * decay_epoch_in_cycle / self.cosine_schedule_period_iters))
        lr_factor = self.min_lr_factor + (peak_lr_factor - self.min_lr_factor) * cosine_factor
        
        return lr_factor

class CosineWarmupConstantScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup, cosine decay, constant phase, and final cosine decay.
    First cycle: warmup + cosine decay + constant LR + final cosine decay to min
    Subsequent cycles: constant LR + final cosine decay to min

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate (only in first cycle).
    cosine_schedule_period_iters : int
        The number of iterations for the cosine decay period in the first cycle.
    constant_lr_iters : int
        The number of iterations to keep constant learning rate.
    final_decay_iters : int
        The number of iterations for final cosine decay from constant LR to min LR.
    constant_lr_factor : float, optional
        The constant learning rate factor as a fraction of the peak learning rate. Default is 0.05 (5%).
    min_lr_factor : float, optional
        The minimum learning rate factor as a fraction of the base learning rate. Default is 0.001 (0.1%).
    subsequent_constant_lr_ratio : float, optional
        For subsequent cycles, the constant LR duration as a percentage of the first cycle's cosine_schedule_period_iters. 
        Default is 1.0 (100% of cosine decay period). Set to 2.0 for 200% of cosine decay period, etc.
        Example: if cosine_schedule_period_iters=100000 and subsequent_constant_lr_ratio=1.5, 
        then subsequent cycles will have constant LR for 150000 iterations.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
        constant_lr_iters: int,
        final_decay_iters: int,
        constant_lr_factor: float = 0.01,
        min_lr_factor: float = 0.001,
        subsequent_constant_lr_ratio: float = 0.6,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        self.constant_lr_iters = constant_lr_iters
        self.final_decay_iters = final_decay_iters
        self.constant_lr_factor = constant_lr_factor
        self.min_lr_factor = min_lr_factor
        self.subsequent_constant_lr_ratio = subsequent_constant_lr_ratio
        
        # Calculate the constant LR duration for subsequent cycles
        self.subsequent_constant_lr_iters = int(cosine_schedule_period_iters * subsequent_constant_lr_ratio)
        super().__init__(optimizer)
        print(f"Using CosineWarmupConstantScheduler with warmup_iters={warmup_iters}, "
              f"cosine_schedule_period_iters={cosine_schedule_period_iters}, constant_lr_iters={constant_lr_iters},"
              f" final_decay_iters={final_decay_iters}, constant_lr_factor={constant_lr_factor}, min_lr_factor={min_lr_factor}, "
              f"subsequent_constant_lr_ratio={subsequent_constant_lr_ratio}, subsequent_constant_lr_iters={self.subsequent_constant_lr_iters}")

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """
        Calculate learning rate factor with multiple phases:
        First cycle: warmup → cosine decay → constant LR → final cosine decay to min
        Subsequent cycles: constant LR → final cosine decay to min
        """
        current_epoch = epoch
        
        # First cycle: warmup + cosine + constant + final decay
        first_cycle_total = (self.warmup_iters + self.cosine_schedule_period_iters + 
                           self.constant_lr_iters + self.final_decay_iters)
        
        if current_epoch < first_cycle_total:
            # We're in the first cycle
            if current_epoch <= self.warmup_iters:
                # Phase 1: Warmup phase (only in first cycle)
                peak_lr_factor = 1.0
                lr_factor = self.min_lr_factor + (peak_lr_factor - self.min_lr_factor) * (current_epoch / self.warmup_iters)
                
            elif current_epoch <= self.warmup_iters + self.cosine_schedule_period_iters:
                # Phase 2: Cosine decay phase (first cycle only)
                decay_epoch = current_epoch - self.warmup_iters
                peak_lr_factor = 1.0
                # Decay from peak to constant LR level
                cosine_factor = 0.5 * (1 + np.cos(np.pi * decay_epoch / self.cosine_schedule_period_iters))
                lr_factor = self.constant_lr_factor + (peak_lr_factor - self.constant_lr_factor) * cosine_factor
                
            elif current_epoch <= self.warmup_iters + self.cosine_schedule_period_iters + self.constant_lr_iters:
                # Phase 3: Constant LR phase
                lr_factor = self.constant_lr_factor
                
            else:
                # Phase 4: Final decay to minimum
                final_decay_epoch = current_epoch - (self.warmup_iters + self.cosine_schedule_period_iters + self.constant_lr_iters)
                # Cosine decay from constant LR to min LR
                decay_progress = final_decay_epoch / self.final_decay_iters
                cosine_factor = 0.5 * (1 + np.cos(np.pi * decay_progress))
                lr_factor = self.min_lr_factor + (self.constant_lr_factor - self.min_lr_factor) * cosine_factor
                
            return lr_factor
        
        # Subsequent cycles: constant LR + final decay only
        remaining_epoch = current_epoch - first_cycle_total
        subsequent_cycle_length = self.subsequent_constant_lr_iters + self.final_decay_iters
        
        # Find position within the current subsequent cycle
        cycle_epoch = remaining_epoch % subsequent_cycle_length
        
        if cycle_epoch < self.subsequent_constant_lr_iters:
            # Constant LR phase (longer duration in subsequent cycles)
            lr_factor = self.constant_lr_factor
        else:
            # Final decay phase
            final_decay_epoch = cycle_epoch - self.subsequent_constant_lr_iters
            decay_progress = final_decay_epoch / self.final_decay_iters
            # Cosine decay from constant LR to min LR
            cosine_factor = 0.5 * (1 + np.cos(np.pi * decay_progress))
            lr_factor = self.min_lr_factor + (self.constant_lr_factor - self.min_lr_factor) * cosine_factor
            
        return lr_factor
def _select_top_beams(
        cache_results: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]],
        beam_size: int,
        ignore_tokens: Optional[List[int]] = None,
        pad_token: int = 0,
        pad_score: float = float('-inf'),
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
      cache_results: 每个 batch 对应的候选列表，每个候选是 (tokens, scores)
      beam_size:   最终每个 batch 保留多少条 beam
      ignore_tokens: 在计算平均分时要忽略的 token 值列表
      pad_token:   用于输出序列 padding 的 token id
      pad_score:   用于输出分数 padding 的值（一般设为 -inf）
      device:      输出张量所在设备
    """

    B = len(cache_results)
    if ignore_tokens is None:
        ignore_tokens = []
    # 为了后面在 GPU 上做比较，这里把 ignore_tokens 转到同设备
    ignore_tensor = torch.tensor(ignore_tokens, device=device or "cpu")

    tokens_list: List[List[torch.Tensor]] = []
    scores_list: List[List[torch.Tensor]] = []

    for i in range(B):
        candidates = cache_results.get(i, [])

        # —— 针对每个候选，计算“有效位置”上的平均分 —— #
        avg_scores: List[float] = []
        for tok, sc in candidates:
            # mask[i]=True 表示 tok[i] 不是要忽略的 token
            # torch.isin 在 PyTorch >=1.10 可用，否则请用多次 != 累积
            mask = ~torch.isin(tok, ignore_tensor)

            if mask.any():
                # 只在 mask=True 的位置上取 sc，再求 mean
                valid_mean = sc[mask].float().mean().item()
            else:
                # 如果整条序列都被忽略，就给一个很小的分数
                valid_mean = pad_score
            avg_scores.append(valid_mean)

        avg_scores_tensor = torch.tensor(avg_scores, device=device or avg_scores[0].device)

        # 选出 top-k
        k = min(beam_size, len(avg_scores_tensor))
        if k > 0:
            topk_idx = torch.topk(avg_scores_tensor, k=k, largest=True).indices.tolist()
        else:
            topk_idx = []

        # 收集被选的 tok 和 sc
        toks_i: List[torch.Tensor] = []
        scs_i:  List[torch.Tensor] = []
        for idx in topk_idx:
            tok, sc = candidates[idx]
            toks_i.append(tok)
            scs_i.append(sc)
        tokens_list.append(toks_i)
        scores_list.append(scs_i)

    # —— 后续：padding 到统一长度并输出 —— #
    all_lengths = [tok.size(0) for toks in tokens_list for tok in toks]
    if not all_lengths:
        raise ValueError("没有任何候选 beam 被选中，无法推断 T_max")
    T_max = max(all_lengths)

    top_tokens = torch.full((B, beam_size, T_max), pad_token, dtype=torch.long, device=device)
    top_scores = torch.full((B, beam_size, T_max), pad_score, dtype=torch.float, device=device)

    for i in range(B):
        for j, tok in enumerate(tokens_list[i]):
            L = tok.size(0)
            top_tokens[i, j, :L] = tok
            top_scores[i, j, :L] = scores_list[i][j]

    top_tokens = top_tokens.view(B * beam_size, T_max)
    top_scores = top_scores.view(B * beam_size, T_max)
    spectra_index = torch.arange(B, device=device).repeat_interleave(beam_size)

    return top_tokens, top_scores, spectra_index
def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


def _aa_pep_score(
    aa_scores: np.ndarray, fits_precursor_mz: bool
) -> Tuple[np.ndarray, float]:
    """
    Calculate amino acid and peptide-level confidence score from the raw amino
    acid scores.

    The peptide score is the mean of the raw amino acid scores. The amino acid
    scores are the mean of the raw amino acid scores and the peptide score.

    Parameters
    ----------
    aa_scores : np.ndarray
        Amino acid level confidence scores.
    fits_precursor_mz : bool
        Flag indicating whether the prediction fits the precursor m/z filter.

    Returns
    -------
    aa_scores : np.ndarray
        The amino acid scores.
    peptide_score : float
        The peptide score.
    """
    peptide_score = np.mean(aa_scores)
    aa_scores = (aa_scores + peptide_score) / 2
    if not fits_precursor_mz:
        peptide_score -= 1
    return aa_scores, peptide_score
