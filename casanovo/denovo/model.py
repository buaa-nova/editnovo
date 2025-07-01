"""A de novo peptide sequencing model."""

from ast import Assert
import collections
import heapq
import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch.nn.functional as F

from casanovo.depthcharge.components.levenstein_util import is_a_loop, item

from ..depthcharge.masses import PeptideMass
import einops
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter
from ..depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder

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
    out_writer : Optional[str]
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
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        max_decoder_iters: int = 1,
        eos_penalty : int = 0,
        max_ratio: float = 0.5,
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
        )
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
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
        
        vocab = self.decoder.vocab_size + 1  # V  ?
        beam = self.n_beams  # S
        # Initialize scores and tokens.
        prev_decoder_out = self.initialize_output_tokens(memories)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()  # [B, L]
        
        # Prepare mass and charge
        masses = self.decoder.mass_encoder(precursors[:, None, 0])
        charges = self.decoder.charge_encoder(precursors[:, 1].int() - 1)
        precursors_embedding = masses + charges[:, None, :]
        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        word_ins_score, predict_mask_tokens, need_greedy = self.decoder.forward_decoder(
                0, self.encoder, prev_decoder_out, precursors_embedding, memories, mem_masks, early_exit=True
            )
        # beam search in mask postion
        # new_seq shape:(B, S, T), new_scores shape:(B, S, T)
        if need_greedy:
            tokens, scores = self.greedy_k_best_on_masks(
                word_ins_score, predict_mask_tokens, prev_decoder_out.output_tokens, prev_decoder_out.output_scores, beam
            )
        else:
            # flag meaning that the decoder has already returned the best tokens and scores
            tokens = predict_mask_tokens
            scores = word_ins_score

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
        tokens = einops.rearrange(tokens, "B S L -> (B S) L")
        scores = einops.rearrange(scores, "B S L -> (B S) L")

        for step in range(self.max_iter + 1):
            # 1. check termination
            (
                finished_beams, #
                beam_fits_precursor,
            ) = self._finish_beams(tokens, prev_decoder_out, precursors, beam, step, self.extra_ignored_ids)

            # 2. collect finalized sentences
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams,
                beam_fits_precursor,
                pred_cache,
            )
            
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
            word_ins_score, predict_mask_tokens, need_greedy = self.decoder.forward_decoder(
                step, self.encoder, 
                decoder_out,
                precursors_embedding[~finished_beams, :], 
                memories[~finished_beams, :],
                mem_masks[~finished_beams, :], 
                early_exit=True
            )

            #tokens_tmp shape: [N_active, S, L]
            if need_greedy:
                tokens_tmp, scores_tmp = self.greedy_k_best_on_masks(
                    word_ins_score, 
                    predict_mask_tokens, 
                    prev_decoder_out.output_tokens[~finished_beams, :], 
                    prev_decoder_out.output_scores[~finished_beams, :], 
                    beam
                )
            else:
                # flag meaning that the decoder has already returned the best tokens and scores
                tokens_tmp = predict_mask_tokens
                scores_tmp = word_ins_score

            

            # 每个spetra只保留 beam_size 条最优beam, tokens shape: [B*S, L], scores shape: [B*S, L]
            new_tokens, new_scores = self.replace_active_beams(
                tokens,
                scores,
                finished_beams,
                tokens_tmp,
                scores_tmp,
                beam
            )

            # Update the tokens and scores for the next step.
            prev_decoder_out._replace(
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
            int, List[Tuple[bool, float, np.ndarray, str]]
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
                torch.equal(pred_cached[-1], pred_peptide)
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
                    beam_fits_precursor[i],  # false < true
                    finalized_hypos["score"],
                    finalized_hypos["postional_scores"],
                    finalized_hypos["sequence"],
                ),
            )


 
    def _finish_beams(
            self,
            tokens:   torch.Tensor,  # [B*S, L]
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
        # 总行数 = B * S
        total_beams, L1 = tokens.size()
        _,  L2  = prev_out_token.size()
        S = beam_size
        # 计算 batch 大小 B
        B = total_beams // S

        # 恢复成 [B, S, L]
        toks = tokens.view(B, S, L1)
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
            seq = "".join(self.decoder.detokenize(token_ids))
            # 计算该序列的质量
            calc_mz = self.peptide_mass_calculator.mass(
                seq=seq,
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

        return finished_beams, beam_fits_precursor

        
    def replace_active_beams(
        tokens: torch.Tensor,           # [B*S, T] 原 tokens
        scores: torch.Tensor,           # [B*S, T] 原 scores 或聚合后[ B*S ]
        finished_beams: torch.BoolTensor,# [B*S] 已完成标记
        new_seqs: torch.LongTensor,     # [N_active, S, T]
        new_scores: torch.Tensor,       # [N_active, S] 或 [N_active, S, T]
        beam_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用 new_seqs/new_scores 更新 tokens/scores 中所有 active beams，
        并保证对每个样本只保留 beam_size 条最优 beam。
        """
        B = tokens.size(0) // beam_size
        T = tokens.size(1)
        device = tokens.device

        # 1) 找到 active beams 的索引和对应样本 id
        active = ~finished_beams                # [B*S]
        all_indices = torch.arange(B * beam_size, device=device)
        active_idx = all_indices[active]        # [N_active]
        sample_id = active_idx // beam_size     # [N_active], 每个 active beam 属于哪个 sample

        # 2) 把 new_seqs/scores 从 [N_active, S, ...] 摊平为 [N_active*S, ...]
        flat_seqs   = new_seqs.reshape(-1, T)           # [N_active*S, T]
        flat_scores = new_scores.reshape(-1, new_scores.size(-1))  # [N_active*S, T] 或 [N_active*S]

        # 3) 为每个 sample 分组，并择优
        #    我们将收集每个 sample 的所有 (flat_seqs, flat_scores, flat_idx)
        best_seqs  = torch.empty_like(tokens)
        best_scores= torch.empty_like(scores)

        for b in range(B):
            # 找到属于 sample b 的所有扁平 candidates
            mask_b = (sample_id == b).nonzero(as_tuple=False).view(-1)
            if mask_b.numel() == 0:
                # 如果这个 sample 没 active beam，直接填原来那些 finished_beams
                start = b * beam_size
                best_seqs[start:start+beam_size]   = tokens[start:start+beam_size]
                best_scores[start:start+beam_size] = scores[start:start+beam_size]
                continue

            # mask_b 中每个 i 对应 flat_seqs[i*S:(i+1)*S]
            # 实际上 greedy_k_best_on_masks 已经按 beam_size 输出了最优 S 条，
            # 所以第 i 条 active beam 展开后 flat_seqs 对应的 [i*S : i*S+S] 就是它的 S 条候选。
            cand_seqs   = flat_seqs[mask_b.repeat_interleave(beam_size)*beam_size + torch.arange(beam_size, device=device)]
            cand_scores = flat_scores[mask_b.repeat_interleave(beam_size)*beam_size + torch.arange(beam_size, device=device)]

            # 4) 选出分数最高的 beam_size 条
            #    先把 cand_scores 聚合为总分（如果是 per-pos，需要 sum）
            if cand_scores.dim() == 2:
                # cand_scores [N_can, T] -> 总分
                total_scores = cand_scores.sum(dim=1)
            else:
                total_scores = cand_scores  # 已经是一维 [N_can]

            topk_scores, topk_idx = torch.topk(total_scores, beam_size, largest=True)
            chosen_seqs   = cand_seqs[topk_idx]
            chosen_scores = cand_scores[topk_idx]

            # 5) 写回 flat tokens/scores
            base = b * beam_size
            best_seqs[base:base+beam_size]   = chosen_seqs
            best_scores[base:base+beam_size] = chosen_scores

        return best_seqs, best_scores
    def greedy_k_best_on_masks(
        self,
        word_ins_score: torch.Tensor,      # [B, T, V]
        predict_mask_tokens: torch.Tensor,  # [B, T]
        prev_output_tokens: torch.Tensor,  # [B, T]
        prev_scores: torch.Tensor,         # [B, T]
        beam_size: int
    )  -> Tuple[torch.Tensor, torch.Tensor]:
        """
        只对 prev_output_tokens==self.decoder.unk 的位置做贪心 Top-K,
        并回填到完整序列和完整得分。返回：
        new_seqs:  [B, beam_size, T] 完整的 token 序列
        new_scores:[B, beam_size, T] 每位置 log-prob (非mask位置填 prev_scores,其它填 beam 得分)
        """
        device = word_ins_score.device
        B, T, V = word_ins_score.shape
        unk = self.decoder.unk

        mask_matrix = predict_mask_tokens.eq(unk)  # [B, T]

        new_seqs   = torch.zeros(B, beam_size, T, dtype=torch.long,   device=device)
        new_scores = torch.zeros(B, beam_size, T, dtype=prev_scores.dtype, device=device)

        for b in range(B):
            mask_idx = mask_matrix[b].nonzero(as_tuple=False).view(-1)  # [M]
            M = mask_idx.size(0)

            if M == 0:
                # 直接用 prev_output_tokens 和 prev_scores 填满
                seq_rep = prev_output_tokens[b].unsqueeze(0).expand(beam_size, T)
                score_rep = prev_scores[b].unsqueeze(0).expand(beam_size, T)
                new_seqs[b] = seq_rep
                new_scores[b] = score_rep
                continue

            logits_b = word_ins_score[b, mask_idx]            # [M, V]
            logp_b   = torch.log_softmax(logits_b, dim=-1)    # [M, V]
            sorted_lp, sorted_idx = logp_b.sort(descending=True, dim=-1)  # [M, V]

            # 初始档位和分数
            init_ranks  = torch.zeros(M, dtype=torch.long, device=device)
            init_scores = sorted_lp[torch.arange(M, device=device), init_ranks].clone()
            results = [(init_ranks, init_scores)]
            used    = {tuple(init_ranks.tolist())}

            for _ in range(1, beam_size):
                best = None
                for prev_ranks, prev_scores_vec in results:
                    for m in range(M):
                        r = prev_ranks[m].item()
                        if r + 1 < V:
                            cand_ranks = prev_ranks.clone()
                            cand_ranks[m] += 1
                            key = tuple(cand_ranks.tolist())
                            if key in used:
                                continue
                            cand_scores = prev_scores_vec.clone()
                            cand_scores[m] = sorted_lp[m, cand_ranks[m]]
                            s = cand_scores.sum().item()
                            if best is None or s > best[0]:
                                best = (s, cand_ranks, cand_scores)
                if best is None:
                    break
                _, best_ranks, best_scores = best
                results.append((best_ranks, best_scores))
                used.add(tuple(best_ranks.tolist()))

            for k, (ranks, score_vec) in enumerate(results):
                token_ids = sorted_idx[torch.arange(M, device=device), ranks]  # [M]
                # 2) clone the full-length template and fill in the new tokens
                base_seq = predict_mask_tokens[b].clone()         # shape [T]
                base_seq[mask_idx] = token_ids                   # fill only mask slots
                new_seqs[b, k, :] = base_seq                     # assign into [B, S, T]

                # 3) build the corresponding full-length score vector
                # first compute non-mask indices once per b (you can hoist this outside the k-loop)
                all_idx      = torch.arange(T, device=device)
                mask_bool    = torch.zeros(T, dtype=torch.bool, device=device)
                mask_bool[mask_idx] = True
                non_mask_idx = all_idx[~mask_bool]                # length T-M

                # now fill new_scores at (b, k, :)
                new_scores[b, k, mask_idx]     = score_vec        # M new scores
                # 断言它们长度相同
                assert len(prev_scores[b]) == len(non_mask_idx), (
                    f"prev_scores length {len(prev_scores[b])} does not match "
                    f"non_mask_idx length {len(non_mask_idx)}"
                )
                new_scores[b, k, non_mask_idx] = prev_scores[b, :]  # T-M old scores


            if len(results) < beam_size:
                last = len(results) - 1
                for k in range(last + 1, beam_size):
                    new_seqs[b, k] = new_seqs[b, last]
                    new_scores[b, k] = new_scores[b, last]

        return new_seqs, new_scores

          

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor
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
       
        # initailize
        for i in range(spectra.size(0)):
            print(f"i= {i}, n_peaks={torch.count_nonzero(spectra[i]).item()/2}")  # 计算非零峰数量
        memory, memory_key_padding_mask = self.encoder(spectra)
        # Prepare mass and charge
        masses = self.decoder.mass_encoder(precursors[:, None, 0])
        charges = self.decoder.charge_encoder(precursors[:, 1].int() - 1)
        precursors_embedding = masses + charges[:, None, :]
        bsz = spectra.size(0)
        prev_decoder_out = self.initialize_output_tokens(memory)
        sent_idxs = torch.arange(bsz)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()
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

            decoder_out = self.decoder.forward_decoder(
                step, self.encoder, prev_decoder_out, precursors_embedding, memory, memory_key_padding_mask
            )

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
    #         return terminated, prev_out_token, prev_out_score, 

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
        pred_cache: Dict[int, List[Tuple[bool, float, torch.Tensor, str]]],
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
                for pep_score, aa_scores, peptide_str in topk:
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
        return self.decoder(sequences, precursors, *self.encoder(spectra))

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


        loss = sum(l["loss"] for l in losses)
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


    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0, weight=None
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: torch.Tensor, dim=None) -> torch.Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)

            losses = F.nll_loss(logits, targets.to(logits.device), reduction="none", weight=weight)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}
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
        # for spectrum_preds in self.forward(batch[0], batch[1]):
        #     for pred_dict in spectrum_preds:
        #         peptides_pred.append(pred_dict["sequence"])
        #         steps.append(pred_dict["steps"])
        #         scores.append(pred_dict["score"].cpu().detach().numpy())
        #         positional_scores.append(pred_dict["positional_scores"].cpu().detach().numpy())
        for spectrum_preds in self.beam_search_forward(batch[0], batch[1]):
            for score, pos_score, pred_peptide_str in spectrum_preds:
                peptides_pred.append(pred_peptide_str)
                steps.append(score["steps"])
                scores.append(score["score"].cpu().detach().numpy())
                positional_scores.append(pos_score.cpu().detach().numpy())


        
        logger.info("Peptide_true | Peptide_predict |Steps |  Full_match | Match_count | n_aa1 | n_aa2")
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
        }
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
            msg = "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
                metrics.get("mask-ins-train", np.nan),
                metrics.get("word-ins-train", np.nan),
                metrics.get("word-del-train", np.nan),
                metrics.get("valid", np.nan),
                metrics.get("mask-ins-valid", np.nan),
                metrics.get("word-ins-valid", np.nan),
                metrics.get("word-del-valid", np.nan),
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
                    ("loss/val_crossentropy_loss", "valid"),
                    ("loss/val_mask-ins_loss", "mask-ins-valid"),
                    ("loss/val_word-ins_loss", "word-ins-valid"),
                    ("loss/val_word-del_loss", "word-del-valid"),
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
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_iters, self.cosine_schedule_period_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor


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
