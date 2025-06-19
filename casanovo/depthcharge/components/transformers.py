"""Base Transformer models for working with mass spectra and peptides"""
import math
import re

import torch
import torch.nn.functional as F

from casanovo.depthcharge.components.levenstein_util import _apply_del_words, _apply_ins_masks, _apply_ins_words, _fill, _get_del_targets, _get_ins_targets, _skip, _skip_encoder_out
from casanovo.depthcharge.components.transformer_layer import TransformerDecoderLayerBase

from .encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from ..masses import PeptideMass
from .. import utils


class SpectrumEncoder(torch.nn.Module):
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        peak_encoder=True,
        dim_intensity=None,
    ):
        """Initialize a SpectrumEncoder"""
        super().__init__()

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))

        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )


    def reorder_encoder_out(self, precursors, memory, memory_key_padding, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            memory, memory_key_padding: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            precursors, memory, memory_key_padding rearranged according to *new_order*
        """

        if precursors is None or precursors.size(0) == 0:
            new_precursors = None
        else:
            new_precursors = precursors.index_select(0, new_order)

        if memory is None or memory.size(0) == 0:
            new_encoder_out = None
        else:
            new_encoder_out = memory.index_select(0, new_order)

        if memory_key_padding is None or memory_key_padding.size(0) == 0:
            new_encoder_padding_mask = None
        else:
            new_encoder_padding_mask = memory_key_padding.index_select(0, new_order)
        
        return new_precursors, new_encoder_out, new_encoder_padding_mask

    def forward(self, spectra):
        """The forward pass.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        zeros = ~spectra.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        peaks = self.peak_encoder(spectra)

        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        peaks = torch.cat([latent_spectra, peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    pos_encoder : bool
        Use positional encodings for the amino acid sequence.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum charge to embed.
    """

    def __init__(
        self,
        dim_model,
        pos_encoder,
        residues,
        max_charge,
    ):
        super().__init__()
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["&"] + ["$"] + ["mask"] # add a placeholder for start
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_model,
            padding_idx=0,
        )
        # # 定义一个专门给 mask 预测用的线性层（或 Embedding）：
        # self.mask_predictor = torch.nn.Linear(dim_model, num_classes_mask, bias=True)

    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse:
            sequence = list(reversed(sequence))

        if not partial:
            sequence += ["$"]
            sequence = ["&"] + sequence
        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    # def detokenize(self, tokens):
    #     """Transform tokens back into a peptide sequence.

    #     Parameters
    #     ----------
    #     tokens : torch.Tensor of shape (n_amino_acids,)
    #         The token for each amino acid in the peptide sequence.

    #     Returns
    #     -------
    #     list of str
    #         The amino acids in the peptide sequence.
    #     """
    #     sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
    #     if "$" in sequence:
    #         idx = sequence.index("$")
    #         sequence = sequence[: idx + 1]

    #     if self.reverse:
    #         sequence = list(reversed(sequence))

    #     return sequence
    def detokenize(self, tokens):
        """Transform tokens back into a peptide sequence.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids,)
            The token for each amino acid in the peptide sequence.

        Returns
        -------
        list of str
            The amino acids in the peptide sequence.
        """
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]

        # if self.reverse:
        #     sequence = list(reversed(sequence))

        return sequence

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class PeptideEncoder(_PeptideTransformer):
    """A transformer encoder for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        residues="canonical",
        max_charge=5,
    ):
        """Initialize a PeptideEncoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(self, sequences, charges):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor of length batch_size
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        charges : torch.Tensor of size (batch_size,)
            The charge state of the peptide

        Returns
        -------
        latent : torch.Tensor of shape (n_sequences, len_sequence, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        sequences = utils.listify(sequences)
        tokens = [self.tokenize(s) for s in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        encoded = self.aa_encoder(tokens)

        # Encode charges
        charges = self.charge_encoder(charges - 1)[:, None]
        encoded = torch.cat([charges, encoded], dim=1)

        # Create mask
        mask = ~encoded.sum(dim=2).bool()

        # Add positional encodings
        encoded = self.pos_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask


class PeptideDecoder(_PeptideTransformer):
    """A transformer decoder for peptide sequences.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    reverse : bool, optional
        Sequence peptides from c-terminus to n-terminus.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        reverse=False,
        residues="canonical",
        max_charge=5,
        max_ratio=None,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse
        self.max_ratio = max_ratio
        # Additional model components
        self.mass_encoder = FloatEncoder(dim_model)

        self.output_embed_dim = dim_model
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(dim_model, n_head, dim_feedforward, dropout,)
                for _ in range(n_layers)
            ]
        )
        self.num_layers = len(self.layers)

        self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)
        self.pad = 0
        self.bos = self._aa2idx["&"]
        self.eos = self._aa2idx["$"]
        self.unk = self._aa2idx["mask"]

        self.embed_mask_ins = torch.nn.Embedding(32, self.output_embed_dim * 2, None)
        self.embed_word_del = torch.nn.Embedding(2, self.output_embed_dim, None)

        self.output_projection = torch.nn.Linear(
            self.aa_encoder.weight.shape[1],
            self.aa_encoder.weight.shape[0],
            bias=False,
        )
        self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)
        self.final_mask_ins = torch.nn.Linear(self.output_embed_dim * 2, 32)
        self.final_word_del = torch.nn.Linear(self.output_embed_dim, 2)

        self.output_projection.weight = self.aa_encoder.weight
        # del_word, ins_mask, ins_word
        self.early_exit = [9, 9, 9]
        assert len(self.early_exit) == 3
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

    def build_decoder_layer(self, embed_dim, n_head, dim_feedforward, dropout_p, no_encoder_attn=False):
        layer = TransformerDecoderLayerBase(embed_dim, n_head, dim_feedforward, dropout_p, no_encoder_attn)
        return layer
    

    def forward_decoder(self, encoder, decoder_out, precursors, memory, memory_key_padding_mask, eos_penalty: float = 0):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        bsz = output_tokens.size(0)
        
        if self.max_ratio is None:
            max_lens = output_tokens.new_full((bsz,), 40, dtype=torch.long)
        else:
            src_lens = (~memory_key_padding_mask).sum(dim=1)
            max_lens = (src_lens * self.max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            precursors_tmp, memory_tmp, memory_key_padding_mask_tmp=_skip_encoder_out(encoder,
                                            precursors, memory, memory_key_padding_mask, can_del_word)
      
            word_del_score, _ = self.forward_word_del(
                normalize=True,
                precurosors=precursors_tmp,
                encoder_out=memory_tmp,
                encoder_out_mask=memory_key_padding_mask_tmp,
                prev_output_tokens=_skip(output_tokens, can_del_word), # skip the samples that cannot delete
                )
            
            word_del_pred = word_del_score.max(-1)[1].bool()
            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                None,  # del attn
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            precursors_tmp, memory_tmp, memory_key_padding_mask_tmp=_skip_encoder_out(encoder,
                                            precursors, memory, memory_key_padding_mask, can_ins_mask)
            mask_ins_score, _ = self.forward_mask_ins(
                normalize=True,
                precurosors=precursors_tmp,
                encoder_out=memory_tmp,
                encoder_out_mask=memory_key_padding_mask_tmp,
                prev_output_tokens=_skip(output_tokens, can_ins_mask)
            )
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            # clip the predicted insertion tokens to the maximum length
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            precursors_tmp, memory_tmp, memory_key_padding_mask_tmp=_skip_encoder_out(encoder,
                                            precursors, memory, memory_key_padding_mask, can_ins_word)
            word_ins_score, word_ins_attn = self.forward_word_ins(
                normalize=True,
                precurosors=precursors_tmp,
                encoder_out_mask=memory_key_padding_mask_tmp,
                encoder_out=memory_tmp,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
        )


    def forward(self, sequences, precursors, memory, memory_key_padding_mask):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The representations from a ``TransformerEncoder``, such as a
           ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.

        """

        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Prepare sequences
        if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]]).to(self.device)
        # shape: (batch_size, l)
        tgt_tokens = tokens
        # shape: (batch_size, l - delete)
        prev_output_tokens = self.inject_noise(tokens)
        
        # generate training labels for insertion
        # mask_ins_targets shape: (batch_size, l - delete - 1)
        # masked_tgt_tokens shape: (batch_size, l), masked_tgt_masks shape: (batch_size, l)
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            prev_output_tokens, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=31)  # for safe prediction
        # loss_class_weights = torch.ones(32, device=mask_ins_targets.device)
        # loss_class_weights[0] = 2 # if mask_ins_targets is 0, it means no insertion, so we give it a higher weight
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)
        # # mask_ins_out shape: (batch_size, l - delete - 1, 32)
        mask_ins_out, _ = self.forward_mask_ins(
            normalize=False,
            precurosors=precursors,
            encoder_out=memory,
            encoder_out_mask=memory_key_padding_mask,
            prev_output_tokens=prev_output_tokens,
        )
        # word_ins_out shape: (batch_size, l, vocab_size)
        word_ins_out, _ = self.forward_word_ins(
            normalize=False,
            precurosors=precursors,
            encoder_out=memory,
            encoder_out_mask=memory_key_padding_mask,
            prev_output_tokens=masked_tgt_tokens,
        )

        # make online prediction
        word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        # word_del_targets shape: (batch_size, l)
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.forward_word_del(
            normalize=False,
            precurosors=precursors,
            encoder_out=memory,
            encoder_out_mask=memory_key_padding_mask,
            prev_output_tokens=word_predictions,
        )

        word_del_masks = word_predictions.ne(self.pad)

        return {
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks, # shape: (batch_size, l - delete - 1)
                "ls": 0.01,
                # "weight": loss_class_weights,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks, # shape: (batch_size, l)
                "ls": 0.01,
                "nll_loss": True,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,  # shape: (batch_size, l)
                "mask": word_del_masks,
            },
        }


    def inject_noise(self, target_tokens):
        def _random_mask(target_tokens):
            """
            随机在每条序列中连续掩码一段长度（至少1）。
            输入:
            target_tokens: LongTensor of shape (B, T)
            输出:
            prev_target_tokens: 同shape，掩码位置替换为 unk
            """
            import torch

            # 特殊 token id
            
            B, T = target_tokens.size()

            # 1) 找到可掩码的位置（排除 pad/bos/eos）
            special = (target_tokens == self.pad) | (target_tokens == self.bos) | (target_tokens == self.eos)
            valid_mask = ~special  # True 表示可掩码
            valid_lens = valid_mask.sum(1).clamp(min=1)  # 每行可掩码数，至少1

            # 2) 设置最小掩码长度
            min_span = 2

            # 3) 计算每条序列的长度范围 length_range = valid_lens - min_span + 1
            #    该范围至少为 1，以便下面的随机采样
            length_range = (valid_lens - min_span + 1).clamp(min=1)  # (B,)

            # 4) 生成 [0,1) 的随机实数，用于映射到 [0, length_range)
            rand_frac = torch.rand(B, device=target_tokens.device)  

            # 5) span_lens = floor(rand_frac * length_range) + min_span
            #    结果在 [min_span, valid_lens]（当 valid_lens>=min_span）  
            span_lens = (rand_frac * length_range.float()).floor().long() + min_span  # (B,)

            # # 6) 对于 valid_lens < min_span 的序列，直接让掩码长度等于它们本身的 valid_lens
            # span_lens = torch.where(valid_lens < min_span, valid_lens, span_lens)

            # 7) 起始位置的随机采样，同上逻辑
            start_max = (valid_lens - span_lens).clamp(min=0)                # (B,)
            rand_frac2 = torch.rand(B, device=target_tokens.device)
            start_offsets = (rand_frac2 * (start_max.float() + 1)).floor().long()  # (B,)

            # 8) 执行掩码，并标记边界
            prev_target_tokens = target_tokens.clone()
            edge_mask = torch.zeros(B, T, dtype=torch.bool, device=target_tokens.device)

            for i in range(B):
                pos = valid_mask[i].nonzero(as_tuple=False).view(-1)  # 所有可掩码位置索引
                s, l = start_offsets[i].item(), span_lens[i].item()
                span = pos[s : s + l]
                prev_target_tokens[i, span] = self.unk

                # 只在段的最左和最右打 True
                edge_mask[i, span[0].item()] = True
                edge_mask[i, span[-1].item()] = True

            return prev_target_tokens, edge_mask

        def _random_delete(target_tokens):
           

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(self.pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(self.bos) | target_tokens.eq(self.eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, self.pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(self.pad).sum(1).max()
            ]

            return prev_target_tokens


        def _random_mask_20pct(target_tokens):
            """
            随机在每条序列中掩码大约 20% 的可掩码 token。
            输入:
            target_tokens: LongTensor of shape (B, T)
            返回:
            prev_target_tokens: LongTensor (B, T)，掩码位置用 unk 替换
            mask_positions:   BoolTensor  (B, T)，掩码位置为 True
            """
       

         
            B, T = target_tokens.size()

            # 标记可掩码位置
            special    = (target_tokens == self.pad) | (target_tokens == self.bos) | (target_tokens == self.eos)
            valid_mask = ~special                         # (B, T)

            prev = target_tokens.clone()
            mask_positions = torch.zeros(B, T, dtype=torch.bool, device=target_tokens.device)

            for i in range(B):
                # 1) 找到第 i 条序列所有可掩码位置的索引
                pos = valid_mask[i].nonzero(as_tuple=False).view(-1)
                n_valid = pos.size(0)

                # 2) 计算要掩码的数量 k = max(1, ceil(20% * n_valid))
                k = max(1, math.ceil(n_valid * 0.2))

                # 3) 随机打乱并选前 k 个
                perm = torch.randperm(n_valid, device=target_tokens.device)
                selected = pos[perm[:k]]

                # 4) 应用掩码
                prev[i, selected] = self.unk
                mask_positions[i, selected] = True
            return prev, mask_positions

        return _random_delete(target_tokens)
    

    def extract_features(
        self,
        prev_output_tokens,
        precursors_out,
        encoder_out=None,
        encoder_out_mask=None,
        early_exit=None,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # Feed through model:
        if prev_output_tokens is None:
            tgt = precursors_out
        else:
            # use precursors to replace BOS
            token_embed = self.aa_encoder(prev_output_tokens)
            tgt = torch.cat([precursors_out, token_embed[:, 1:, :]], dim=1)

        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        x = self.pos_encoder(tgt)

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)
        # attn = None
        # inner_states = [x]
        # layers = self.layers if layers is None else layers
        # early_exit = len(layers) if early_exit is None else early_exit
        # encoder_out = encoder_out.transpose(0, 1) if encoder_out is not None else None
        # decoder layers
        # for _, layer in enumerate(layers[:early_exit]):
        #     x, attn = layer(
        #         x,
        #         encoder_out,
        #         encoder_out_mask,
        #         self_attn_mask=None,
        #         self_attn_padding_mask=tgt_key_padding_mask,
        #     )
        #     inner_states.append(x)
        preds = self.transformer_decoder(
            tgt=x,
            memory=encoder_out,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_out_mask.to(self.device),
        )


        # T x B x C -> B x T x C
        # x = x.transpose(0, 1)

        # return x, {"attn": attn, "inner_states": inner_states}
        return preds, {"attn": None, "inner_states": None}


    def forward_word_ins(self, normalize, precurosors, encoder_out_mask,
                         encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            precursors_out=precurosors,
            encoder_out=encoder_out,
            encoder_out_mask=encoder_out_mask,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        # decoder_out = self.output_projection(features)
        decoder_out = self.final(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]
    
    def forward_mask_ins(self, normalize, precurosors, encoder_out, encoder_out_mask,
                         prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            precursors_out=precurosors,
            encoder_out=encoder_out,
            encoder_out_mask=encoder_out_mask,
            early_exit=self.early_exit[1],
            layers=self.layers, # use shared layers
            **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        # decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        decoder_out = self.final_mask_ins(features_cat)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]


    def forward_word_del(self, normalize,precurosors, encoder_out, encoder_out_mask,
                         prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            precursors_out=precurosors,
            encoder_out=encoder_out,
            encoder_out_mask=encoder_out_mask,
            early_exit=self.early_exit[0],
            layers=self.layers,
            **unused
        )
        # decoder_out = F.linear(features, self.embed_word_del.weight)
        decoder_out = self.final_word_del(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]   
def generate_tgt_mask(sz):
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
