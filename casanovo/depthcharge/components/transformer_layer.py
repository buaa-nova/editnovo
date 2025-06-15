from typing import Dict, List, Optional
import torch
import torch.nn as nn

from casanovo.depthcharge.components.levenstein_util import get_activation_fn
from casanovo.fairseq.multihead_attention import MultiheadAttention

class TransformerDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, embed_dim, n_head, dim_feedforward, dropout_p, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
    

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            n_head,
            self.dropout_p,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.dropout_module = nn.Dropout(p=dropout_p)
        self.activation_dropout_module = nn.Dropout(p=dropout_p)
        self.attn_ln = (
            self.LayerNorm(self.embed_dim)
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        self.activation_fn = get_activation_fn('relu')
    
        self.normalize_before = False

        self.self_attn_layer_norm = self.LayerNorm(self.embed_dim)

      
        self.encoder_attn = self.build_encoder_attention(self.embed_dim, self.n_head, self.dropout_p)
        self.encoder_attn_layer_norm = self.LayerNorm(self.embed_dim)

        self.ffn_layernorm = self.LayerNorm(self.dim_feedforward)
   

        self.fc1 = self.build_fc1(
            self.embed_dim,
            self.dim_feedforward,
        )
        self.fc2 = self.build_fc2(
            self.dim_feedforward,
            self.embed_dim,
        )

        self.final_layer_norm = self.LayerNorm(self.embed_dim)
        self.need_attn = True

    
    def LayerNorm(self, normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
       
        return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(
        self, embed_dim,n_head, dropout, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            n_head,
            dropout=dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

    def build_encoder_attention(self, embed_dim, n_heads, dropout):
        return MultiheadAttention(
            embed_dim,
            n_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            dropout=dropout,
            encoder_decoder_attention=True,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # 1. self attention layer
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
    
        y = x
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # 2. cross attention layer
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # Feed-Forward layer
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
       
        return x, attn

