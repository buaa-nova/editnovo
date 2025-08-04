# Ranker to rank the peptides based on their scores
import torch
import torch.nn as nn
import torch.nn.functional as F

class Ranker(nn.Module):
    def __init__(self, dim_model=128, n_head=8, dim_feedforward=512, n_layer=3, dropout_p=0.1,  temp: float = 0.07):
        super(Ranker, self).__init__()
        self.dim_model = dim_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout_p
        layer = nn.TransformerDecoderLayer(
            d_model=self.dim_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=n_layer)

        self.weight_proj_spec = nn.Linear(dim_model, 1)
        self.weight_proj_pep  = nn.Linear(dim_model, 1)
        self.logit_scale      = nn.Parameter(torch.log(torch.tensor(1.0 / temp)))


    def pool_spectrum(self, spec_enc: torch.Tensor, spec_mask: torch.Tensor) -> torch.Tensor:
        """
        spec_enc: (B, T, d_model)
        returns:  (B, d_model)
        """
        # compute per-position scores → (B, T)
        raw_w = self.weight_proj_spec(spec_enc).squeeze(-1)
        raw_w = raw_w.masked_fill(spec_mask, float('-inf'))
        a = torch.softmax(raw_w, dim=1)              # (B, T)
        # weighted sum → (B, d_model)
        return torch.einsum('btd,bt->bd', spec_enc, a)
    
    def pool_peptide(self,
                     spec_enc: torch.Tensor,
                     pep_enc:  torch.Tensor,
                     spec_mask: torch.BoolTensor,
                    ) -> torch.Tensor:
        """
        first cross-attend peptide→spectrum, then weighted pool:
          pep_enc:  (B, L, d_model)
        returns:    (B, d_model)
        """
        tgt_key_padding_mask = pep_enc.sum(axis=2) == 0
        attn_out = self.transformer_decoder(
            tgt=pep_enc,  # (B, L, d_model)
            memory=spec_enc,  # (B, T, d_model)
            memory_key_padding_mask=spec_mask,  # (B, T)
            tgt_key_padding_mask=tgt_key_padding_mask,  # (B, L)
        )  # → (B, L, d_model)

        raw_w = self.weight_proj_pep(attn_out).squeeze(-1)  # (B, L)
        a = torch.softmax(raw_w, dim=1)                 # (B, L)
        return torch.einsum('bld,bl->bd', attn_out, a)      # (B, d_model)

    def forward(self,
                spec_enc: torch.Tensor,
                pep_enc:  torch.Tensor,
                spec_mask: torch.BoolTensor,
               ) -> torch.Tensor:
        """
        returns contrastive logits matrix of shape (B, B)
        """
        # 1) pool each modality
        z_spec = self.pool_spectrum(spec_enc, spec_mask)     # (B, d_model)
        z_pep  = self.pool_peptide(spec_enc, pep_enc, spec_mask)  # (B, d_model)

        # 2) normalize
        z_spec = F.normalize(z_spec, dim=1)
        z_pep  = F.normalize(z_pep,  dim=1)

        # 3) compute B×B scaled cosine logits
        scale  = self.logit_scale.exp()
        logits = scale * (z_spec @ z_pep.t())     # (B, B)

        return logits
