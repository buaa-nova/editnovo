import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformers import PeakEncoder


##############################################
# 新增的基于差分注意力的 Transformer Encoder 层
##############################################
class DiffTransformerEncoderLayer(nn.Module):
    """
    带差分注意力的 Transformer Encoder 层。
    在标准多头注意力中，额外加入一项 φ(|mz_i - mz_j|)。
    """

    def __init__(self, dim_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.dim_model = dim_model
        self.n_head = n_head
        self.d_k = dim_model // n_head

        # 定义 Q, K, V 线性映射
        self.W_q = nn.Linear(dim_model, dim_model)
        self.W_k = nn.Linear(dim_model, dim_model)
        self.W_v = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        # 定义 φ 函数：一个小的 MLP，用于将 m/z 差值映射为标量
        self.diff_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

        # 前馈网络
        self.feedforward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mz, src_key_padding_mask=None):
        """
        参数：
          x: Tensor，形状 (B, L, dim_model)，表示各 token 的嵌入
          mz: Tensor，形状 (B, L, 1)，每个 token 对应的 m/z 值。
              对于全局 token，我们规定其 m/z 为 0，不参与差分计算。
          src_key_padding_mask: 可选的 padding 掩码，形状 (B, L)
        返回：
          x: Tensor，形状 (B, L, dim_model)，经过自注意力和前馈网络处理后的输出。
        """
        B, L, _ = x.shape

        # 计算 Q, K, V
        Q = self.W_q(x)  # (B, L, dim_model)
        K = self.W_k(x)  # (B, L, dim_model)
        V = self.W_v(x)  # (B, L, dim_model)

        # 重塑为多头形式：先变成 (B, L, n_head, d_k) 再转置为 (B, n_head, L, d_k)
        Q = Q.view(B, L, self.n_head, self.d_k).transpose(1, 2)  # (B, n_head, L, d_k)
        K = K.view(B, L, self.n_head, self.d_k).transpose(1, 2)  # (B, n_head, L, d_k)
        V = V.view(B, L, self.n_head, self.d_k).transpose(1, 2)  # (B, n_head, L, d_k)

        # 计算标准注意力分数: (Q·K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, n_head, L, L)

        # 计算 m/z 差分项
        # mz: (B, L, 1) --> 计算每一对 token 的差值: diff_{ij} = |mz_i - mz_j|
        diff = torch.abs(mz - mz.transpose(1, 2))  # (B, L, L)
        # 对于全局 token（index 0），不加入差分信息：将对应行和列置 0
        diff[:, 0, :] = 0
        diff[:, :, 0] = 0

        # 将 diff 送入 φ 函数：注意输入需要扩展最后一维
        diff_term = self.diff_mlp(diff.unsqueeze(-1)).squeeze(-1)  # (B, L, L)
        # 将 diff_term 加到所有注意力头上：扩展维度 (B, 1, L, L)
        scores = scores + diff_term.unsqueeze(1)

        # 如果提供了 padding 掩码，则将掩码位置的分数置为 -inf
        if src_key_padding_mask is not None:
            # src_key_padding_mask: (B, L)，扩展为 (B, 1, 1, L)
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)  # (B, n_head, L, L)
        attn = self.dropout(attn)
        # 计算上下文向量
        context = torch.matmul(attn, V)  # (B, n_head, L, d_k)
        # 将多头结果拼接：先转置为 (B, L, n_head, d_k) 再 reshape 为 (B, L, dim_model)
        context = context.transpose(1, 2).contiguous().view(B, L, self.dim_model)
        # 经过线性映射得到最终输出
        x2 = self.out_proj(context)
        # 残差连接和 LayerNorm
        x = self.norm1(x + self.dropout(x2))
        # 前馈网络
        x_ff = self.feedforward(x)
        x = self.norm2(x + x_ff)
        return x


class DiffTransformerEncoder(nn.Module):
    """
    堆叠多个 DiffTransformerEncoderLayer 构成的 Transformer Encoder。
    """

    def __init__(self, num_layers, dim_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DiffTransformerEncoderLayer(dim_model, n_head, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mz, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, mz, src_key_padding_mask)
        return x


##############################################
# 修改后的 SpectrumEncoder
##############################################
class SpectrumEncoderRefine(nn.Module):
    """
    修改后的质谱编码器，基于 Transformer Encoder 并引入差分注意力机制。

    参数说明与原版类似，新增的功能是利用 m/z 差分调整注意力分数，
    从而使模型能够自动“学习”哪些峰差值（接近氨基酸质量）更重要。
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
        super().__init__()
        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        if peak_encoder:
            # 这里假设 PeakEncoder 已经定义好
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = nn.Linear(2, dim_model)

        # 使用我们自定义的 DiffTransformerEncoder 替换原有的 TransformerEncoder
        self.transformer_encoder = DiffTransformerEncoder(
            num_layers=n_layers,
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, spectra):
        """
        参数：
          spectra: Tensor，形状 (B, n_peaks, 2)，其中每个峰是 (m/z, intensity)
                   注意输入已进行零填充，使得每个光谱的峰数量相同。
        返回：
          latent: Tensor，形状 (B, n_peaks+1, dim_model)，包含全局 token 及各峰的编码
          mask: Tensor，形状 (B, n_peaks+1)，指示哪些位置为填充
        """
        # 计算哪些峰为填充：对每个峰的两个值求和，若和为 0 则认为是填充
        zeros = ~spectra.sum(dim=2).bool()  # (B, n_peaks)
        # 构造 padding 掩码：全局 token 不为填充（False），后面接 zeros
        mask = [torch.zeros((spectra.shape[0], 1), dtype=torch.bool, device=spectra.device), zeros]
        mask = torch.cat(mask, dim=1)  # (B, n_peaks+1)

        # 对峰进行编码：得到 (B, n_peaks, dim_model)
        peaks = self.peak_encoder(spectra)
        # 全局光谱表示 token
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
        # 拼接全局 token 和各峰编码，得到 (B, n_peaks+1, dim_model)
        x = torch.cat([latent_spectra, peaks], dim=1)

        # 同时构造 m/z 值张量：全局 token 的 m/z 设为 0，后续各峰取 spectra 中的 m/z（第 0 列）
        global_mz = torch.zeros((spectra.shape[0], 1, 1), device=spectra.device, dtype=spectra.dtype)
        peak_mz = spectra[:, :, 0:1]  # (B, n_peaks, 1)
        mz = torch.cat([global_mz, peak_mz], dim=1)  # (B, n_peaks+1, 1)

        # 将 x 及 m/z 值送入我们的差分注意力 Transformer Encoder
        x = self.transformer_encoder(x, mz, src_key_padding_mask=mask)
        return x, mask

    @property
    def device(self):
        return next(self.parameters()).device
