import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformers import PeakEncoder

# 定义图注意力层
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # 定义可学习的权重矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 定义注意力系数计算的参数
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 定义用于处理 m/z 差值的 MLP
        self.diff_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, mz, adj):
        """
        参数：
        h: 输入的节点特征，形状 (B, N, in_features)
        mz: m/z 值，形状 (B, N, 1)
        adj: 邻接矩阵，形状 (B, N, N)
        返回：
        经过图注意力层处理后的节点特征，形状 (B, N, out_features)
        """
        B, N, _ = h.size()

        # 线性变换
        Wh = torch.matmul(h, self.W)  # (B, N, out_features)

        # 计算注意力系数
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (B, N, N, 2 * out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (B, N, N)

        # 计算 m/z 差分项
        diff = torch.abs(mz - mz.transpose(1, 2))  # (B, N, N)
        # 对于全局 token（index 0），不加入差分信息：将对应行和列置 0
        diff[:, 0, :] = 0
        diff[:, :, 0] = 0
        diff_term = self.diff_mlp(diff.unsqueeze(-1)).squeeze(-1)  # (B, N, N)
        e = e + diff_term  # 将 m/z 差分项加入注意力分数

        # 应用邻接矩阵掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # (B, N, N)
        attention = F.softmax(attention, dim=2)  # (B, N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # (B, N, N)

        # 计算上下文向量
        h_prime = torch.matmul(attention, Wh)  # (B, N, out_features)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, _ = Wh.size()
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, out_features)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, out_features)
        cat_Wh = torch.cat([Wh1, Wh2], dim=3)  # (B, N, N, 2 * out_features)
        return cat_Wh

# 定义图注意力编码器
class GraphAttentionEncoder(nn.Module):
    def __init__(self, in_features, out_features, n_layers, dropout, alpha=0.2):
        super(GraphAttentionEncoder, self).__init__()
        self.layers = nn.ModuleList([
            GraphAttentionLayer(in_features if i == 0 else out_features, out_features, dropout, alpha)
            for i in range(n_layers)
        ])

    def forward(self, x, mz, adj):
        for layer in self.layers:
            x = layer(x, mz, adj)
        return x

# 修改后的质谱编码器
class SpectrumEncoderRefine(nn.Module):
    """
    修改后的质谱编码器，基于图注意力机制。

    参数说明与原版类似，利用图注意力机制来学习峰之间的关系。
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
            alpha=0.2
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

        # 使用我们自定义的 GraphAttentionEncoder 替换原有的 TransformerEncoder
        self.graph_attention_encoder = GraphAttentionEncoder(
            dim_model, dim_model, n_layers, dropout, alpha
        )

    def forward(self, spectra, mz, adj):
        """
        参数：
        spectra: Tensor，形状 (B, n_peaks, 2)，其中每个峰是 (m/z, intensity)
                 注意输入已进行零填充，使得每个光谱的峰数量相同。
        mz: m/z 值，形状 (B, n_peaks, 1)
        adj: 邻接矩阵，形状 (B, n_peaks+1, n_peaks+1)
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

        # 构造全局 token 的 m/z 值（设为 0）
        global_mz = torch.zeros((mz.shape[0], 1, 1), device=mz.device)
        mz = torch.cat([global_mz, mz], dim=1)  # (B, n_peaks+1, 1)

        # 将 x、mz 及邻接矩阵送入我们的图注意力编码器
        x = self.graph_attention_encoder(x, mz, adj)
        return x, mask

    @property
    def device(self):
        return next(self.parameters()).device