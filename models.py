from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMWSTM_ADRAN_Plus(nn.Module):
    """
    Proposed hybrid model with switches for ablation studies.

    Ablation switches:
    - use_transition=False: removes learnable Markov transition matrix.
    - use_attention=False: replaces self-attention with identity representation.
    - use_anomaly=False: removes anomaly-amplification layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_clusters: int = 9,
        sequence_length: int = 30,
        dropout_rate: float = 0.10,
        num_heads: int = 4,
        use_transition: bool = True,
        use_attention: bool = True,
        use_anomaly: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_transition = use_transition
        self.use_attention = use_attention
        self.use_anomaly = use_anomaly

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.mmwstm_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        self.emission_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, num_clusters),
        )
        if use_transition:
            self.transition_matrix = nn.Parameter(torch.randn(num_clusters, num_clusters) * 0.01)
        else:
            self.register_buffer("transition_matrix", torch.eye(num_clusters))
        self.mmwstm_output = nn.Linear(hidden_dim * 2 + num_clusters, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.anomaly_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.adran_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        self.adran_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fusion_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def multi_head_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        q = self.query_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.attn_out(context), weights

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)

        lstm_out, _ = self.mmwstm_lstm(embedded)
        lstm_last = lstm_out[:, -1, :]
        cluster_logits = self.emission_network(lstm_last)
        cluster_probs = torch.softmax(cluster_logits, dim=-1)
        if self.use_transition:
            transition = torch.softmax(self.transition_matrix, dim=-1)
            state_features = torch.matmul(cluster_probs, transition)
        else:
            state_features = cluster_probs
        mmwstm_out = self.mmwstm_output(torch.cat([lstm_last, state_features], dim=1))

        if self.use_attention:
            attn_output, attention_weights = self.multi_head_attention(embedded)
            attn_output = self.layer_norm1(embedded + attn_output)
        else:
            attn_output = embedded
            b, t, _ = x.shape
            attention_weights = torch.zeros(b, self.num_heads, t, t, device=x.device)

        if self.use_anomaly:
            anomaly_weights = self.anomaly_network(attn_output)
            adran_input = attn_output * (1.0 + anomaly_weights)
        else:
            adran_input = attn_output

        gru_out, _ = self.adran_gru(adran_input)
        gru_last = gru_out[:, -1, :]
        adran_out = self.adran_output(gru_last)

        combined = torch.cat([mmwstm_out, adran_out], dim=1)
        fusion_weights = self.fusion_attention(combined)
        fused = fusion_weights[:, 0:1] * mmwstm_out + fusion_weights[:, 1:2] * adran_out
        output = self.output_layer(fused)
        return output, cluster_probs, attention_weights


class TemporalTransformer(nn.Module):
    """Compact transformer baseline for sequence-to-one forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class TCNRegressor(nn.Module):
    """Compact temporal convolution baseline."""

    def __init__(self, input_dim: int, channels=(64, 64, 64), kernel_size: int = 3, dropout: float = 0.10):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels[-1], 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: B,T,F -> B,F,T
        h = self.net(x.transpose(1, 2))
        h = h[:, :, : x.shape[1]]
        return self.head(h[:, :, -1])


class NBEATSLike(nn.Module):
    """Lightweight N-BEATS-like baseline using flattened input sequence."""

    def __init__(self, input_dim: int, sequence_length: int, hidden_dim: int = 128, n_blocks: int = 4, dropout: float = 0.10):
        super().__init__()
        in_dim = input_dim * sequence_length
        blocks = []
        for _ in range(n_blocks):
            blocks.extend([nn.Linear(in_dim if not blocks else hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = x.reshape(x.shape[0], -1)
        return self.head(self.net(h))
