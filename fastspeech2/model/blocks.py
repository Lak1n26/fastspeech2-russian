"""
Basic building blocks for FastSpeech2 model.

This module contains fundamental components:
- Multi-Head Attention (implemented from scratch, no nn.Transformer!)
- Feed-Forward Network
- Positional Encoding
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        """
        batch_size = query.size(0)

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        if torch.isnan(Q).any() or torch.isinf(Q).any():
            Q = torch.nan_to_num(Q, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(K).any() or torch.isinf(K).any():
            K = torch.nan_to_num(K, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(V).any() or torch.isinf(V).any():
            V = torch.nan_to_num(V, nan=0.0, posinf=1.0, neginf=-1.0)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = torch.clamp(scores, min=-50, max=50)
        if torch.isnan(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0)

        if mask is not None:
            mask = mask.to(scores.device)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        if torch.isnan(context).any():
            context = torch.nan_to_num(context, nan=0.0)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.out_linear(context)
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)

        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        """
        seq_len = x.size(1)
        max_len = self.pe.size(1)

        if seq_len > max_len:
            device = x.device
            d_model = self.pe.size(2)
            pe_extended = torch.zeros(1, seq_len, d_model, device=device, dtype=x.dtype)

            pe_extended[:, :max_len, :] = self.pe.to(device)
            position = torch.arange(
                max_len, seq_len, dtype=torch.float, device=device
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=device).float()
                * (-math.log(10000.0) / d_model)
            )

            pe_extended[:, max_len:, 0::2] = torch.sin(position * div_term)
            pe_extended[:, max_len:, 1::2] = torch.cos(position * div_term)

            x = x + pe_extended
        else:
            x = x + self.pe[:, :seq_len, :].to(x.device)

        return self.dropout(x)


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer (FFT) Block.
    """

    def __init__(
        self, d_model: int, num_heads: int = 4, d_ff: int = 1024, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of FFT block.
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class ConvBlock(nn.Module):
    """
    Convolutional block with layer normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        return x
