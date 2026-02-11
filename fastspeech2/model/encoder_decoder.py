"""
Encoder and Decoder for FastSpeech2.
"""

from typing import Optional

import torch
import torch.nn as nn

from fastspeech2.model.blocks import FFTBlock, PositionalEncoding


class Encoder(nn.Module):
    """
    FastSpeech2 Encoder.

    Consists of:
    1. Input embedding (character/phoneme to vector)
    2. Positional encoding
    3. Stack of FFT blocks
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [FFTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.d_model = d_model

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder.
        """
        x = self.embedding(x)
        x = x * (self.d_model**0.5)
        x = self.pos_encoding(x)
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, x.size(1), -1)
        else:
            attn_mask = None

        for layer in self.layers:
            x = layer(x, attn_mask)

        return x


class Decoder(nn.Module):
    """
    FastSpeech2 Decoder.

    Consists of:
    1. Positional encoding
    2. Stack of FFT blocks
    3. Linear projection to mel dimension
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        n_mels: int = 80,
    ):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [FFTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(d_model, n_mels)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of decoder.
        """
        x = self.pos_encoding(x)
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, x.size(1), -1)
        else:
            attn_mask = None
        for layer in self.layers:
            x = layer(x, attn_mask)

        mel = self.linear(x)
        return mel
