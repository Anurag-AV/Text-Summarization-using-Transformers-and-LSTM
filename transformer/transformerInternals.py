import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
import math
from collections import defaultdict, Counter
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# TRANSFORMER INTERNALS
# ============================================================================

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Use -1e4 instead of -1e9 to avoid float16 overflow
            scores = scores.masked_fill(mask == 0, -1e4)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        x = torch.matmul(attention, V)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        x = self.W_o(x)

        return x

# ============================================================================
# FEED FORWARD NETWORK
# ============================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ============================================================================
# ENCODER LAYER (with Pre-LN for better deep network training)
# ============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-LN: Layer norm before attention (better for deep networks)
        normed = self.norm1(x)
        attn_output = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout(attn_output)

        # Pre-LN: Layer norm before feed-forward
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x

# ============================================================================
# DECODER LAYER (with Pre-LN for better deep network training)
# ============================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Pre-LN: Layer norm before self-attention
        normed = self.norm1(x)
        attn_output = self.self_attn(normed, normed, normed, tgt_mask)
        x = x + self.dropout(attn_output)

        # Pre-LN: Layer norm before cross-attention
        normed = self.norm2(x)
        attn_output = self.cross_attn(normed, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(attn_output)

        # Pre-LN: Layer norm before feed-forward
        normed = self.norm3(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x

