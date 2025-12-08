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

from transformerInternals import PositionalEncoding, MultiHeadAttention, EncoderLayer, DecoderLayer, FeedForward

# ============================================================================
# TRANSFORMER MODEL (with final layer norms for Pre-LN architecture)
# ============================================================================
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_encoder_layers,
                 n_decoder_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # Final layer norms for Pre-LN architecture
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization with adjusted scaling for deeper networks"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

    def encode(self, src, src_mask):
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        # Final layer norm for Pre-LN architecture
        x = self.encoder_norm(x)
        return x

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Final layer norm for Pre-LN architecture
        x = self.decoder_norm(x)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.fc_out(decoder_output)
        return output