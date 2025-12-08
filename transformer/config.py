
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
# HYPERPARAMETERS
# ============================================================================
class Config:
    # Data parameters
    train_file = '../data/train.csv'
    val_file = '../data/validation.csv'
    test_file = '../data/test.csv'

    # BPE Tokenizer parameters
    vocab_size = 50000
    tokenizer_file = 'bpe_tokenizer_all.pkl'

    # Model parameters (INCREASED DEPTH)
    d_model = 512
    n_heads = 8
    n_encoder_layers = 8  # INCREASED from 4 to 8 (2x deeper)
    n_decoder_layers = 8  # INCREASED from 4 to 8 (2x deeper)
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 512
    max_summary_len = 64

    # Training parameters (adjusted for deeper network)
    batch_size = 24  # Slightly reduced due to increased memory from deeper model
    learning_rate = 0.0005  # Reduced for more stable training with deeper network
    n_epochs = 10
    warmup_steps = 4000  # Increased warmup for deeper network
    gradient_clip = 0.5  # Tighter clipping for deeper network stability
    label_smoothing = 0.1
    gradient_accumulation_steps = 2  # Increased to maintain effective batch size

    # Data sampling for faster training
    max_train_samples = 250
    max_val_samples = 100
    max_test_samples = 100

    # Generation parameters
    beam_size = 5
    length_penalty = 0.6
    top_k_summaries = 5
    max_generation_batches = 100

    # Model saving
    model_file = 'transformer_summarization_deep.pt'
    vocab_file = 'vocabulary.pkl'

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Performance optimizations
    num_workers = 0
    pin_memory = True if torch.cuda.is_available() else False
    compile_model = False
    use_flash_attention = False

    # Progress settings
    log_interval = 50
    validate_every_n_epochs = 1

