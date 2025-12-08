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
# SUMMARIZER
# ============================================================================
class SummarizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_article_len, max_summary_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        article = str(self.df.iloc[idx]['article'])
        summary = str(self.df.iloc[idx]['highlights'])

        # Encode
        article_ids = self.tokenizer.encode(article, add_special_tokens=True)
        summary_ids = self.tokenizer.encode(summary, add_special_tokens=True)

        # Truncate
        article_ids = article_ids[:self.max_article_len]
        summary_ids = summary_ids[:self.max_summary_len]

        return {
            'article': torch.tensor(article_ids, dtype=torch.long),
            'summary': torch.tensor(summary_ids, dtype=torch.long)
        }