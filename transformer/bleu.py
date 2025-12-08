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
# BLEU SCORE
# ============================================================================
class BLEUScore:
    def __init__(self, max_n=4):
        self.max_n = max_n

    def _get_ngrams(self, tokens, n):
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams

    def _modified_precision(self, reference, hypothesis, n):
        """Calculate modified n-gram precision"""
        ref_ngrams = Counter(self._get_ngrams(reference, n))
        hyp_ngrams = Counter(self._get_ngrams(hypothesis, n))

        if not hyp_ngrams:
            return 0.0

        clipped_counts = {}
        for ngram in hyp_ngrams:
            clipped_counts[ngram] = min(hyp_ngrams[ngram], ref_ngrams.get(ngram, 0))

        numerator = sum(clipped_counts.values())
        denominator = sum(hyp_ngrams.values())

        return numerator / denominator if denominator > 0 else 0.0

    def _brevity_penalty(self, reference, hypothesis):
        """Calculate brevity penalty"""
        ref_len = len(reference)
        hyp_len = len(hypothesis)

        if hyp_len > ref_len:
            return 1.0
        elif hyp_len == 0:
            return 0.0
        else:
            return math.exp(1 - ref_len / hyp_len)

    def compute(self, references, hypotheses):
        """
        Compute BLEU score
        references: list of reference token lists
        hypotheses: list of hypothesis token lists
        """
        assert len(references) == len(hypotheses)

        precisions = [[] for _ in range(self.max_n)]
        total_ref_len = 0
        total_hyp_len = 0

        for ref, hyp in zip(references, hypotheses):
            total_ref_len += len(ref)
            total_hyp_len += len(hyp)

            for n in range(1, self.max_n + 1):
                prec = self._modified_precision(ref, hyp, n)
                precisions[n-1].append(prec)

        # Average precisions
        avg_precisions = [sum(p) / len(p) if p else 0.0 for p in precisions]

        # Geometric mean of precisions
        if min(avg_precisions) > 0:
            geo_mean = math.exp(sum(math.log(p) for p in avg_precisions) / self.max_n)
        else:
            geo_mean = 0.0

        # Brevity penalty
        if total_hyp_len > total_ref_len:
            bp = 1.0
        elif total_hyp_len == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - total_ref_len / total_hyp_len)

        bleu = bp * geo_mean

        return {
            'bleu': bleu,
            'precisions': avg_precisions,
            'brevity_penalty': bp,
            'length_ratio': total_hyp_len / total_ref_len if total_ref_len > 0 else 0.0
        }
