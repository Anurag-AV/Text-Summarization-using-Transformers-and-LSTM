
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
# BEAM SEARCH
# ============================================================================
class BeamSearch:
    def __init__(self, model, tokenizer, beam_size=4, max_len=128, length_penalty=0.6):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.sos_id = tokenizer.special_tokens['<SOS>']
        self.eos_id = tokenizer.special_tokens['<EOS>']
        self.pad_id = tokenizer.special_tokens['<PAD>']

    def generate(self, src, src_mask):
        """Generate summary using beam search"""
        self.model.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_output = self.model.encode(src, src_mask)

        # Initialize beams: (batch_size, beam_size, seq_len)
        beams = torch.full((batch_size, self.beam_size, 1), self.sos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially


        completed_beams = [[] for _ in range(batch_size)]

        for step in range(self.max_len - 1):
            # Prepare decoder input
            tgt = beams.view(batch_size * self.beam_size, -1)

            # Create target mask
            tgt_len = tgt.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).unsqueeze(0).unsqueeze(0)

            # Expand encoder output for beam search
            encoder_output_expanded = encoder_output.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
            encoder_output_expanded = encoder_output_expanded.view(batch_size * self.beam_size, -1, self.model.d_model)

            src_mask_expanded = src_mask.unsqueeze(1).repeat(1, self.beam_size, 1, 1, 1)
            src_mask_expanded = src_mask_expanded.view(batch_size * self.beam_size, 1, 1, -1)

            # Decode
            with torch.no_grad():
                decoder_output = self.model.decode(tgt, encoder_output_expanded, src_mask_expanded, tgt_mask)
                logits = self.model.fc_out(decoder_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)

            # Reshape log probs
            log_probs = log_probs.view(batch_size, self.beam_size, -1)

            # Calculate scores
            vocab_size = log_probs.size(-1)
            scores = beam_scores.unsqueeze(-1) + log_probs
            scores = scores.view(batch_size, -1)

            # Get top k scores and indices
            top_scores, top_indices = torch.topk(scores, self.beam_size, dim=-1)

            # Calculate which beam and which token
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update beams - maintain structure per batch
            new_beams_by_batch = [[] for _ in range(batch_size)]
            new_scores_by_batch = [[] for _ in range(batch_size)]

            for b in range(batch_size):
                for i in range(self.beam_size):
                    beam_idx = beam_indices[b, i]
                    token_idx = token_indices[b, i]

                    # Get previous beam
                    prev_beam = beams[b, beam_idx]

                    # Check if EOS
                    if token_idx == self.eos_id:
                        # Apply length penalty
                        score = top_scores[b, i] / ((prev_beam.size(0) + 1) ** self.length_penalty)
                        completed_beams[b].append((prev_beam.tolist() + [token_idx.item()], score.item()))
                        # Add a dummy beam to maintain beam_size count
                        new_beams_by_batch[b].append(prev_beam)  # Keep the previous beam
                        new_scores_by_batch[b].append(torch.tensor(-1e9, device=device))  # Very low score
                    else:
                        new_beam = torch.cat([prev_beam, token_idx.unsqueeze(0)])
                        new_beams_by_batch[b].append(new_beam)
                        new_scores_by_batch[b].append(top_scores[b, i])

            # Check if all beams are completed
            if all(len(completed_beams[b]) >= self.beam_size for b in range(batch_size)):
                break

            # Convert to tensor (pad if necessary)
            max_len = max(max(beam.size(0) for beam in batch_beams) for batch_beams in new_beams_by_batch)
            
            all_beams = []
            all_scores = []
            
            for b in range(batch_size):
                batch_beams = []
                for beam in new_beams_by_batch[b]:
                    if beam.size(0) < max_len:
                        padding = torch.full((max_len - beam.size(0),), self.pad_id, dtype=torch.long, device=device)
                        beam = torch.cat([beam, padding])
                    batch_beams.append(beam)
                all_beams.append(torch.stack(batch_beams))
                all_scores.append(torch.stack(new_scores_by_batch[b]))

            beams = torch.stack(all_beams)
            beam_scores = torch.stack(all_scores)

        # Get best beam for each batch
        results = []
        for b in range(batch_size):
            if completed_beams[b]:
                best_beam = max(completed_beams[b], key=lambda x: x[1])[0]
            else:
                best_beam = beams[b, 0].tolist()
            results.append(best_beam)

        return results
