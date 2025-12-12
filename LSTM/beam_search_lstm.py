import torch
import torch.nn.functional as F

# ============================================================================
# BEAM SEARCH FOR LSTM
# ============================================================================
class BeamSearchLSTM:
    def __init__(self, model, tokenizer, beam_size=4, max_len=128, length_penalty=0.6):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.sos_id = tokenizer.special_tokens['<SOS>']
        self.eos_id = tokenizer.special_tokens['<EOS>']
        self.pad_id = tokenizer.special_tokens['<PAD>']

    def generate(self, src, src_lens):
        """Generate summary using beam search"""
        self.model.eval()
        batch_size = src.size(0)
        device = src.device

        # Only support batch_size=1 for now
        if batch_size > 1:
            results = []
            for i in range(batch_size):
                result = self.generate(src[i:i+1], src_lens[i:i+1])
                results.extend(result)
            return results

        # Encode
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(src, src_lens)
            hidden, cell = self.model._bridge_hidden(hidden, cell)

        # Initialize beams
        beams = [(torch.tensor([self.sos_id], device=device), 0.0, hidden, cell)]
        completed_beams = []

        # encoder_outputs has shape (batch_size, actual_seq_len, hidden_dim)
        # The actual_seq_len comes from the unpacked sequence
        actual_seq_len = encoder_outputs.size(1)
        
        # Create mask based on actual sequence length from encoder outputs
        # mask should be (batch_size, actual_seq_len)
        mask = torch.ones(batch_size, actual_seq_len, dtype=torch.bool, device=device)
        
        # Set positions beyond the actual source length to False
        src_len = src_lens[0].item()
        if src_len < actual_seq_len:
            mask[:, src_len:] = False

        for step in range(self.max_len - 1):
            candidates = []

            for seq, score, h, c in beams:
                if seq[-1].item() == self.eos_id:
                    completed_beams.append((seq, score))
                    continue

                # Get last token
                input_token = seq[-1]

                # Decode
                with torch.no_grad():
                    output, new_h, new_c, _ = self.model.decoder(
                        input_token.unsqueeze(0), h, c, encoder_outputs, mask
                    )
                    log_probs = F.log_softmax(output, dim=-1)

                # Get top k tokens
                top_log_probs, top_indices = torch.topk(log_probs[0], self.beam_size)

                for log_prob, idx in zip(top_log_probs, top_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score, new_h, new_c))

            if not candidates:
                break

            # Sort by score and keep top beam_size
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_size]

        # Add remaining beams to completed
        for seq, score, _, _ in beams:
            completed_beams.append((seq, score))

        # Get best beam
        if completed_beams:
            # Apply length penalty
            best_beam = max(completed_beams,
                          key=lambda x: x[1] / (len(x[0]) ** self.length_penalty))
            return [best_beam[0].cpu().tolist()]
        else:
            return [[self.sos_id, self.eos_id]]