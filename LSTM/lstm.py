import torch
import torch.nn as nn
import math

# ============================================================================
# SEQ2SEQ MODEL
# ============================================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Bridge to convert encoder hidden states to decoder hidden states
        if encoder.bidirectional:
            self.bridge_h = nn.Linear(encoder.hidden_dim * 2, decoder.hidden_dim)
            self.bridge_c = nn.Linear(encoder.hidden_dim * 2, decoder.hidden_dim)
        else:
            self.bridge_h = None
            self.bridge_c = None

    def _bridge_hidden(self, hidden, cell):
        """Convert encoder hidden states to decoder hidden states"""
        if self.encoder.bidirectional:
            # Concatenate forward and backward hidden states
            # hidden: (n_layers * 2, batch_size, hidden_dim)
            batch_size = hidden.shape[1]

            # Reshape to separate layers and directions
            hidden = hidden.view(self.encoder.n_layers, 2, batch_size, self.encoder.hidden_dim)
            cell = cell.view(self.encoder.n_layers, 2, batch_size, self.encoder.hidden_dim)

            # Concatenate forward and backward for each layer
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
            cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

            # Project to decoder hidden dim
            hidden = torch.tanh(self.bridge_h(hidden))
            cell = torch.tanh(self.bridge_c(cell))

        return hidden, cell

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        """
        src: (batch_size, src_len)
        src_lens: (batch_size,)
        tgt: (batch_size, tgt_len)
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)

        # Bridge encoder hidden to decoder hidden
        hidden, cell = self._bridge_hidden(hidden, cell)

        # Create mask for attention
        mask = (src != 0)

        # First input to decoder is <SOS> token
        input = tgt[:, 0]

        for t in range(1, tgt_len):
            # Decode
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)

            # Store output
            outputs[:, t, :] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            # Get the highest predicted token
            top1 = output.argmax(1)

            # Use teacher forcing or predicted token
            input = tgt[:, t] if teacher_force else top1

        return outputs