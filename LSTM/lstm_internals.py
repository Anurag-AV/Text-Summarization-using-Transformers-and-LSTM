import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# LSTM ENCODER
# ============================================================================
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, bidirectional=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, src, src_lens):
        """
        src: (batch_size, seq_len)
        src_lens: (batch_size,)
        """
        # Embed
        embedded = self.dropout(self.embedding(src))

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        packed_outputs, (hidden, cell) = self.lstm(packed)

        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)


        return outputs, hidden, cell


# ============================================================================
# ATTENTION MECHANISM
# ============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim, bidirectional_encoder=True):
        super().__init__()

        encoder_dim = hidden_dim * 2 if bidirectional_encoder else hidden_dim

        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        hidden: (batch_size, hidden_dim) - current decoder hidden state
        encoder_outputs: (batch_size, src_len, encoder_dim)
        mask: (batch_size, src_len)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        # Apply mask
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e4)

        # Calculate attention weights
        attention_weights = F.softmax(attention, dim=1)

        return attention_weights


# ============================================================================
# LSTM DECODER WITH ATTENTION
# ============================================================================
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, bidirectional_encoder=True):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.encoder_dim = hidden_dim * 2 if bidirectional_encoder else hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.attention = Attention(hidden_dim, bidirectional_encoder)

        self.lstm = nn.LSTM(
            embedding_dim + self.encoder_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim + self.encoder_dim + embedding_dim, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        """
        input: (batch_size,) - current token
        hidden: (n_layers, batch_size, hidden_dim)
        cell: (n_layers, batch_size, hidden_dim)
        encoder_outputs: (batch_size, src_len, encoder_dim)
        mask: (batch_size, src_len)
        """
        # input: (batch_size,) -> (batch_size, 1)
        input = input.unsqueeze(1)

        # Embed
        embedded = self.dropout(self.embedding(input))

        # Calculate attention weights using top layer hidden state
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)

        # Apply attention to encoder outputs
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        # Concatenate embedded input and context
        lstm_input = torch.cat((embedded, context), dim=2)

        # LSTM forward
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Make prediction
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))

        return prediction, hidden, cell, attn_weights.squeeze(1)