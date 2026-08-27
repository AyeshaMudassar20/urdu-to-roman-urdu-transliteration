"""
Seq2Seq LSTM architecture for Urdu -> Roman Urdu transliteration.

This module defines the exact encoder/decoder architecture that the trained
checkpoint (best_seq2seq_model.pkl) was fit to. The layer shapes and parameter
names below were verified directly against the saved state_dict, so a
checkpoint trained with this file will load with `load_state_dict` without
any key mismatches:

    encoder.embedding.weight        (INPUT_VOCAB_SIZE, EMBED_DIM)
    encoder.lstm.*                  bidirectional, ENC_LAYERS layers
    encoder.fc_hidden / fc_cell     bridge: 2*HIDDEN_DIM -> HIDDEN_DIM
    decoder.embedding.weight        (OUTPUT_VOCAB_SIZE, EMBED_DIM)
    decoder.lstm.*                  unidirectional, DEC_LAYERS layers
    decoder.fc_out                  HIDDEN_DIM -> OUTPUT_VOCAB_SIZE

There is no attention mechanism: the encoder's final bidirectional hidden and
cell states are projected down to a single HIDDEN_DIM vector and used to
initialize every layer of the decoder.

Note: dropout probability has no learnable parameters and therefore cannot be
recovered from a saved state_dict. DROPOUT below is a reasonable default
(0.3); adjust it if you are retraining from scratch and want a different value.
"""

import random

import torch
import torch.nn as nn

# Architecture constants, verified against the trained checkpoint's tensor shapes.
INPUT_VOCAB_SIZE = 54    # Urdu-side SentencePiece vocab (near character-level)
OUTPUT_VOCAB_SIZE = 32   # Roman Urdu-side SentencePiece vocab (near character-level)
EMBED_DIM = 256
HIDDEN_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 4
DROPOUT = 0.3

PAD_IDX = 0
SOS_IDX = 2  # <s>
EOS_IDX = 3  # </s>


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=INPUT_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=ENC_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bridge: bidirectional final state (2 * hidden_dim) -> single hidden_dim vector
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final layer's forward/backward states.
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)

        hidden_bridged = torch.tanh(self.fc_hidden(hidden_cat))
        cell_bridged = torch.tanh(self.fc_cell(cell_cat))

        return outputs, hidden_bridged, cell_bridged


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim=OUTPUT_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=DEC_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell):
        # input_token: (batch,) -> (batch, 1)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        _, hidden, cell = self.encoder(src)

        # Replicate the bridged encoder state across every decoder LSTM layer.
        num_dec_layers = self.decoder.lstm.num_layers
        hidden = hidden.unsqueeze(0).repeat(num_dec_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(num_dec_layers, 1, 1)

        input_token = trg[:, 0]  # <s>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, max_len=100, sos_idx=SOS_IDX, eos_idx=EOS_IDX):
        """Greedy decoding for a single (batch=1) source sequence at inference time."""
        self.eval()
        _, hidden, cell = self.encoder(src)

        num_dec_layers = self.decoder.lstm.num_layers
        hidden = hidden.unsqueeze(0).repeat(num_dec_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(num_dec_layers, 1, 1)

        input_token = torch.tensor([sos_idx], device=self.device)
        output_ids = []

        for _ in range(max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)
            token_id = top1.item()
            if token_id == eos_idx:
                break
            output_ids.append(token_id)
            input_token = top1

        return output_ids


def build_model(device: torch.device) -> Seq2Seq:
    encoder = Encoder()
    decoder = Decoder()
    return Seq2Seq(encoder, decoder, device).to(device)
