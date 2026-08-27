"""
Dataset / tokenization utilities for the Urdu -> Roman Urdu transliteration task.

Loads the pretrained SentencePiece models in data/tokenizers/ and the parallel
text splits in data/splits/, and exposes a PyTorch Dataset + collate function
that pads batches to the longest sequence.
"""

import os

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizers")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")


def load_tokenizers():
    ur_sp = spm.SentencePieceProcessor(model_file=os.path.join(TOKENIZER_DIR, "urdu_bpe.model"))
    en_sp = spm.SentencePieceProcessor(model_file=os.path.join(TOKENIZER_DIR, "eng_bpe.model"))
    return ur_sp, en_sp


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def encode_line(sp: spm.SentencePieceProcessor, line: str):
    ids = sp.encode(line, out_type=int)
    return [SOS_IDX] + ids + [EOS_IDX]


class TransliterationDataset(Dataset):
    """Pairs of (Urdu ids, Roman Urdu ids) for one split ('train' / 'val' / 'test')."""

    def __init__(self, split: str, ur_sp: spm.SentencePieceProcessor, en_sp: spm.SentencePieceProcessor):
        assert split in {"train", "val", "test"}
        ur_lines = read_lines(os.path.join(SPLITS_DIR, f"{split}.ur"))
        en_lines = read_lines(os.path.join(SPLITS_DIR, f"{split}.en"))
        assert len(ur_lines) == len(en_lines), "Source/target line counts must match"

        self.pairs = list(zip(ur_lines, en_lines))
        self.ur_sp = ur_sp
        self.en_sp = en_sp

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ur_line, en_line = self.pairs[idx]
        src_ids = torch.tensor(encode_line(self.ur_sp, ur_line), dtype=torch.long)
        trg_ids = torch.tensor(encode_line(self.en_sp, en_line), dtype=torch.long)
        return src_ids, trg_ids


def collate_batch(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, trg_padded
