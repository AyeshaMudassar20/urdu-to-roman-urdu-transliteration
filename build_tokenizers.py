"""
Train the two SentencePiece tokenizers used by this project from the raw
parallel text splits.

The trained tokenizer binaries (*.model) are not committed to this repo (see
README) -- run this script once to regenerate them locally before training or
running inference. It writes:

    data/tokenizers/urdu_bpe.model / .vocab   (source side, Urdu script)
    data/tokenizers/eng_bpe.model  / .vocab   (target side, Roman Urdu)

Vocab sizes (54 / 32) match the ones the original checkpoint's embedding
layers were trained with (see model.py).

Usage:
    python build_tokenizers.py --train-ur data/splits/train.ur --train-en data/splits/train.en
"""

import argparse
import os

import sentencepiece as spm

TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "data", "tokenizers")


def train_tokenizer(input_path: str, model_prefix: str, vocab_size: int):
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )


def main():
    parser = argparse.ArgumentParser(description="Train the Urdu / Roman Urdu SentencePiece tokenizers")
    parser.add_argument("--train-ur", type=str, default="data/splits/train.ur")
    parser.add_argument("--train-en", type=str, default="data/splits/train.en")
    parser.add_argument("--ur-vocab-size", type=int, default=54)
    parser.add_argument("--en-vocab-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print(f"Training Urdu tokenizer from {args.train_ur} (vocab_size={args.ur_vocab_size})...")
    train_tokenizer(args.train_ur, os.path.join(TOKENIZER_DIR, "urdu_bpe"), args.ur_vocab_size)

    print(f"Training Roman Urdu tokenizer from {args.train_en} (vocab_size={args.en_vocab_size})...")
    train_tokenizer(args.train_en, os.path.join(TOKENIZER_DIR, "eng_bpe"), args.en_vocab_size)

    print(f"Done. Tokenizers written to {TOKENIZER_DIR}/")


if __name__ == "__main__":
    main()
