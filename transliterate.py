"""
Command-line inference for the Urdu -> Roman Urdu Seq2Seq model.

Loads the trained checkpoint (best_seq2seq_model.pkl, not included in this
repo -- see README for how to obtain or retrain it) and greedily decodes a
Roman Urdu transliteration for a given Urdu input string.

Usage:
    python transliterate.py --checkpoint best_seq2seq_model.pkl --text "تم کیسے ہو"
"""

import argparse

import torch

from dataset import EOS_IDX, SOS_IDX, load_tokenizers
from model import build_model


def transliterate(model, ur_sp, en_sp, text: str, device: torch.device, max_len: int = 100) -> str:
    ids = [SOS_IDX] + ur_sp.encode(text, out_type=int) + [EOS_IDX]
    src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    output_ids = model.greedy_decode(src, max_len=max_len, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
    return en_sp.decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Transliterate Urdu text into Roman Urdu")
    parser.add_argument("--checkpoint", type=str, default="best_seq2seq_model.pkl")
    parser.add_argument("--text", type=str, required=True, help="Urdu text to transliterate")
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ur_sp, en_sp = load_tokenizers()
    model = build_model(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    result = transliterate(model, ur_sp, en_sp, args.text, device, max_len=args.max_len)
    print(result)


if __name__ == "__main__":
    main()
