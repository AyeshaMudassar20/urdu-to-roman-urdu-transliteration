# Urdu to Roman Urdu Transliteration

A sequence-to-sequence LSTM model that transliterates Urdu-script text into Roman Urdu (Urdu written in the Latin alphabet), trained on 21,000 parallel lines with character-level SentencePiece tokenization.

## Overview

Transliteration differs from translation: the goal isn't to translate meaning between languages, but to re-render the *same* language in a different script — e.g. `وہ گتھی آج تک سلجھا رہا ہوں` → `vo gutthi aaj tak suljha raha huun`. This project frames that as a Seq2Seq problem: an encoder reads the Urdu-script sequence, and a decoder generates the Roman Urdu sequence one token at a time.

## Dataset

21,000 parallel Urdu / Roman Urdu lines, split into:

| Split | Lines |
|---|---|
| Train | 16,800 |
| Validation | 2,100 |
| Test | 2,100 |

Both sides are tokenized with dedicated SentencePiece models trained to near character-level granularity: 54 tokens for the Urdu side, 32 tokens for the Roman Urdu side. This keeps the vocabulary small and lets the model generalize to words it hasn't seen, which matters for a transliteration task where the "vocabulary" is effectively open (any Urdu word can appear).

## Architecture

No attention mechanism — a bridged encoder/decoder LSTM:

- **Encoder:** bidirectional, 2-layer LSTM (embedding dim 256, hidden size 512 per direction)
- **Bridge:** the encoder's final forward+backward hidden and cell states are concatenated (1024-dim) and projected down to a single 512-dim vector via two linear layers (`fc_hidden`, `fc_cell`)
- **Decoder:** unidirectional, 4-layer LSTM (hidden size 512), initialized from the bridged encoder state, with dropout and a final linear layer projecting to the 32-token Roman Urdu vocabulary

This exact architecture (layer counts, dimensions, and parameter names) was verified directly against the tensor shapes in the originally trained checkpoint, so `model.py` matches it precisely.

## Repository Structure

```
urdu-to-roman-urdu-transliteration/
├── model.py              # Encoder / Decoder / Seq2Seq architecture
├── dataset.py            # SentencePiece loading, PyTorch Dataset, padding collate fn
├── build_tokenizers.py   # Trains the two SentencePiece tokenizers from raw text
├── train.py              # Training loop with teacher forcing
├── transliterate.py      # CLI inference (greedy decoding)
├── app.py                # Streamlit demo
├── requirements.txt
├── data/
│   ├── tokenizers/
│   │   ├── urdu_bpe.vocab   # Urdu-side SentencePiece vocab (human-readable)
│   │   └── eng_bpe.vocab    # Roman Urdu-side SentencePiece vocab (human-readable)
│   └── splits/
│       ├── val.ur / val.en     # 15-line sample (full 2,100-line split used for training/eval, not committed)
│       └── test.ur / test.en   # 15-line sample (full 2,100-line split used for training/eval, not committed)
├── LICENSE
└── README.md
```

**Note on data and tokenizers:** the full training/validation/test splits (16,800 / 2,100 / 2,100 lines) and the compiled SentencePiece `.model` binaries are not committed to this repository -- they're large, easily regenerable files, kept out of version control the same way the trained checkpoint is (see below). `data/splits/val.ur`/`val.en` and `test.ur`/`test.en` each ship a 15-line sample so the exact one-line-per-example parallel format is clear. To train or evaluate for real, supply your own full `data/splits/{train,val,test}.{ur,en}` files in the same format and run `build_tokenizers.py` first.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python build_tokenizers.py --train-ur data/splits/train.ur --train-en data/splits/train.en
python train.py --epochs 20 --batch-size 64 --lr 0.001
```

Saves the best checkpoint (by validation loss) to `best_seq2seq_model.pkl` in the repo root.

## Inference

Command line:

```bash
python transliterate.py --checkpoint best_seq2seq_model.pkl --text "تم کیسے ہو"
```

Streamlit demo:

```bash
streamlit run app.py
```

## Note on the trained checkpoint

`best_seq2seq_model.pkl` (~74MB) is **not** included in this repository — it exceeds GitHub's web upload limit. `train.py` will regenerate it locally in about the time it takes to run 20 epochs over the 16,800-line training set on a single GPU. `model.py`, `dataset.py`, `build_tokenizers.py`, and your own training split are everything needed to reproduce it from scratch.

## Future Improvements

- Add an attention mechanism (e.g. Luong or Bahdanau) over encoder outputs — the current model relies purely on a single bridged state, which tends to bottleneck longer inputs
- Report quantitative metrics (character error rate, BLEU) on the test split
- Try beam search decoding instead of greedy decoding for better output quality
- Experiment with a larger/subword-level vocabulary vs. the current near-character-level tokenization

## Author

Ayesha Mudassar — [github.com/AyeshaMudassar20](https://github.com/AyeshaMudassar20)

## License

MIT License — see `LICENSE`.
