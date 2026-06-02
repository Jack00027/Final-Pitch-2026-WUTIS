# Firm Embeddings from Investor Holdings

Building vector representations of firms by training a BERT model on investor holdings data.

## Idea

Treat each investor's portfolio as a "sentence" of firms (tokens). By training a
BERT-style model on these holdings sequences, firms that are held together by
similar investors end up close in the resulting embedding space — yielding a
data-driven notion of firm similarity for investment analysis.

## Structure

- `Data.py` — data loading and preprocessing of investor holdings.
- `BERT_training.py` — model definition and training loop.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # to be added
```
