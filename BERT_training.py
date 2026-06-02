"""Train a BERT masked-language model on investor holdings to embed firms.

Pipeline
--------
1. Load (or synthesise) a holdings table and build per-investor firm sequences.
2. Build a firm vocabulary; firms are the model's "tokens".
3. Train a small BERT-for-MLM: random firms in each portfolio are masked and the
   model learns to predict them from the rest of the portfolio.
4. Export the learned firm embedding matrix (the input word-embeddings) plus a
   helper to query nearest-neighbour firms.

Run
---
    # smoke test on synthetic data (no real data needed):
    python BERT_training.py --synthetic --epochs 3

    # on your own holdings table:
    python BERT_training.py --data path/to/holdings.parquet --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, get_linear_schedule_with_warmup

from Data import (
    FirmVocab,
    HoldingsDataset,
    build_sequences,
    load_holdings,
    make_collate_fn,
    make_synthetic_holdings,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def build_model(vocab_size: int, args: argparse.Namespace) -> BertForMaskedLM:
    """A compact BERT sized for a firm vocabulary rather than natural language."""
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_len,
        pad_token_id=0,  # FirmVocab places [PAD] at id 0
        # Holdings have no inherent order, so a single position/segment is fine,
        # but we keep learned position embeddings for the [CLS] + portfolio layout.
        type_vocab_size=1,
    )
    return BertForMaskedLM(config)


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    # --- data -------------------------------------------------------------- #
    if args.synthetic or not args.data:
        print("Using synthetic holdings.")
        df = make_synthetic_holdings(seed=args.seed)
    else:
        print(f"Loading holdings from {args.data}")
        df = load_holdings(args.data)

    sequences = build_sequences(
        df, min_holdings=args.min_holdings, max_len=args.max_len
    )
    vocab = FirmVocab.from_sequences(sequences, min_count=args.min_firm_count)
    print(f"Investors: {len(sequences):,} | Firm vocab: {len(vocab):,}")

    dataset = HoldingsDataset(sequences, vocab, max_len=args.max_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(vocab, args.mlm_prob),
        num_workers=args.num_workers,
        drop_last=False,
    )

    # --- model ------------------------------------------------------------- #
    model = build_model(len(vocab), args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )

    # --- training loop ----------------------------------------------------- #
    model.train()
    for epoch in range(1, args.epochs + 1):
        running, seen = 0.0, 0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item() * batch["input_ids"].size(0)
            seen += batch["input_ids"].size(0)
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {running/seen:.4f}")
        print(f"Epoch {epoch} done | avg loss {running/seen:.4f}")

    # --- export ------------------------------------------------------------ #
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(out_dir / "vocab.json")
    model.save_pretrained(out_dir / "model")
    export_embeddings(model, vocab, out_dir)
    print(f"Saved model, vocab and firm embeddings to {out_dir}/")


@torch.no_grad()
def export_embeddings(model: BertForMaskedLM, vocab: FirmVocab, out_dir: Path) -> None:
    """Save the learned firm embedding matrix (rows aligned to vocab ids)."""
    emb = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    np.save(out_dir / "firm_embeddings.npy", emb)
    (out_dir / "firm_ids.txt").write_text("\n".join(vocab.itos))
    print(f"Embedding matrix: {emb.shape} (vocab x hidden)")


def nearest_firms(
    firm: str,
    emb: np.ndarray,
    vocab: FirmVocab,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Cosine-nearest firms to a query firm, for sanity-checking embeddings."""
    if firm not in vocab.stoi:
        raise KeyError(f"{firm!r} not in vocabulary")
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    q = normed[vocab.stoi[firm]]
    sims = normed @ q
    order = np.argsort(-sims)
    results = []
    for idx in order:
        if idx == vocab.stoi[firm] or idx in vocab.special_ids:
            continue
        results.append((vocab.itos[idx], float(sims[idx])))
        if len(results) >= k:
            break
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # data
    p.add_argument("--data", type=str, default=None, help="holdings table (.csv/.parquet/.feather)")
    p.add_argument("--synthetic", action="store_true", help="train on synthetic holdings")
    p.add_argument("--min-holdings", type=int, default=2)
    p.add_argument("--min-firm-count", type=int, default=1, help="drop firms held < N times")
    p.add_argument("--max-len", type=int, default=128)
    # model
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="artifacts")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
