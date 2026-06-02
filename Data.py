"""Data pipeline for firm embeddings from investor holdings.

Core idea
---------
Each investor's portfolio is treated as a "sentence" whose "words" are the firms
they hold. A BERT masked-language model trained on these sequences learns firm
embeddings where firms co-held by similar investors are close together.

This module handles:
  * loading (or synthesising) a long holdings table  -> build_*/load_holdings
  * grouping holdings into per-investor firm sequences -> build_sequences
  * building a firm vocabulary                          -> FirmVocab
  * a torch Dataset + an MLM collator for training      -> HoldingsDataset, mlm_collate

The real-data path expects a table with one row per (investor, firm) holding.
The synthetic path lets you run the whole pipeline end-to-end with no data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class FirmVocab:
    """Maps firm identifiers (e.g. tickers/CUSIPs) to integer token ids."""

    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"
    MASK = "[MASK]"
    SPECIAL_TOKENS = [PAD, UNK, CLS, MASK]

    def __init__(self, firms: Iterable[str]):
        # Special tokens occupy the first ids so PAD == 0.
        self.itos: list[str] = list(self.SPECIAL_TOKENS) + list(firms)
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]

    @property
    def cls_id(self) -> int:
        return self.stoi[self.CLS]

    @property
    def mask_id(self) -> int:
        return self.stoi[self.MASK]

    @property
    def special_ids(self) -> set[int]:
        return {self.stoi[t] for t in self.SPECIAL_TOKENS}

    def encode(self, firms: Sequence[str]) -> list[int]:
        return [self.stoi.get(f, self.unk_id) for f in firms]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FirmVocab":
        itos = json.loads(Path(path).read_text())
        # Reconstruct directly to preserve exact id ordering.
        vocab = cls.__new__(cls)
        vocab.itos = itos
        vocab.stoi = {tok: i for i, tok in enumerate(itos)}
        return vocab

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[str]], min_count: int = 1) -> "FirmVocab":
        counts: dict[str, int] = {}
        for seq in sequences:
            for firm in seq:
                counts[firm] = counts.get(firm, 0) + 1
        # Sort by frequency (desc) then name for deterministic ids.
        firms = sorted(
            (f for f, c in counts.items() if c >= min_count),
            key=lambda f: (-counts[f], f),
        )
        return cls(firms)


# --------------------------------------------------------------------------- #
# Loading / building holdings
# --------------------------------------------------------------------------- #
def load_holdings(
    path: str | Path,
    investor_col: str = "investor_id",
    firm_col: str = "firm_id",
    weight_col: str | None = "weight",
) -> pd.DataFrame:
    """Load a long holdings table (one row per investor-firm holding).

    Accepts .csv / .parquet / .feather. Returns a DataFrame with normalised
    columns: 'investor_id', 'firm_id', and (if available) 'weight'.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".feather":
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)

    rename = {investor_col: "investor_id", firm_col: "firm_id"}
    if weight_col and weight_col in df.columns:
        rename[weight_col] = "weight"
    df = df.rename(columns=rename)

    df["investor_id"] = df["investor_id"].astype(str)
    df["firm_id"] = df["firm_id"].astype(str)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    return df[["investor_id", "firm_id", "weight"]]


def build_sequences(
    df: pd.DataFrame,
    min_holdings: int = 2,
    max_len: int = 128,
    sort_by_weight: bool = True,
) -> list[list[str]]:
    """Group a holdings table into one firm sequence per investor.

    If ``sort_by_weight`` is set, each investor's firms are ordered by holding
    size (largest first) so that truncation to ``max_len`` keeps the most
    significant positions.
    """
    sequences: list[list[str]] = []
    for _, group in df.groupby("investor_id", sort=False):
        if sort_by_weight and "weight" in group.columns:
            group = group.sort_values("weight", ascending=False)
        firms = group["firm_id"].tolist()
        if len(firms) < min_holdings:
            continue
        sequences.append(firms[:max_len])
    return sequences


def make_synthetic_holdings(
    n_investors: int = 2000,
    n_firms: int = 500,
    n_sectors: int = 12,
    avg_holdings: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic holdings with latent 'sector' structure.

    Firms belong to hidden sectors; investors favour a couple of sectors, so
    firms in the same sector are co-held and should cluster in embedding space.
    Useful for smoke-testing the full training pipeline without real data.
    """
    rng = np.random.default_rng(seed)
    firm_sector = rng.integers(0, n_sectors, size=n_firms)
    firms_by_sector = {s: np.where(firm_sector == s)[0] for s in range(n_sectors)}

    rows = []
    for inv in range(n_investors):
        # Each investor tilts toward 1-3 sectors.
        n_pref = rng.integers(1, 4)
        prefs = rng.choice(n_sectors, size=n_pref, replace=False)
        n_hold = max(2, int(rng.poisson(avg_holdings)))
        held: set[int] = set()
        for _ in range(n_hold):
            # 85% of picks come from preferred sectors, 15% are noise.
            if rng.random() < 0.85:
                sector = rng.choice(prefs)
                pool = firms_by_sector[sector]
            else:
                pool = np.arange(n_firms)
            if len(pool) == 0:
                continue
            held.add(int(rng.choice(pool)))
        for firm in held:
            rows.append((f"INV{inv:05d}", f"FIRM{firm:04d}", float(rng.uniform(0.1, 10.0))))

    return pd.DataFrame(rows, columns=["investor_id", "firm_id", "weight"])


# --------------------------------------------------------------------------- #
# Dataset + MLM collator
# --------------------------------------------------------------------------- #
@dataclass
class HoldingsDataset(Dataset):
    """Encodes firm sequences into token-id tensors with a leading [CLS]."""

    sequences: list[list[str]]
    vocab: FirmVocab
    max_len: int = 128

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        firms = self.sequences[idx][: self.max_len - 1]  # leave room for [CLS]
        ids = [self.vocab.cls_id] + self.vocab.encode(firms)
        return torch.tensor(ids, dtype=torch.long)


def mlm_collate(
    batch: list[torch.Tensor],
    vocab: FirmVocab,
    mlm_prob: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Pad a batch and apply BERT-style masked-language-model masking.

    Returns input_ids, attention_mask and labels (-100 at unmasked positions).
    Of the selected tokens: 80% -> [MASK], 10% -> random firm, 10% unchanged.
    """
    pad_id = vocab.pad_id
    max_len = max(seq.size(0) for seq in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        input_ids[i, : seq.size(0)] = seq
        attention_mask[i, : seq.size(0)] = 1

    labels = input_ids.clone()

    # Candidate positions: real tokens that are not special and not padding.
    special_ids = torch.tensor(sorted(vocab.special_ids))
    is_special = torch.isin(input_ids, special_ids)
    maskable = (attention_mask == 1) & ~is_special

    prob = torch.full(input_ids.shape, mlm_prob)
    prob[~maskable] = 0.0
    selected = torch.bernoulli(prob).bool()

    labels[~selected] = -100  # only compute loss on selected positions

    # 80% -> [MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & selected
    input_ids[replace_mask] = vocab.mask_id

    # 10% -> random firm token (exclude the special-token id range)
    random_fill = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
        & selected
        & ~replace_mask
    )
    random_firms = torch.randint(
        len(vocab.SPECIAL_TOKENS), len(vocab), input_ids.shape, dtype=torch.long
    )
    input_ids[random_fill] = random_firms[random_fill]
    # remaining ~10% of selected positions are left unchanged

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_collate_fn(vocab: FirmVocab, mlm_prob: float = 0.15):
    """Return a picklable collate_fn bound to a vocab (for DataLoader workers)."""
    return partial(mlm_collate, vocab=vocab, mlm_prob=mlm_prob)


if __name__ == "__main__":
    # Quick smoke test of the data pipeline.
    df = make_synthetic_holdings(n_investors=200, n_firms=80, seed=1)
    print(f"holdings rows: {len(df):,}")
    seqs = build_sequences(df, min_holdings=2, max_len=64)
    print(f"investor sequences: {len(seqs)}  (example len {len(seqs[0])})")
    vocab = FirmVocab.from_sequences(seqs)
    print(f"vocab size (incl. specials): {len(vocab)}")
    ds = HoldingsDataset(seqs, vocab, max_len=64)
    collate = make_collate_fn(vocab)
    batch = collate([ds[i] for i in range(4)])
    print({k: tuple(v.shape) for k, v in batch.items()})
    print("masked positions in batch:", int((batch["labels"] != -100).sum()))
