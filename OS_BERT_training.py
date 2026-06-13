# Train a BERT masked-language model on investor holdings to embed firms.

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


# =========================================================================
# Hyperparameters
# =========================================================================
def default_config():
    """Every hyperparameter lives here in one dict. Edit values to change defaults."""
    return {
        # I/O
        "seq_dir":           "data_wutis",
        "emb_dir":           "embeddings_os",
        "model_dir":         "models_os",
        # Architecture (per the paper)
        "hidden_size":       64,
        "num_layers":        4,
        "num_heads":         2,
        "intermediate_size": 256,
        "context_window":    62,
        "max_position":      64,    # 62 + [CLS] + [SEP]
        # Pretraining
        "mask_prob":         0.15,
        "batch_size":        64,
        "pretrain_lr":       5e-4,
        "pretrain_epochs":   10,
        "weight_decay":      0.01,
        "warmup_frac":       0.1,
        "train_split":       0.9,
        # Sentence-transformer fine-tuning
        "finetune_lr":       2e-4,
        "finetune_epochs":   3,
        # Misc
        "seed":              42,
        "skip_existing":     True,
        "device": ("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu")),
    }


# =========================================================================
# Vocabulary and tokenizer
# =========================================================================
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3, 4


def build_vocab(tokens):
    """Build per-quarter integer<->string mappings from a flat list of tokens.
    For OS-BERT the tokens are investor IDs."""
    unique_tokens = sorted(set(tokens))
    itos = SPECIAL_TOKENS + unique_tokens
    stoi = {tok: i for i, tok in enumerate(itos)}
    return itos, stoi


def build_tokenizer(itos, model_max_length):
    """Wrap our vocab as a HuggingFace tokenizer (needed by Trainer + collator).
    WordLevel = no subword splitting — investor IDs are atomic units."""
    vocab_dict = {tok: i for i, tok in enumerate(itos)}
    backend = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(
        tokenizer_object  = backend,
        pad_token         = "[PAD]",
        unk_token         = "[UNK]",
        cls_token         = "[CLS]",
        sep_token         = "[SEP]",
        mask_token        = "[MASK]",
        model_max_length  = model_max_length,
    )


def encode_tokens(tokens, stoi):
    """Convert investor-id strings to integer ids (unknown -> [UNK])."""
    return [stoi.get(t, UNK_ID) for t in tokens]


def wrap_and_pad(body, max_len):
    """Wrap a token-id list with [CLS]/[SEP] and pad to max_len.

    body: list of integer token ids (just the investors, no special tokens)
    Returns (input_ids, attention_mask) as plain Python lists.
    """
    body = body[: max_len - 2]                          # leave room for [CLS] and [SEP]
    input_ids      = [CLS_ID] + body + [SEP_ID]
    attention_mask = [1] * len(input_ids)               # 1 = real token, 0 = padding
    pad_len = max_len - len(input_ids)
    input_ids      += [PAD_ID] * pad_len
    attention_mask += [0] * pad_len
    return input_ids, attention_mask


# =========================================================================
# Datasets
# =========================================================================
class MLMDataset(Dataset):
    """Yields one tokenized ownership sequence at a time. The data collator
    handles the actual masking later, at batching time, using the tokenizer's
    knowledge of which token ids are special."""

    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids, attention_mask = wrap_and_pad(self.sequences[idx], self.max_len)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class PairDataset(Dataset):
    """Splits each ownership sequence into even/odd ranks for sentence-transformer
    training. Each item is (half_a, half_b) where both halves come from the
    same asset."""

    def __init__(self, sequences, max_len):
        self.sequences = [s for s in sequences if len(s) >= 2]
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def _to_tensors(self, body):
        input_ids, attention_mask = wrap_and_pad(body, self.max_len)
        return (torch.tensor(input_ids,      dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long))

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        evens = seq[0::2]   # ranks 1, 3, 5, ...
        odds  = seq[1::2]   # ranks 2, 4, 6, ...
        return self._to_tensors(evens), self._to_tensors(odds)


# =========================================================================
# Model setup
# =========================================================================
def build_bert_config(vocab_size, cfg):
    return BertConfig(
        vocab_size              = vocab_size,
        hidden_size             = cfg["hidden_size"],
        num_hidden_layers       = cfg["num_layers"],
        num_attention_heads     = cfg["num_heads"],
        intermediate_size       = cfg["intermediate_size"],
        max_position_embeddings = cfg["max_position"],
        type_vocab_size         = 1,
        pad_token_id            = PAD_ID,
        hidden_act              = "gelu",
    )


# =========================================================================
# Small helpers used in the training loops
# =========================================================================
def cosine_lr(step, total_steps, base_lr, warmup_steps):
    """Linear warmup, then cosine decay to zero."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr):
    """Override the learning rate in every optimizer parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over non-pad positions.

    hidden_states:  (batch, seq_len, hidden_size)  - BERT's output vectors
    attention_mask: (batch, seq_len)               - 1 for real, 0 for pad
    Returns:        (batch, hidden_size)           - one vector per asset sequence
    """
    mask = attention_mask.unsqueeze(-1).float()        # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)         # zero out padding, then sum
    counts = mask.sum(dim=1).clamp(min=1)              # number of real tokens
    return summed / counts                             # mean


def contrastive_loss(emb_a, emb_b, temperature=20.0):
    """Symmetric InfoNCE loss.

    For each row i, the matched pair (a_i, b_i) should have a higher cosine
    similarity than any mismatched pair (a_i, b_j) for j != i. We enforce
    this in both directions (rows and columns of the similarity matrix).

    emb_a, emb_b: each (batch, hidden_size), already L2-normalized.
    """
    similarity = emb_a @ emb_b.t() * temperature       # (batch, batch)
    targets = torch.arange(emb_a.size(0), device=emb_a.device)
    loss_a_to_b = F.cross_entropy(similarity,      targets)
    loss_b_to_a = F.cross_entropy(similarity.t(), targets)
    return (loss_a_to_b + loss_b_to_a) / 2


def make_amp_context(device):
    """Return an autocast context manager: bf16 on CUDA, no-op elsewhere.

    Created once and reused with 'with ctx: ...' in the inner training loop.
    """
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# =========================================================================
# Phase 1 — Pre-training 
# =========================================================================
def pretrain_mlm(model, train_dataset, val_dataset, tokenizer, cfg, output_dir):
    """Pre-train BERT with masked-language-modeling, using the standard
    HuggingFace recipe: Trainer + DataCollatorForLanguageModeling.

    The data collator handles the 15% random masking with the 80/10/10 split
    (80% replaced with [MASK], 10% random token, 10% kept), using the
    tokenizer to know which token ids are special and shouldn't be masked.
    """
    use_bf16 = (cfg["device"] == "cuda")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer       = tokenizer,
        mlm             = True,
        mlm_probability = cfg["mask_prob"],
    )

    training_args = TrainingArguments(
        output_dir                  = str(output_dir / "trainer_logs"),
        overwrite_output_dir        = True,
        num_train_epochs            = cfg["pretrain_epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"],
        learning_rate               = cfg["pretrain_lr"],
        weight_decay                = cfg["weight_decay"],
        warmup_ratio                = cfg["warmup_frac"],
        eval_strategy               = "epoch",
        logging_strategy            = "epoch",
        save_strategy               = "no",       # we save manually after
        report_to                   = "none",     # no wandb/tensorboard
        bf16                        = use_bf16,   # mixed precision on CUDA
        seed                        = cfg["seed"],
        dataloader_num_workers      = 0,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = data_collator,
    )
    trainer.train()
    return trainer.state.log_history


# =========================================================================
# Phase 2 — Sentence-transformer fine-tuning (manual loop, bf16 autocast)
# =========================================================================
def encode_for_pooling(model, input_ids, attention_mask):
    """Run BERT and return L2-normalized mean-pooled embeddings, one per
    ownership sequence."""
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = mean_pool(output.last_hidden_state, attention_mask)
    return F.normalize(pooled, dim=-1)


def finetune_sentence_transformer(model, pair_dataset, cfg):
    """Siamese contrastive fine-tuning with in-batch negatives.

    For each batch of (even-half, odd-half) pairs, push matched halves of the
    same asset together and mismatched halves apart in the embedding space.
    """
    device       = cfg["device"]
    epochs       = cfg["finetune_epochs"]
    batch_size   = cfg["batch_size"]
    base_lr      = cfg["finetune_lr"]
    weight_decay = cfg["weight_decay"]
    warmup_frac  = cfg["warmup_frac"]

    model.to(device)
    loader = DataLoader(pair_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=weight_decay)
    total_steps  = max(1, len(loader) * epochs)
    warmup_steps = int(total_steps * warmup_frac)
    amp_ctx      = make_amp_context(device)

    history = {"loss": []}
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_examples = 0

        for (ids_a, attn_a), (ids_b, attn_b) in loader:
            ids_a, attn_a = ids_a.to(device), attn_a.to(device)
            ids_b, attn_b = ids_b.to(device), attn_b.to(device)

            # Schedule: warmup then cosine decay
            set_optimizer_lr(optimizer,
                cosine_lr(step, total_steps, base_lr, warmup_steps))
            optimizer.zero_grad()

            # Forward pass + contrastive loss (in bf16 on CUDA)
            with amp_ctx:
                emb_a = encode_for_pooling(model, ids_a, attn_a)
                emb_b = encode_for_pooling(model, ids_b, attn_b)
                loss = contrastive_loss(emb_a, emb_b)

            # Backward pass + weight update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_n = emb_a.size(0)
            epoch_loss_sum += loss.item() * batch_n
            n_examples += batch_n
            step += 1

        avg_loss = epoch_loss_sum / max(1, n_examples)
        history["loss"].append(avg_loss)
        print(f"    finetune ep{epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    return history


# =========================================================================
# Embedding extraction (bf16 autocast on CUDA)
# =========================================================================
@torch.no_grad()
def compute_asset_embeddings(model, assets_df, stoi, cfg):
    """Run each asset's top-62 owners through BERT, return mean-pooled
    embeddings as a float32 numpy array of shape (N_assets, hidden_size)."""
    device      = cfg["device"]
    batch_size  = cfg["batch_size"]
    hidden_size = cfg["hidden_size"]
    max_len     = cfg["context_window"] + 2
    amp_ctx     = make_amp_context(device)

    model.eval()
    output = np.zeros((len(assets_df), hidden_size), dtype=np.float32)

    for start in range(0, len(assets_df), batch_size):
        end = min(start + batch_size, len(assets_df))

        # Tokenize a batch of ownership sequences into padded id/attention tensors
        batch_ids, batch_attn = [], []
        for j in range(start, end):
            raw_tokens = list(assets_df.iloc[j]["tokens"])
            body = encode_tokens(raw_tokens, stoi)[: cfg["context_window"]]
            ids, attn = wrap_and_pad(body, max_len)
            batch_ids.append(ids)
            batch_attn.append(attn)
        ids_tensor  = torch.tensor(batch_ids,  dtype=torch.long, device=device)
        attn_tensor = torch.tensor(batch_attn, dtype=torch.long, device=device)

        # Forward pass + mean-pool
        with amp_ctx:
            hidden = model(input_ids=ids_tensor,
                           attention_mask=attn_tensor).last_hidden_state
            pooled = mean_pool(hidden, attn_tensor)

        # bf16 -> float32 before going to numpy (parquet can't store bf16)
        output[start:end] = pooled.float().cpu().numpy()

    return output


# =========================================================================
# Small pipeline helpers
# =========================================================================
def split_by_issuer(df, encoded_sequences, train_split, seed):
    """Assign each issuer (not each row) to train or val, then return
    the corresponding lists of encoded sequences. This keeps all chunks of
    a given asset on the same side of the split."""
    rng = random.Random(seed)
    issuer_ids = df["issuer_id"].unique().tolist()
    rng.shuffle(issuer_ids)
    n_train = int(len(issuer_ids) * train_split)
    train_iss = set(issuer_ids[:n_train])

    train_seqs, val_seqs = [], []
    for i, iss in enumerate(df["issuer_id"]):
        if iss in train_iss:
            train_seqs.append(encoded_sequences[i])
        else:
            val_seqs.append(encoded_sequences[i])
    return train_seqs, val_seqs


def save_embeddings_parquet(embeddings, assets_df, hidden_size, out_path):
    """Write one row per asset with metadata columns + dim_000..dim_(d-1).

    Identifier columns mirror the R sequence output: issuer_id is the asset key,
    isin is the extra security-level label (one per issuer, carried through from
    the holdings pipeline). isin is optional so older sequence files without it
    still work.
    """
    meta = {"issuer_id": assets_df["issuer_id"].values}
    if "isin" in assets_df.columns:
        meta["isin"] = assets_df["isin"].values
    meta["quarter_end"] = assets_df["quarter_end"].values
    df = pd.DataFrame(meta)
    for d in range(hidden_size):
        df[f"dim_{d:03d}"] = embeddings[:, d]
    df.to_parquet(out_path, index=False)


# =========================================================================
# Per-quarter pipeline
# =========================================================================
def process_quarter(parquet_path, cfg):
    q_label   = parquet_path.stem.replace("q_", "")
    emb_path  = Path(cfg["emb_dir"])   / f"q_{q_label}.parquet"
    model_dir = Path(cfg["model_dir"]) / f"q_{q_label}"

    if cfg["skip_existing"] and emb_path.exists():
        print(f"  [skip] q_{q_label} (embedding exists)")
        return

    print(f"\n── q_{q_label} ──")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load this quarter
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        print("  (empty quarter, skipping)")
        return
    df["tokens"] = df["tokens"].apply(list)
    sequences = df["tokens"].tolist()

    # Assets with >62 owners are split into multiple chunks by the R pipeline.
    # We train on all chunks but embed only the FIRST chunk (top-62 owners by
    # ownership share) per asset. n_investors_full is constant across an asset's
    # chunks, so the stable sort keeps the original chunk order and keep="first"
    # selects chunk 1.
    first_chunks = (df.sort_values(["issuer_id", "n_investors_full"],
                                   ascending=[True, False])
                      .drop_duplicates("issuer_id", keep="first")
                      .reset_index(drop=True))
    print(f"  {len(df):,} sequences from {len(first_chunks):,} assets")

    # 2. Vocab + HF tokenizer  (vocabulary = investor ids)
    all_tokens = [t for seq in sequences for t in seq]
    itos, stoi = build_vocab(all_tokens)
    vocab_size = len(itos)
    max_len    = cfg["context_window"] + 2

    tokenizer = build_tokenizer(itos, model_max_length=max_len)
    tokenizer.save_pretrained(model_dir / "tokenizer")
    print(f"  vocab: {vocab_size:,} tokens")

    # 3. Encode every sequence, then split issuers into train / val
    encoded = [encode_tokens(seq, stoi) for seq in sequences]
    train_seqs, val_seqs = split_by_issuer(
        df, encoded, cfg["train_split"], cfg["seed"])
    train_dataset = MLMDataset(train_seqs, max_len)
    val_dataset   = MLMDataset(val_seqs,   max_len)

    # 4. Pre-train BERT (MLM) via Trainer
    bert_config = build_bert_config(vocab_size, cfg)
    model = BertForMaskedLM(bert_config)
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")
    pretrain_log = pretrain_mlm(
        model, train_dataset, val_dataset, tokenizer, cfg, model_dir)

    # 5. Save the post-pretrain checkpoint (for ablation studies)
    bert = model.bert  # BERT body without the MLM head
    torch.save(bert.state_dict(), model_dir / "bert_pretrain.pt")
    print(f"  saved pretrain checkpoint -> {model_dir / 'bert_pretrain.pt'}")

    # 6. Sentence-transformer fine-tuning
    pair_dataset = PairDataset(train_seqs, max_len)
    if len(pair_dataset) >= cfg["batch_size"]:
        finetune_history = finetune_sentence_transformer(bert, pair_dataset, cfg)
    else:
        finetune_history = {"loss": []}
        print("  (skipping fine-tuning: too few pairs for one batch)")

    # 7. Asset embeddings on the fine-tuned model
    embeddings = compute_asset_embeddings(bert, first_chunks, stoi, cfg)

    # 8. Save final outputs
    Path(cfg["emb_dir"]).mkdir(parents=True, exist_ok=True)
    save_embeddings_parquet(
        embeddings, first_chunks, cfg["hidden_size"], emb_path)
    torch.save(bert.state_dict(), model_dir / "bert.pt")
    (model_dir / "history.json").write_text(json.dumps(
        {"pretrain_log": pretrain_log, "finetune": finetune_history},
        indent=2, default=str))
    print(f"  saved -> {emb_path}")

    # 9. Free GPU memory before next quarter
    del model, bert, train_dataset, val_dataset, pair_dataset
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()


# =========================================================================
# Entry point
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir",         default="data_wutis")
    parser.add_argument("--emb-dir",         default="embeddings_os")
    parser.add_argument("--model-dir",       default="models_os")
    parser.add_argument("--hidden-size",     type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size",      type=int, default=64)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--no-skip", action="store_true",
                        help="re-train quarters even if embedding exists")
    parser.add_argument("--test", action="store_true",
                        help="use data_wutis/test as input dir")
    args = parser.parse_args()

    cfg = default_config()
    cfg["seq_dir"]         = "data_wutis/test"    if args.test else args.seq_dir
    cfg["emb_dir"]         = "embeddings_os/test" if args.test else args.emb_dir
    cfg["model_dir"]       = "models_os/test"     if args.test else args.model_dir
    cfg["hidden_size"]     = args.hidden_size
    cfg["pretrain_epochs"] = args.pretrain_epochs
    cfg["finetune_epochs"] = args.finetune_epochs
    cfg["batch_size"]      = args.batch_size
    cfg["seed"]            = args.seed
    cfg["skip_existing"]   = not args.no_skip

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    print(f"device         : {cfg['device']}")
    print(f"hidden_size    : {cfg['hidden_size']}")
    print(f"pretrain epochs: {cfg['pretrain_epochs']}")
    print(f"finetune epochs: {cfg['finetune_epochs']}")
    print(f"batch_size     : {cfg['batch_size']}")
    print(f"bf16           : {cfg['device'] == 'cuda'}")
    print(f"seq_dir        : {cfg['seq_dir']}")

    files = sorted(Path(cfg["seq_dir"]).glob("q_*.parquet"))
    if not files:
        print(f"No q_*.parquet files found in {cfg['seq_dir']}")
        return
    print(f"\n{len(files)} quarters to process\n")

    for path in files:
        process_quarter(path, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()