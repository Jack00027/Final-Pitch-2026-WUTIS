"""
RelativeValue.py — embedding-implied relative-value signal for the WUTIS
Health Care long-short book.

Implements the three-step valuation framework on the OS-BERT asset embeddings:

  1. Valuation measure. Each quarter, within Health Care, regress log market
     equity on log book equity; the residual p_perp is a book-equity-purged
     valuation level (a generalized market-to-book).

         p_a = gamma * b_a + alpha + p_perp_a        (OLS, cross-sectional)

  2. Embedding-implied fair value. Regress p_perp on the OS-BERT embeddings
     with ridge (k-fold CV); the fitted value is what the firm's valuation
     should be, given how the market prices its embedding peers.

         p_perp_a = beta' x_a + delta + eps_a        (RidgeCV)
         p_perp_hat_a = beta_hat' x_a + delta_hat

  3. Mispricing signal. The ridge residual:

         signal_a = p_perp_a - p_perp_hat_a
         signal < 0  -> cheap vs peers  -> LONG
         signal > 0  -> rich  vs peers  -> SHORT

  Portfolio: rank by signal, long the cheapest tercile, short the richest,
  equal-weight and dollar-neutral (a signal-weighted variant is also emitted).

Rotation note. Per-quarter embeddings are identified only up to rotation. Ridge
*fitted values* are rotation-invariant when the penalty is isotropic (lambda||beta||^2)
and the features are not rescaled per dimension. So we L2-normalize each asset's
embedding onto the unit sphere — a per-row rescaling that commutes with rotation
and matches the cosine geometry the fine-tuning optimized — and let RidgeCV center
internally. We deliberately do NOT per-dimension standardize: that is a
basis-dependent operation that would break the rotation invariance the
cross-quarter signal relies on.

Inputs
  --emb-dir       directory of OS-BERT embeddings: q_YYYY-MM-DD.parquet with
                  columns issuer_id, quarter_end, dim_000 ... dim_(d-1)
  --fundamentals  one panel file (.parquet/.csv) keyed by (issuer_id, quarter_end):
                    me            market equity (USD, > 0)
                    be            book equity  (USD, > 0)
                    gics_sector   GICS sector code (Health Care = 35)
                                  -- or an is_healthcare boolean instead
                  optional point-in-time exclusion flags:
                    pending_ma            bool -> drop if True
                    pre_revenue_biotech   bool -> drop if True
                  Build this by joining your issuer_id -> CUSIP/permno crosswalk
                  to CRSP/Compustat (JKP) characteristics.

Output
  signals/q_YYYY-MM-DD.parquet -- one row per Health Care name with the valuation
  decomposition, signal, ranks, leg, and weights. The 45-day entry lag and P&L
  belong in the downstream backtest, not here; this file is the as-of-quarter signal.

Usage
  python RelativeValue.py --emb-dir embeddings_os --fundamentals fundamentals.parquet
  python RelativeValue.py --test --fundamentals fundamentals.parquet   # embeddings_os/test -> signals/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV


# =========================================================================
# Config
# =========================================================================
def default_config():
    return {
        "emb_dir":         "embeddings_os",
        "fundamentals":    None,
        "sig_dir":         "signals",
        "healthcare_gics": 35,
        "n_buckets":       3,                       # terciles
        "min_names":       30,                      # skip a quarter with fewer HC names
        "ridge_alphas":    np.logspace(-3, 3, 25),
        "cv_folds":        10,
        "winsor":          0.01,                    # clip signal tails before ranking
    }


# =========================================================================
# Helpers
# =========================================================================
def dim_columns(df):
    return [c for c in df.columns if c.startswith("dim_")]


def l2_normalize_rows(X):
    """Project each embedding onto the unit sphere. Commutes with rotation, so
    downstream ridge fitted values stay rotation-invariant."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def norm_quarter(s):
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


def load_fundamentals(path):
    p = Path(path)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    if "quarter_end" not in df.columns:
        raise ValueError("fundamentals must have a 'quarter_end' column")
    df["quarter_end"] = norm_quarter(df["quarter_end"])
    return df


# =========================================================================
# Per-quarter signal
# =========================================================================
def build_quarter_signal(emb_q, fund_q, cfg):
    """Returns a per-name DataFrame with the valuation decomposition, signal,
    ranks, leg, and weights — or None if too few Health Care names survive."""
    df = emb_q.merge(fund_q, on=["issuer_id", "quarter_end"], how="inner")

    # Health Care subset
    if "gics_sector" in df.columns:
        df = df[df["gics_sector"] == cfg["healthcare_gics"]]
    elif "is_healthcare" in df.columns:
        df = df[df["is_healthcare"].astype(bool)]
    else:
        raise ValueError("fundamentals need 'gics_sector' or 'is_healthcare'")

    # optional point-in-time exclusions
    for flag in ("pending_ma", "pre_revenue_biotech"):
        if flag in df.columns:
            df = df[~df[flag].astype(bool)]

    # positive me, be for the logs
    df = df[(df["me"] > 0) & (df["be"] > 0)].copy()
    if len(df) < cfg["min_names"]:
        return None

    # Step 1 -- valuation residual: log me ~ log be
    df["p"] = np.log(df["me"].to_numpy())
    df["b"] = np.log(df["be"].to_numpy())
    ols = LinearRegression().fit(df[["b"]].to_numpy(), df["p"].to_numpy())
    df["p_perp"] = df["p"].to_numpy() - ols.predict(df[["b"]].to_numpy())

    # Step 2 -- embedding-implied fair value (rotation-invariant ridge)
    dims = dim_columns(df)
    X = l2_normalize_rows(df[dims].to_numpy(dtype=float))
    ridge = RidgeCV(alphas=cfg["ridge_alphas"], cv=cfg["cv_folds"],
                    fit_intercept=True).fit(X, df["p_perp"].to_numpy())
    df["p_perp_hat"] = ridge.predict(X)

    # Step 3 -- mispricing signal (log-valuation units)
    df["signal"] = df["p_perp"].to_numpy() - df["p_perp_hat"].to_numpy()

    # winsorize tails, then cross-sectional z-score and percentile rank
    lo, hi = df["signal"].quantile([cfg["winsor"], 1 - cfg["winsor"]])
    df["signal"] = df["signal"].clip(lo, hi)
    s = df["signal"].to_numpy()
    df["signal_z"] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    df["signal_rank"] = df["signal"].rank(pct=True)

    # terciles: long the cheapest (lowest signal), short the richest (highest)
    q = cfg["n_buckets"]
    df["bucket"] = pd.qcut(df["signal"].rank(method="first"), q,
                           labels=range(q)).astype(int)
    df["leg"] = np.where(df["bucket"] == 0, "long",
                np.where(df["bucket"] == q - 1, "short", "flat"))

    # equal-weight, dollar-neutral
    n_long = int((df["leg"] == "long").sum())
    n_short = int((df["leg"] == "short").sum())
    df["weight_ew"] = 0.0
    df.loc[df["leg"] == "long",  "weight_ew"] =  1.0 / max(n_long, 1)
    df.loc[df["leg"] == "short", "weight_ew"] = -1.0 / max(n_short, 1)

    # signal-weighted variant: weight ~ |signal| within each leg, dollar-neutral
    df["weight_sw"] = 0.0
    for leg, sign in (("long", 1.0), ("short", -1.0)):
        m = df["leg"] == leg
        w = df.loc[m, "signal"].abs()
        if w.sum() > 0:
            df.loc[m, "weight_sw"] = sign * (w / w.sum()).to_numpy()

    df.attrs["ridge_alpha"] = float(ridge.alpha_)
    df.attrs["ridge_r2"] = float(ridge.score(X, df["p_perp"].to_numpy()))
    return df


# =========================================================================
# Per-quarter driver
# =========================================================================
def process_quarter(emb_path, fund, cfg):
    q_label = emb_path.stem.replace("q_", "")
    emb_q = pd.read_parquet(emb_path)
    emb_q["quarter_end"] = norm_quarter(emb_q["quarter_end"])

    fund_q = fund[fund["quarter_end"] == emb_q["quarter_end"].iloc[0]]
    if fund_q.empty:
        print(f"  [skip] q_{q_label}: no fundamentals for this quarter")
        return None

    sig = build_quarter_signal(emb_q, fund_q, cfg)
    if sig is None:
        print(f"  [skip] q_{q_label}: fewer than {cfg['min_names']} Health Care names")
        return None

    out_cols = ["issuer_id", "quarter_end", "me", "be", "p_perp", "p_perp_hat",
                "signal", "signal_z", "signal_rank", "leg", "weight_ew", "weight_sw"]
    out = sig[out_cols].sort_values("signal").reset_index(drop=True)

    sig_dir = Path(cfg["sig_dir"])
    sig_dir.mkdir(parents=True, exist_ok=True)
    out_path = sig_dir / f"q_{q_label}.parquet"
    out.to_parquet(out_path, index=False)

    print(f"  q_{q_label}: {len(out)} HC names | "
          f"ridge alpha={sig.attrs['ridge_alpha']:.3g}, R2={sig.attrs['ridge_r2']:.3f} | "
          f"long {int((out['leg']=='long').sum())} / short {int((out['leg']=='short').sum())} "
          f"-> {out_path}")
    return out


# =========================================================================
# Entry point
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", default="embeddings_os")
    parser.add_argument("--fundamentals", default=None,
                        help="panel (.parquet/.csv) with me, be, sector by (issuer_id, quarter_end)")
    parser.add_argument("--sig-dir", default="signals")
    parser.add_argument("--test", action="store_true",
                        help="use embeddings_os/test -> signals/test")
    args = parser.parse_args()

    cfg = default_config()
    cfg["emb_dir"] = "embeddings_os/test" if args.test else args.emb_dir
    cfg["sig_dir"] = "signals/test"       if args.test else args.sig_dir
    cfg["fundamentals"] = args.fundamentals

    if cfg["fundamentals"] is None:
        raise SystemExit(
            "Provide --fundamentals: a panel with me, be, and sector keyed by "
            "(issuer_id, quarter_end). See the module docstring for the schema.")

    fund = load_fundamentals(cfg["fundamentals"])
    files = sorted(Path(cfg["emb_dir"]).glob("q_*.parquet"))
    if not files:
        raise SystemExit(f"No q_*.parquet in {cfg['emb_dir']}")

    print(f"{len(files)} quarter(s) of embeddings\n")
    for f in files:
        process_quarter(f, fund, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()