# Healthcare Relative Value via Asset Embeddings

### A WUTIS — WU Trading & Investment Society — Investment Pitch

> Build holdings-based **asset embeddings** for US firms with the **OS-BERT**
> methodology of Gabaix, Koijen, Richmond & Yogo (2025), use them to define
> **peer groups** for US Health Care firms, and trade a **sector-neutral,
> embedding-clustered daily residual-reversion** book — long the day's
> underperformers and short the outperformers *within* each embedding peer
> group. The headline test: do embedding peer groups beat GICS industries for
> Health Care relative value?

---

## 1. The idea

Sophisticated investors price firms using a far richer information set than the handful of
accounting ratios and GICS labels analysts typically rely on. The Asset Embeddings paper
shows — theoretically and empirically — that **portfolio holdings encode all information
relevant for prices**, and that low-dimensional vectors learned from holdings (asset
embeddings) capture firm similarity far better than crude industry labels.
We apply this to US Health Care, a sector where industry classification is especially
misleading: a clinical-stage biotech, a dividend-paying pharma major, a med-tech compounder,
and a managed-care payer all sit under "Health Care," yet they are held by entirely different
investor clienteles. We learn each firm's embedding from *who owns it*, **cluster firms into
peer groups** by embedding similarity, and then run a short-horizon **mean-reversion** trade
*within* each cluster: each day, long the names that lagged their peers and short the names
that ran ahead, betting on next-day reversion of the cross-sectional residual.

The embeddings only **define the peer group** — the alpha is short-horizon residual reversion,
the robust, embedding-appropriate use ("find inefficiencies *within* economically similar
groups", rather than "predict the price level"). The pitch claim is therefore a *relative* one
that survives even a modest absolute Sharpe: **embedding peer groups beat GICS industry groups**
for Health Care relative value.

---

## 2. Investment thesis

1. **Holdings reveal true comparables.** Two firms are economically similar if the same kinds
   of investors choose to hold them — not because they share an industry code. Asset embeddings
   recover this similarity directly from ownership.
2. **Embeddings define better peer groups.** Clustering firms on their holdings-based embeddings
   groups together names with a shared investor clientele, which is a cleaner notion of "comparable"
   than a GICS industry bucket — especially in a heterogeneous sector like Health Care.
3. **Within-group deviations mean-revert.** On any given day a name that has run ahead of its
   embedding peers is a candidate short and a laggard is a candidate long. Trading the
   cross-sectional residual (return minus cluster mean, vol-adjusted) isolates relative
   mispricing while netting out cluster- and sector-wide moves.
4. **Health Care is the right proving ground.** High within-sector heterogeneity (biotech vs.
   pharma vs. devices vs. services vs. payers vs. life-science tools) means embedding-based
   peer groups should add the most value precisely where industry buckets fail.

---

## 3. Methodology — OS-BERT asset embeddings

We follow the **Ownership-Shares BERT (OS-BERT)** construction from the paper, the
asset-side dual of PS-BERT. The core idea: treat each **asset as a "sentence"** and each
**investor as a "token."**

### 3.1 Data — WRDS / FactSet Ownership

Quarterly institutional holdings from the FactSet Ownership Data on WRDS
([WUTIS_Data.r](WUTIS_Data.r)):

- **13F holdings** (`factset_own.wrds_own_13f`) within ±7 days of each quarter-end, with
  positive adjusted market value (`adj_mv > 0`).
- Securities are mapped to their **issuer entity** via `own_sec_entity_eq`
  (`fsym_id → factset_entity_id`); **the asset is the issuer**, and holdings are aggregated to
  the (investor, issuer, quarter) grain. A single representative **ISIN** (the security with the
  largest value) is carried through per issuer-quarter to bridge to price/valuation data later.
- **Universe:** the **full US cross-section** (all sectors), so that co-ownership with both
  generalist and specialist investors is visible. We keep firms held by ≥ `MIN_INVESTORS` (10)
  investors and investors holding ≥ `MIN_STOCKS` (10) stocks, iterating the bipartite pruning to
  convergence, and drop investors whose single largest position exceeds `MAX_TOP1_PCT` (70%) of
  the portfolio. Period covered: **2005 → 2026** at quarter-ends.

**Universe design note.** OS-BERT is trained on the full US cross-section (a healthcare name held
mostly by dedicated biotech funds is very different from one held by broad large-cap value funds —
that contrast is signal). The **Health Care restriction is applied at the trading step**, not at
data cleaning: we extract embeddings for every firm, then cluster and trade *only* Health Care
names.

### 3.2 Ownership sequences

For each (asset, quarter), order the firm's investors by **descending ownership share** and
represent each investor as a token (within an asset-quarter everyone holds at the same price, so
descending `adj_mv` is descending ownership share):

```
investor_8214, investor_119, investor_4471, ...
```

Sequences longer than the 62-token context window are split into equal chunks. Output:
one row per (asset, chunk) in `data_wutis/q_YYYY-MM-DD.parquet`.

### 3.3 OS-BERT training (per quarter, two stages)

Implemented in [OS_BERT_training.py](OS_BERT_training.py). The vocabulary is the set of investor
IDs (atomic WordLevel tokens, no subword splitting), rebuilt per quarter.

**Stage 1 — masked-investor pre-training.** A small bidirectional encoder (4 attention layers,
2 heads, hidden size 64, context window 62) is trained with masked-token prediction via the
HuggingFace `Trainer`: mask 15% of investors in each asset's owner list (80% `[MASK]`, 10% random
investor, 10% unchanged) and predict them from the co-owners. The attention mechanism yields a
*contextualized* embedding for each owner, conditioned on the asset's other holders.

**Stage 2 — sentence-transformer fine-tuning.** Split each asset's owner list into even-rank and
odd-rank halves to form positive pairs; negatives are other assets in the batch. A symmetric
InfoNCE / cosine objective pulls the two halves of the same asset together and pushes different
assets apart, so the pooled asset vectors are comparable by **cosine similarity**.

**Embedding extraction.** Mean-pool the contextualized owner vectors over the firm's **top-62
owners** (the first chunk) to obtain one embedding per asset. Output: `embeddings_os/q_*.parquet`
(`issuer_id`, `isin`, `quarter_end`, `dim_000…dim_063`) plus the per-quarter model checkpoints
under `models_os/q_*/`.

| Setting | Value | Rationale |
|---|---|---|
| Embedding dimension (`hidden_size`) | 64 | Paper baseline; raise via `--hidden-size` |
| Attention layers / heads | 4 / 2 | Paper baseline |
| Context window | 62 owners | Paper baseline |
| Masking | 15% (80/10/10) | Devlin et al. (2019) recipe |
| Pre-train / fine-tune epochs | 10 / 3 | Defaults (`--pretrain-epochs`, `--finetune-epochs`) |
| Training | per cross-section (quarter) | Embeddings are cross-sectional, re-estimated each quarter |

Two helper SLURM scripts target a GPU cluster: [01_smoke_test_os.sh](01_smoke_test_os.sh) (CUDA
sanity check) and [02_train_os.sh](02_train_os.sh) (full sweep, or `--first-only` dress rehearsal).
A `--test` flag runs a 2-quarter subset through `data_wutis/test/` → `embeddings_os/test/`.

### 3.4 Embedding diagnostics

[explore_embeddings.R](explore_embeddings.R) sanity-checks a quarter's embeddings: spherical
k-means clustering, a cosine-silhouette score vs. a random-placebo baseline, a 2D projection
(UMAP / t-SNE / PCA fallback), and nearest-neighbour "similar firms" tours. Exhibits land in
`exhibits/` (`clusters_*.png`, `clusters_*.parquet`).

---

## 4. The trading strategy — embedding-clustered daily reversion

Implemented in [healthcare_statarb.R](healthcare_statarb.R). Each trading window (entry =
quarter-end + 45 days, to respect the 13F reporting lag; held until the next entry) uses the
**prior quarter's** OS-BERT embeddings to cluster stocks. Each day, within each cluster, it longs
the underperformers and shorts the outperformers on a vol-adjusted residual (return minus cluster
mean), betting on next-day reversion. **Positions are restricted to GICS Health Care**; clusters
can be formed within Health Care only (`CLUSTER_SCOPE = "healthcare"`) or on the broad universe.

Key design choices (all parameters at the top of the script):

- **Clustering:** spherical k-means (`skmeans`, with a Euclidean `kmeans` fallback) on
  L2-normalised embeddings, `N_CLUSTERS = 8` (kept close to the count of HC GICS industries for a
  fair benchmark).
- **Signal:** within each cluster-day, z-score the residual by trailing 60-day vol, sort into
  quintiles, long the bottom / short the top, dollar-neutral with gross leverage 1.
- **Costs:** turnover-based, `COST_BPS = 5` per unit turnover; daily rebalancing is intentionally
  high-turnover, so the **net** row is the honest number and `REBALANCE_DAYS` is the main cost lever.
- **Benchmark:** the *same* engine re-run with **GICS industry groups** in place of embedding
  clusters (`RUN_GICS_BENCHMARK = TRUE`). The headline claim is **embedding net Sharpe > GICS net
  Sharpe**.

Inputs are the OS-BERT embeddings plus three WRDS pulls; outputs are written to `results/`
(`statarb_daily.parquet`, `statarb_summary.csv`).

### 4.1 Supporting WRDS data

| Script | Pulls | Output | Used for |
|---|---|---|---|
| [wrds_pull_crsp.R](wrds_pull_crsp.R) | CRSP CIZ daily stock file (`crsp_a_stock.dsf_v2`), total returns | `prices_crsp2.parquet` | daily returns for the backtest |
| [wrds_pull_finratio.R](wrds_pull_finratio.R) | WRDS Financial Ratios firm-level (`wrdsapps_finratio.firm_ratio`) | `finratio.parquet` | CUSIP → gvkey bridge (and valuation ratios) |
| [wrds_pull_gics.R](wrds_pull_gics.R) | Compustat company header (`comp.company`) | `gics.parquet` | gvkey → GICS sector / industry, survivorship-free |

Identifiers are reconciled on the **8-char CUSIP** (US ISIN = `US` + 9-char CUSIP, so chars 3–10
are the 8-char CUSIP; CRSP/finratio CUSIPs take their first 8).

---

## 5. Pipeline

```
WRDS / FactSet Ownership (13F)
        │
        ▼
  WUTIS_Data.r              ── quarterly investor-token sequences (full US universe)
        │                       data_wutis/q_YYYY-MM-DD.parquet
        ▼
  OS_BERT_training.py        ── per-quarter OS-BERT asset embeddings
        │                       embeddings_os/q_YYYY-MM-DD.parquet
        │                       models_os/q_YYYY-MM-DD/
        ├───────────────► explore_embeddings.R  ── clustering / silhouette / maps → exhibits/
        ▼
  healthcare_statarb.R       ── embedding clusters → daily within-cluster residual reversion
        │                       (Health Care only) vs GICS benchmark
        │                       results/statarb_daily.parquet, results/statarb_summary.csv
        ▲
        │   prices_crsp2.parquet · finratio.parquet · gics.parquet
        └── wrds_pull_crsp.R · wrds_pull_finratio.R · wrds_pull_gics.R
```

---

## 6. Repository layout

| Path | What it is |
|---|---|
| [WUTIS_Data.r](WUTIS_Data.r) | 13F holdings → quarterly investor-token sequences (`data_wutis/`) |
| [OS_BERT_training.py](OS_BERT_training.py) | OS-BERT per-quarter training + embedding extraction (`embeddings_os/`, `models_os/`) |
| [explore_embeddings.R](explore_embeddings.R) | Embedding diagnostics, clustering, 2D maps (`exhibits/`) |
| [healthcare_statarb.R](healthcare_statarb.R) | Embedding-clustered daily reversion backtest vs GICS (`results/`) |
| [wrds_pull_crsp.R](wrds_pull_crsp.R) · [wrds_pull_finratio.R](wrds_pull_finratio.R) · [wrds_pull_gics.R](wrds_pull_gics.R) | Supporting WRDS pulls (prices, ratios, GICS) |
| [01_smoke_test_os.sh](01_smoke_test_os.sh) · [02_train_os.sh](02_train_os.sh) | SLURM scripts for GPU training |
| [requirements.txt](requirements.txt) | Python dependencies (PyTorch, transformers, …) |
| `data_wutis/`, `embeddings_os/` | Quarterly sequence and embedding parquet files (gitignored) |
| `*.parquet` | Large data artifacts (`prices_crsp2`, `finratio`, `gics`; gitignored) |

---

## 7. How to run

**1. Set WRDS credentials** (an `.Renviron`, gitignored, is the convention here):

```
WRDS_USER=your_user
WRDS_PASSWORD=your_password
```

**2. Pull data (R):**

```r
Rscript WUTIS_Data.r          # ownership sequences  -> data_wutis/
Rscript wrds_pull_crsp.R      # daily returns        -> prices_crsp2.parquet
Rscript wrds_pull_finratio.R  # CUSIP->gvkey bridge  -> finratio.parquet
Rscript wrds_pull_gics.R      # GICS sectors         -> gics.parquet
```

**3. Train OS-BERT (Python):**

```bash
pip install -r requirements.txt
python OS_BERT_training.py            # full sweep over data_wutis/
python OS_BERT_training.py --test     # 2-quarter smoke run
# on a SLURM cluster: sbatch 02_train_os.sh   (or --first-only)
```

**4. Inspect embeddings & run the strategy (R):**

```r
Rscript explore_embeddings.R   # diagnostics + cluster maps -> exhibits/
Rscript healthcare_statarb.R   # backtest + GICS benchmark  -> results/
```

---

## References

Gabaix, X., Koijen, R. S. J., Richmond, R. J., & Yogo, M. (2025). *Asset Embeddings.*
Working paper. — methodology (OS-BERT, holdings-based firm similarity) underlying this project.

Supporting: Devlin et al. (2019), *BERT*; Reimers & Gurevych (2019), *Sentence-BERT*;
Koijen & Yogo (2019), *A Demand System Approach to Asset Pricing*; Jensen, Kelly & Pedersen
(2023), firm characteristics and returns.

---

*Prepared by WUTIS — WU Trading & Investment Society.*
