# Healthcare Relative Value via Asset Embeddings

### A WUTIS — WU Trading & Investment Society — Investment Pitch

> Build holdings-based **asset embeddings** for US Health Care firms with the **OS-BERT**
> methodology of Gabaix, Koijen, Richmond & Yogo (2025), then trade a **sector-neutral
> long–short** book that is long firms trading *cheap* and short firms trading *rich*
> relative to the value implied by their embedding peers.

---

## 1. The pitch in one paragraph

Sophisticated investors price firms using a far richer information set than the handful of
accounting ratios and GICS labels analysts typically rely on. The Asset Embeddings paper
shows — theoretically and empirically — that **portfolio holdings encode all information
relevant for prices**, and that low-dimensional vectors learned from holdings (asset
embeddings) explain cross-sectional valuations dramatically better than firm characteristics.
We apply this to US Health Care, a sector where crude industry classification is especially
misleading: a clinical-stage biotech, a dividend-paying pharma major, a med-tech compounder,
and a managed-care payer all sit under "Health Care," yet they are held by entirely different
investor clienteles. We learn each firm's embedding from *who owns it*, regress valuations on
those embeddings to obtain an **embedding-implied fair value**, and go long the most
undervalued names against short positions in the most overvalued. The deviation from fair
value is our alpha signal; mean reversion toward the peer-implied value is the trade.

---

## 2. Investment thesis

1. **Holdings reveal true comparables.** Two firms are economically similar if the same kinds
   of investors choose to hold them — not because they share a SIC code. Asset embeddings
   recover this similarity directly from ownership.
2. **Embeddings define a high-quality valuation anchor.** Holdings-based embeddings explain
   well over 50% of the cross-sectional variation in relative valuations, versus roughly 15%
   for a standard set of firm characteristics, with explanatory power that keeps rising as the
   embedding dimension grows. The fitted value from regressing valuation on embeddings is a
   far better "fair value" than a peer-multiple or factor model.
3. **Deviations mean-revert.** A firm priced richly relative to its embedding peers is a
   candidate short; a firm priced cheaply is a candidate long. Trading the cross-sectional
   spread isolates relative mispricing while netting out sector-wide moves.
4. **Health Care is the right proving ground.** High within-sector heterogeneity (biotech vs.
   pharma vs. devices vs. services vs. payers vs. life-science tools) means embedding-based
   similarity should add the most value precisely where industry buckets fail.

---

## 3. Methodology — OS-BERT asset embeddings

We follow the **Ownership-Shares BERT (OS-BERT)** construction from the paper, the
asset-side dual of PS-BERT. The core idea: treat each **asset as a "sentence"** and each
**investor as a "token."**

### 3.1 Data — WRDS / FactSet Ownership

Quarterly institutional holdings from the FactSet Ownership Data on WRDS, following the
paper's Appendix D construction:

- **13F holdings** (`wrds_own_13f`, hedge funds) aggregated to the rollup entity level, with
  hedge funds identified via the entity sub-type in `own_ent_institutions`.
- **Fund holdings** (`wrds_own_fund` + `own_ent_funds`) for mutual funds, ETFs, closed-end
  funds, and variable annuity funds, keeping the last report within ±5 days of quarter-end.
- Merged to CRSP/Compustat (Jensen et al., 2023) by historical CUSIP → `permco` for book
  equity, market equity, and the sector classification.
- **Universe:** GICS **Health Care (sector 35)**, US-listed. Micro/nano caps removed; we keep
  firms held by ≥ 20 investors and investors holding ≥ 20 stocks, iterating to convergence,
  and drop investors with a single position above 75% of the portfolio.

**Universe design note.** OS-BERT is trained on the full US cross-section so that co-ownership
with both generalist and specialist investors is visible (a healthcare name held mostly by
dedicated biotech funds is very different from one held by broad large-cap value funds — that
contrast is signal). We then *extract* the embeddings for the Health Care sub-universe and run
the relative-value step within it. Training on Health Care alone is a viable alternative but
leaves a thin per-quarter corpus (~200 names), so the full-market-then-extract route is the
baseline.

### 3.2 Ownership sequences

For each (asset, quarter), order the firm's investors by **descending ownership share** and
represent each investor as a token:

```
Vanguard Total Stock Market, BlackRock Health Sciences, ARK Genomic Revolution, ...
```

Sequences longer than the 62-token context window are split into equal chunks.

### 3.3 OS-BERT training (per quarter, two stages)

**Stage 1 — masked-investor pre-training.** A small bidirectional encoder (4 attention layers,
2 heads, context window 62) is trained with masked-token prediction: mask 15% of investors in
each asset's owner list (80% `[MASK]`, 10% random investor, 10% unchanged) and predict them
from the co-owners via cross-entropy. The attention mechanism yields a *contextualized*
embedding for each owner, conditioned on the asset's other holders.

**Stage 2 — sentence-transformer fine-tuning.** Split each asset's owner list into even-rank
and odd-rank halves to form positive pairs; negatives are other assets in the batch. A
symmetric InfoNCE / cosine objective pulls the two halves of the same asset together and
pushes different assets apart. This guarantees the pooled asset vectors are comparable by
**cosine similarity**.

**Embedding extraction.** Mean-pool the contextualized owner vectors over the firm's **top-62
owners** to obtain one embedding per asset. Two firms are close in this space when they are
held by similar investor clienteles.

| Setting | Value | Rationale |
|---|---|---|
| Embedding dimension | 64–128 | OS-BERT's relative-valuation power keeps rising with dimension |
| Attention layers / heads | 4 / 2 | Paper baseline |
| Context window | 62 owners | Paper baseline |
| Masking | 15% (80/10/10) | Devlin et al. (2019) recipe |
| Training | per cross-section (quarter) | Embeddings are cross-sectional, re-estimated each quarter |

---

## 4. From embeddings to the portfolio

### 4.1 The relative-value signal

**Step 1 — valuation measure.** In each quarter, within Health Care, regress log market equity
on log book equity:

$$ p_{a} = \gamma\, b_{a} + \alpha + p_{a}^{\perp} $$

The residual $p_a^{\perp}$ is our valuation measure — a generalization of market-to-book that
strips out the mechanical book-equity component.

**Step 2 — embedding-implied fair value.** Regress the valuation residual on the OS-BERT asset
embeddings (standardized, ridge penalty by 10-fold cross-validation):

$$ p_{a}^{\perp} = \beta' x_{a} + \delta + \varepsilon_{a} $$

The fitted value $\hat p_a^{\perp} = \hat\beta' x_a + \hat\delta$ is what the firm's valuation
*should* be, given how investors value its embedding peers.

**Step 3 — mispricing.** The signal is the regression residual:

$$ \text{signal}_a = p_a^{\perp} - \hat p_a^{\perp} $$

- $\text{signal}_a < 0$ → priced **cheap** relative to peers → **LONG**
- $\text{signal}_a > 0$ → priced **rich** relative to peers → **SHORT**

### 4.2 Portfolio construction

- Rank Health Care firms by the mispricing signal each quarter.
- **Long** the cheapest tercile/decile, **short** the richest, **dollar-neutral** (equal long
  and short notional).
- Because the universe is a single sector, the book is naturally a within-sector relative-value
  trade; optionally neutralize residual beta and sub-industry tilts.
- **Rebalance quarterly**, in step with holdings disclosure, with an implementation lag that
  respects 13F/fund reporting delays (see risks).
- Weighting: equal-weight as the baseline, signal-weighted as a variant.

### 4.3 Optional enhancement — supervised fine-tuning

The paper notes OS-BERT may gain from fine-tuning toward the valuation target. As an
extension we can fine-tune the embeddings (or the ridge stage) directly on the relative-value
objective rather than using the purely unsupervised embeddings.

---

## 5. Pipeline

```
WRDS / FactSet Ownership (13F + Fund)
        │
        ▼
  data_cleaning.R            ── quarterly ownership sequences (Health Care universe)
        │                       data/q_YYYY-MM-DD.parquet
        ▼
  os_bert_training.py        ── per-quarter OS-BERT asset embeddings
        │                       embeddings/q_YYYY-MM-DD.parquet
        │                       models/q_YYYY-MM-DD/
        ▼
  relative_value.py          ── valuation residual → ridge fair value → mispricing signal
        │                       signals/q_YYYY-MM-DD.parquet
        ▼
  backtest.py                ── long–short construction, rebalancing, P&L, risk
        │                       results/  (returns, Sharpe, drawdowns, exhibits)
        ▼
  pitch/                     ── slides, charts, and exhibits for the WUTIS presentation
```

## 6. Repository layout

```
.
├── data_cleaning.R          # WRDS FactSet Ownership → quarterly HC ownership sequences
├── os_bert_training.py      # OS-BERT: per-quarter asset embeddings (asset = sentence)
├── relative_value.py        # valuation residual + ridge fair value + mispricing signal
├── backtest.py              # long–short portfolio, rebalancing, performance & risk
├── pitch/                   # WUTIS deck and exhibits
├── data/                    # ownership sequences (one parquet per quarter)
├── embeddings/              # asset embeddings (one parquet per quarter)
├── models/                  # per-quarter OS-BERT checkpoints
├── signals/                 # per-quarter mispricing signals
├── results/                 # backtest output
├── .Renviron                # WRDS_USER / WRDS_PASSWORD (gitignored)
└── README.md
```

## 7. Key parameters

| Parameter | Value | Where |
|---|---|---|
| Sector universe | GICS Health Care (35), US | `data_cleaning.R` |
| Sample period | 2005-Q1 … latest | `data_cleaning.R` |
| Min investors per stock | 20 | `data_cleaning.R` |
| Min stocks per investor | 20 | `data_cleaning.R` |
| Max single-holding weight | 75% | `data_cleaning.R` |
| Embedding dimension | 64–128 | `os_bert_training.py` |
| OS-BERT layers / heads / context | 4 / 2 / 62 | `os_bert_training.py` |
| Masking rate | 15% (80/10/10) | `os_bert_training.py` |
| Valuation measure | residual of log ME on log BE | `relative_value.py` |
| Fair-value model | ridge on embeddings, 10-fold CV | `relative_value.py` |
| Rebalance frequency | quarterly | `backtest.py` |
| Book structure | dollar-neutral long–short | `backtest.py` |

## 8. Reproducing

**Requirements**

- **R:** `tidyverse`, `lubridate`, `dbplyr`, `RPostgres`, `arrow`
- **Python ≥ 3.10:** `torch`, `transformers`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`
- WRDS credentials in `.Renviron`:
  ```
  WRDS_USER=...
  WRDS_PASSWORD=...
  ```

**Run**

```bash
# 1. Build quarterly Health Care ownership sequences (pulls from WRDS)
Rscript data_cleaning.R

# 2. Train OS-BERT and extract asset embeddings (one model per quarter)
python os_bert_training.py

# 3. Build the relative-value mispricing signal
python relative_value.py

# 4. Backtest the long–short book
python backtest.py
```

## 9. Risks & caveats

- **Reporting lag / look-ahead.** 13F filings arrive up to 45 days after quarter-end and funds
  report on varied schedules; the backtest must lag entry to when holdings were actually
  observable. No firm enters the signal before its ownership data is public.
- **Cross-sectional, not time-series.** Per-quarter embeddings are identified only up to
  rotation, so the signal compares firms *within* a quarter — it is not a level that can be
  tracked across quarters without alignment.
- **Value traps.** A firm priced below its embedding peers may be cheap for a reason the
  embedding does not capture; the signal is a relative-value prior, not a guarantee.
- **Event risk in Health Care.** Binary catalysts (trial readouts, FDA decisions, patent
  cliffs, M&A) can swamp relative-value signals, especially in clinical-stage biotech;
  position sizing and possibly excluding pre-revenue names should be considered.
- **Shorting frictions.** Borrow availability and cost for smaller-cap healthcare names can
  erode the short leg.
- **Ownership crowding.** Because the signal is built from holdings, crowded names can unwind
  together; monitor concentration on both legs.

## 10. References

Gabaix, X., Koijen, R. S. J., Richmond, R. J., & Yogo, M. (2025). *Asset Embeddings.*
Working paper. — methodology (OS-BERT, relative valuation benchmark) underlying this project.

Supporting: Devlin et al. (2019), *BERT*; Reimers & Gurevych (2019), *Sentence-BERT*;
Koijen & Yogo (2019), *A Demand System Approach to Asset Pricing*; Jensen, Kelly & Pedersen
(2023), firm characteristics and returns.

---

*Prepared by WUTIS — WU Trading & Investment Society.*
