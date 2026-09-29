# Machine Learning from Portfolio Holdings

### An asset-embedding approach to US Health Care equity selection

*WUTIS — WU Trading & Investment Society · Global Markets · Final Pitch, June 2026 · Investment horizon: 3 months*

We learn a vector representation ("embedding") of every US stock from **who owns it**, using the OS-BERT method of Gabaix, Koijen, Richmond & Yogo (2025). The embeddings define each firm's true, ownership-implied peer set. We then ask: *given its peers, how should this stock be valued?* Health Care stocks that trade far below that peer-implied valuation are bought, and those far above it are sold short. The result is a market-neutral 5/5 long/short book, rebalanced quarterly as new 13F filings arrive.

---

## 1. The idea

Institutional investors price firms using a far richer information set than a few accounting ratios and a GICS label. The Asset Embeddings paper shows that **portfolio holdings encode the information relevant for prices**, and that low-dimensional vectors learned from holdings capture firm similarity better than industry classifications.

Health Care is where this should matter most. A clinical-stage biotech, a dividend-paying pharma major, a med-tech compounder and a hospital operator all sit under "Health Care", yet they are held by very different investor clienteles. Ownership-based peers are sharper comparables. Sharper comparables give a cleaner relative-valuation signal.

## 2. Pipeline

```
 WRDS: FactSet 13F holdings           WRDS: Compustat, CRSP
          │                                    │
          ▼                                    ▼
 WUTIS_Data.r                         wrds_pull_fundamentals.R ─► fundamentals.parquet
   └► data_wutis/q_*.parquet          wrds_pull_crsp.R ─────────► prices_crsp2.parquet
          │                           wrds_pull_gics.R ─────────► gics.parquet
          ▼                           merge_fundamentals_crsp.R ► fundamentals_merged.parquet
 OS_BERT_training.py                               │
   └► embeddings_os/q_*.parquet                    │
          │                                        │
          └──────────────────┬─────────────────────┘
                             ▼
               RelativeValue_strategy.r ─► valuation_metric.parquet
                             │              ai_predictions.parquet
                             │              ridge_oos_results.parquet
            ┌────────────────┼──────────────────────┐
            ▼                ▼                      ▼
   Backtest_trading.r   generate_live_picks_today.r   Regression_visual.r
   (7Y backtest +       (current 5/5 basket →         (OOS R² over time)
    tear sheet)          picks_crsp.csv)
```

## 3. Method

### 3.1 Ownership "sentences" — [WUTIS_Data.r](Backend/data_pull/WUTIS_Data.r)

- **Source:** FactSet 13F holdings on WRDS, quarterly, **2005 Q1 → 2026 Q1**, covering the **whole US market** and not just Health Care. A biotech held mainly by specialist funds only looks distinctive against the full investor landscape.
- **Asset** = issuer entity (one representative ISIN each). **Token** = investor ID.
- **Cleaning:**
  - Drop investors whose largest position exceeds 70% of their book.
  - Then iteratively keep only investors holding ≥ 10 stocks and stocks held by ≥ 10 investors.
- **Sequences:** each stock's investors are ordered from largest to smallest ownership share. Lists longer than 62 are split into chunks.

### 3.2 OS-BERT embeddings — [OS_BERT_training.py](Backend/embeddings/OS_BERT_training.py)

A small BERT model is trained **from scratch each quarter**, with investors playing the role of words:

1. **Masked-language-model pre-training.** 15% of the investors in each sequence are hidden, and the model learns to predict them from the rest. To succeed it must learn which investors co-own similar firms.
2. **Sentence-transformer fine-tuning.** Each sequence is split into even- and odd-ranked owners, and a contrastive (InfoNCE) loss pulls the two halves of the same stock together. This makes embeddings directly comparable by cosine similarity.
3. **Extraction.** The embedding is the mean-pooled output over a stock's top-62 owners.

| Setting | Value |
| --- | --- |
| Embedding size | 64 |
| Layers / heads / FFN | 4 / 2 / 256 |
| Context window | 62 investors |
| Pre-training | 10 epochs, lr 5e-4, 90/10 issuer-level split |
| Fine-tuning | 3 epochs, lr 2e-4 |

Training ran on a SLURM GPU cluster ([02_train_os.sh](Backend/embeddings/02_train_os.sh); [01_smoke_test_os.sh](Backend/embeddings/01_smoke_test_os.sh) checks the GPU environment first).

### 3.3 Valuation residual and mispricing — [RelativeValue_strategy.r](Strategy/RelativeValue_strategy.r)

Following Section 4.1 of the paper, for each quarter *t*:

1. **Strip out what book equity explains.** A cross-sectional regression, with both sides winsorised at 1%:

   $$\log ME_{at} = \gamma_t \log BE_{at} + \alpha_t + p^{\perp}_{at}$$

   The residual $p^{\perp}_{at}$ is the part of a stock's valuation that book equity cannot explain.

2. **Predict $p^{\perp}$ from ownership peers.** A ridge regression on the L2-normalised embeddings, $p^{\perp}_{at} = \beta_t'x_{at} + \delta_t + \epsilon_{at}$. It uses **5-fold cross-fitting at the firm level**, so every stock's prediction comes from a model that never saw it.

4. **Signal:**

   $$\widehat{p^{\perp}_{at}}$$

   Negative means cheap relative to ownership peers. Positive means rich.

The script also computes the pooled out-of-sample R² of the ridge step. [Regression_visual.r](Strategy/Regression_visual.r) plots it over time.

Inputs:
- The universe is every US stock (US ISIN) that has an embedding in that quarter.
- Fundamentals come from Compustat quarterly (`fundq`) and are **lagged one quarter** to avoid look-ahead.
- Market equity comes from CRSP via the CCM link table.
- A liquidity screen keeps the largest names by market cap each quarter (`LIQ_PCT`).

### 3.4 Trading rule — [Backtest_trading.r](Strategy/Backtest_trading.r)

- **Universe:** Health Care only (GICS sector 35, from Compustat `comp.company`).
- **Portfolio:** equal-weight, dollar-neutral. **Long the 5 most negative mispricings, short the 5 most positive.**
- **Rebalancing:** quarterly, as new 13F filings regenerate the embeddings.
- **Evaluation window:** the last 7 years (2019–2026). The script prints total gain, annualised return and volatility, Sharpe, Sortino, max drawdown and quarterly win rate.

### 3.5 Current picks — [generate_live_picks_today.r](Strategy/generate_live_picks_today.r)

The same model is applied to the latest embedding quarter (2025-12-31):
- **Universe:** US Health Care, top 50% by CRSP market cap.
- **Output:** the 5 cheapest and 5 richest names are written to `picks_crsp.csv`.

## 4. Results presented in the pitch

Out-of-sample backtest, 2019–2026, **gross of costs**:

| Metric | Value |
| --- | --- |
| Total cumulative gain | 4.15× |
| Annualised return | 22.57% |
| Sortino ratio | 2.87 |
| Win rate (quarterly) | 69.0% |

Basket at the time of the pitch (embeddings as of 2025-12-31):

| Long (ownership-cheap) | Short (ownership-rich) |
| --- | --- |
| Fresenius Medical Care (FMS) | DaVita (DVA) |
| Smith & Nephew (SNN) | Brookdale Senior Living (BKD) |
| Community Health Systems (CYH) | Amneal Pharmaceuticals (AMRX) |
| Genmab (GMAB) | HCA Healthcare (HCA) |
| Azenta (AZTA) | McKesson (MCK) |

### Limitations

- **Returns are a proxy.** Backtest returns are quarter-over-quarter changes in market equity. That approximates price return, but it also picks up share issuance and buybacks and ignores dividends. It is not a CRSP total-return series.
- **No trading costs.** Transaction and borrow costs are not modelled. Short positions in small Health Care names can be expensive to borrow.
- **GICS labels are static.** Each firm's current Compustat GICS label is applied to all history. This is survivorship-free in coverage but not point-in-time.


## 5. Repository layout

```
Backend/
  data_pull/
    WUTIS_Data.r                 13F holdings → ownership sequences
    wrds_pull_fundamentals.R     Compustat book equity, earnings, market value
    wrds_pull_crsp.R             CRSP daily stock file (returns, market cap)
    wrds_pull_gics.R             GICS sector per gvkey
    merge_fundamentals_crsp.R    CCM link: Compustat ↔ CRSP market equity
  embeddings/
    OS_BERT_training.py          OS-BERT training + embedding extraction
    01_smoke_test_os.sh          SLURM GPU sanity check
    02_train_os.sh               SLURM training job
    requirements.txt             Python dependencies
Strategy/
  RelativeValue_strategy.r       p⊥, cross-fitted ridge, mispricing signal
  Backtest_trading.r             Health Care 5/5 long/short backtest
  generate_live_picks_today.r    Current basket from the latest embeddings
  Regression_visual.r            Out-of-sample R² chart
```

**No data is included.** Every data file is derived from licensed WRDS sources (FactSet, Compustat, CRSP) and is excluded via `.gitignore`: holdings sequences, embeddings, fundamentals, prices and predictions. You need your own WRDS access to reproduce the results.

## 6. How to run

All scripts read and write their data files by bare filename, so **run everything from the repository root**.

**1. WRDS credentials.** Put these in `.Renviron` in the repo root (git-ignored):

```
WRDS_USER=your_user
WRDS_PASSWORD=your_password
```

**2. Pull data (R).** Required packages: `tidyverse`, `arrow`, `lubridate`, `dbplyr`, `RPostgres`, `glmnet`, `scales`.

```bash
Rscript Backend/data_pull/WUTIS_Data.r                # → data_wutis/
Rscript Backend/data_pull/wrds_pull_fundamentals.R    # → fundamentals.parquet
Rscript Backend/data_pull/wrds_pull_crsp.R            # → prices_crsp2.parquet (large)
Rscript Backend/data_pull/wrds_pull_gics.R            # → gics.parquet
Rscript Backend/data_pull/merge_fundamentals_crsp.R   # → fundamentals_merged.parquet
```

**3. Train embeddings (Python, GPU recommended).**

```bash
pip install -r Backend/embeddings/requirements.txt
python Backend/embeddings/OS_BERT_training.py          # all quarters → embeddings_os/
python Backend/embeddings/OS_BERT_training.py --test   # quick trial on data_wutis/test/
```

**4. Build the signal, backtest and picks (R).**

```bash
Rscript Strategy/RelativeValue_strategy.r       # → ai_predictions.parquet
Rscript Strategy/Backtest_trading.r             # tear sheet + equity curve
Rscript Strategy/generate_live_picks_today.r    # → picks_crsp.csv
Rscript Strategy/Regression_visual.r            # OOS R² chart
```

## Team

Elias Söser (Team Lead) · Isabelle Afkhampour · Jacopo Mei · Florian Wimmer · Viktoriia Yasinska

## References

- Gabaix, X., Koijen, R. S. J., Richmond, R. J., & Yogo, M. (2025). *Asset Embeddings.* Working paper.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.*
- Koijen, R. S. J., & Yogo, M. (2019). *A Demand System Approach to Asset Pricing.*

---

*For educational purposes only. Nothing in this repository is investment advice. Past performance, including backtested performance, is not indicative of future results.*
