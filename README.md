# Healthcare Relative Value via Asset Embeddings

### A WUTIS — WU Trading & Investment Society — Investment Pitch

> Build holdings-based **asset embeddings** for US firms with the **OS-BERT** methodology of Gabaix, Koijen, Richmond & Yogo (2025), use them to define **peer groups** for US Health Care firms, and trade a **sector-neutral, embedding-clustered daily residual-reversion** book — long the day's underperformers and short the outperformers *within* each embedding peer group. The headline test: do embedding peer groups beat GICS industries for Health Care relative value?

---

## 1. The idea

Sophisticated investors price firms using a far richer information set than the handful of accounting ratios and GICS labels analysts typically rely on.

The Asset Embeddings paper shows — theoretically and empirically — that  **portfolio holdings encode all information relevant for prices** , and that low-dimensional vectors learned from holdings (asset embeddings) capture firm similarity far better than crude industry labels. We apply this to US Health Care, a sector where industry classification is especially misleading: a clinical-stage biotech, a dividend-paying pharma major, a med-tech compounder, and a managed-care payer all sit under "Health Care," yet they are held by entirely different investor clienteles. We learn each firm's embedding from  *who owns it* , **cluster firms into peer groups** by embedding similarity, and then run a short-horizon **mean-reversion** trade *within* each cluster: each day, long the names that lagged their peers and short the names that ran ahead, betting on next-day reversion of the cross-sectional residual.

---

## 2. The thesis

1. **Ownership reveals the real "comparables."** The same investors holding two stocks is
   a stronger signal of similarity than a similar firm characteristics.
2. **Better peer groups → cleaner trades.** When a group truly contains look-alike
   companies, a stock straying from the group is more likely a temporary mispricing than
   real news — exactly the situation mean reversion profits from.
3. **It nets out the noise.** Trading *within* a group (long the laggards, short the
   leaders) cancels out big market- and sector-wide moves, leaving just the relative
   mispricing we're after.
4. **Health Care is the perfect test case.** It's the sector where industry labels fail
   hardest, so smarter grouping should add the most value here.

---

## 3. OS-BERT asset embeddings

This is the technical heart of the project. We borrow a tool from language AI.

### The language analogy

Modern AI models (like BERT) understand a word by looking at the words around it — "bank"
means something different next to "river" than next to "money." We reuse this idea:

- A **stock** is like a **sentence**.
- Each **investor who owns it** is like a **word** in that sentence.

By training a small BERT-style model to "read" these ownership sentences, we get, for each
stock, a compact vector (an **embedding**) that captures its ownership profile.
Stocks with similar embeddings are held by similar investors — i.e. they're true peers.
We then group (cluster) stocks by these embeddings.

> The official name for this recipe is **OS-BERT** (Ownership-Shares BERT), from Gabaix,
> Koijen, Richmond & Yogo (2025).

### 3.1 The data — who owns what

Source: quarterly institutional holdings (13F filings) from **FactSet Ownership on WRDS**,
built by [WUTIS_Data.r](WUTIS_Data.r).

- We take every 13F holding near each quarter-end, across the **whole US market** (all
  sectors, not just Health Care — see the note below).
- The "asset" is the **company** (issuer), and we record, for each company each quarter,
  the list of investors who hold it, **ranked from biggest owner to smallest**.
- We clean out noise: keep companies held by **≥ 10 investors**, keep investors holding
  **≥ 10 stocks**, and drop investors who are dangerously concentrated (one position > 70%
  of their book). This repeats until the set is stable.
- Coverage: **2005 → 2026**, one snapshot per quarter.

The output is one "ownership sentence" per company per quarter, saved to
`data_wutis/q_YYYY-MM-DD.parquet`.

> **Why train on the whole market, not just Health Care?** It's more informative to see
> that a biotech is held mostly by specialist biotech funds *versus* broad index funds —
> and you only see that contrast against the full investor landscape. So we learn
> embeddings for *every* stock, and only **narrow down to Health Care when we trade**.

### 3.2 Turning ownership into "sentences"

For each company in a quarter, we line up its investors from largest to smallest stake:

```
investor_8214, investor_119, investor_4471, ...   (biggest owner → smallest)
```

A company can have hundreds of owners, but the model only reads up to **62** at a time, so
very long lists are split into chunks.

### 3.3 Training the model (done once per quarter)

Run by [OS_BERT_training.py](OS_BERT_training.py). Each investor ID is treated as a "word."
Training happens in two stages:

**Stage 1 — learn the language of ownership.** We hide 15% of the investors in each
company's list and train the model to guess them back from the others (the standard BERT
"fill in the blank" trick). To predict a missing owner well, the model has to learn which
investors tend to co-own the same kinds of companies.

**Stage 2 — make similar companies line up.** We split each company's owner list into two
halves and teach the model that the two halves of the *same* company should look alike,
while different companies should look different. This makes the final embeddings directly
comparable by similarity.

**The result:** one embedding (a list of 64 numbers) per company per quarter, saved to
`embeddings_os/q_*.parquet`. Similar numbers ⇒ similar owners ⇒ true peers.

| Setting               | Value             | In plain terms                                |
| --------------------- | ----------------- | --------------------------------------------- |
| Embedding size        | 64 numbers        | How detailed each company's "fingerprint" is  |
| Owners read per stock | 62                | The model's attention span                    |
| Model size            | 4 layers, 2 heads | A deliberately small, fast model              |
| Re-trained            | every quarter     | Ownership changes over time, so we refresh it |

### 3.4 Proving the peer groups beat GICS

[explore_embeddings.R](explore_embeddings.R) is the **visual and quantitative companion to
the backtest**. The trade in section 4 lives or dies on one claim — that ownership-based
peer groups are better than GICS industries for Health Care. This script shows *why* that's
true, on a single representative quarter, before any trading happens.

It deliberately mirrors the strategy: it clusters the embeddings **within Health Care** into
the **same 8 groups** the backtest uses (`K` matched to `N_CLUSTERS`), then puts those
embedding clusters head-to-head with the official **GICS Health Care sub-industries**
(Biotechnology, Pharmaceuticals, HC Equipment & Supplies, HC Providers & Services, HC
Technology, Life Sciences Tools).

What it produces:

- **A "how clean are the groups?" score** (cosine *silhouette*) for three groupings, all
  measured in the same embedding space: the **embedding clusters**, the **GICS industries**,
  and a **random placebo** as a sanity floor. Higher = tighter, more self-similar groups.
- **A similarity score between the two groupings** (the *adjusted Rand index*). A **low**
  value means the embeddings carve Health Care up **differently** from
  GICS: they find structure the industry labels miss.
- **A cluster × GICS composition table** (`cluster_gics_composition_*.csv`) showing what mix
  of GICS industries sits inside each embedding cluster — i.e. exactly where clusters merge
  or cut across the standard buckets.
- **Two side-by-side 2D maps** of the same Health Care firms (UMAP, with t-SNE / PCA
  fallbacks): one coloured by **embedding cluster** (`hc_clusters_embedding_*.png`), one by
  **GICS industry** (`hc_clusters_gics_*.png`). Seeing the two colourings *not* line up is
  the argument, made visually.
- **A nearest-neighbour tour**: for the most representative firm in each cluster, its closest
  look-alikes by ownership, each tagged with its GICS industry. The headline artifact is that
  a firm's true peers often **span several GICS industries** — exactly the cross-industry
  grouping GICS can't produce.

Everything lands in `exhibits/`: the two PNG maps, `hc_clusters_*.parquet` (cluster
assignments + map coordinates per firm), and the composition CSV. Point `EMB_FILE` at the
same quarter the strategy trades; it needs `finratio.parquet` and `gics.parquet` present (so
run the WRDS pulls in section 4 first).

---

## 4. The trade — buy the laggards, short the leaders

Run by [healthcare_statarb.R](healthcare_statarb.R). Putting it all together:

1. **Group.** Use last quarter's embeddings to sort stocks into **8 peer clusters**
   (we wait 45 days after quarter-end, because 13F filings are public with a lag).
2. **Restrict to Health Care.** We only ever hold Health Care names.
3. **Score, every day.** Within each cluster, measure how far each stock's return sits
   from its cluster's average.
4. **Trade.** Buy the biggest laggards, short the biggest leaders, in equal dollar amounts
   (so the book is market-neutral). Bet on reversion the next day.
5. **Be honest about costs.** Trading daily racks up turnover, so we subtract realistic
   trading costs (5 bps per unit traded). The **after-cost** number is the one that counts.

**The crucial comparison:** we then run the *identical* strategy but group stocks by
**GICS industry** instead of embeddings. The pitch succeeds if the embedding version earns
a higher after-cost Sharpe ratio than the GICS version.

Results are written to `results/` (`statarb_daily.parquet`, `statarb_summary.csv`) and the
script prints a side-by-side Embedding-vs-GICS scorecard.

### Supporting market data

The backtest also needs prices and a way to identify each company. Three small WRDS pulls
provide that:

| Script                                    | What it grabs                    | Why we need it                    |
| ----------------------------------------- | -------------------------------- | --------------------------------- |
| [wrds_pull_crsp.R](wrds_pull_crsp.R)         | Daily stock returns (CRSP)       | The actual returns we trade on    |
| [wrds_pull_finratio.R](wrds_pull_finratio.R) | A CUSIP → company-key bridge    | Links prices to the right company |
| [wrds_pull_gics.R](wrds_pull_gics.R)         | GICS industry labels (Compustat) | The benchmark we test against     |

(All three are matched together on the 8-character CUSIP, the common ID across sources.)

---

## 5. The whole pipeline at a glance

```
WRDS / FactSet 13F holdings
        │
        ▼
  WUTIS_Data.r          ──►  "ownership sentences"     data_wutis/q_*.parquet
        │
        ▼
  OS_BERT_training.py   ──►  company embeddings        embeddings_os/q_*.parquet
        │
        ├──►  explore_embeddings.R   ──►  cluster maps & checks   exhibits/
        │
        ▼
  healthcare_statarb.R  ──►  the backtest + GICS benchmark        results/
        ▲
        │   needs:  prices_crsp2.parquet · finratio.parquet · gics.parquet
        └── from:   wrds_pull_crsp.R · wrds_pull_finratio.R · wrds_pull_gics.R
```

---

## 6. What's in this folder

| File                                                                                                            | What it does                                                                 |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [WUTIS_Data.r](WUTIS_Data.r)                                                                                       | Builds ownership sentences from 13F holdings →`data_wutis/`               |
| [OS_BERT_training.py](OS_BERT_training.py)                                                                         | Trains OS-BERT, produces embeddings →`embeddings_os/`, `models_os/`     |
| [explore_embeddings.R](explore_embeddings.R)                                                                       | Validates HC peer groups vs GICS — maps, scores, neighbours →`exhibits/` |
| [healthcare_statarb.R](healthcare_statarb.R)                                                                       | The trading strategy + GICS comparison →`results/`                        |
| [wrds_pull_crsp.R](wrds_pull_crsp.R), [wrds_pull_finratio.R](wrds_pull_finratio.R), [wrds_pull_gics.R](wrds_pull_gics.R) | Pull prices, IDs, and industry labels                                        |
| [01_smoke_test_os.sh](01_smoke_test_os.sh), [02_train_os.sh](02_train_os.sh)                                          | Run the training on a GPU cluster                                            |
| [requirements.txt](requirements.txt)                                                                               | Python dependencies                                                          |
| `data_wutis/`, `embeddings_os/`, `*.parquet`                                                              | Data files (large, kept out of git)                                          |

---

## 7. How to run it yourself

**1. Add your WRDS login** to a file called `.Renviron` (kept private, not committed):

```
WRDS_USER=your_user
WRDS_PASSWORD=your_password
```

**2. Pull the data** (R):

```r
Rscript WUTIS_Data.r          # ownership sentences  -> data_wutis/
Rscript wrds_pull_crsp.R      # daily returns        -> prices_crsp2.parquet
Rscript wrds_pull_finratio.R  # ID bridge            -> finratio.parquet
Rscript wrds_pull_gics.R      # industry labels      -> gics.parquet
```

**3. Train the embeddings** (Python):

```bash
pip install -r requirements.txt
python OS_BERT_training.py            # all quarters
python OS_BERT_training.py --test     # quick 2-quarter trial
# on a GPU cluster:  sbatch 02_train_os.sh
```

**4. Explore and trade** (R):

```r
Rscript explore_embeddings.R   # cluster maps & sanity checks -> exhibits/
Rscript healthcare_statarb.R   # the backtest + GICS benchmark -> results/
```

---

## References

Gabaix, X., Koijen, R. S. J., Richmond, R. J., & Yogo, M. (2025). *Asset Embeddings.*
Working paper — the source of the OS-BERT method.

Supporting: Devlin et al. (2019), *BERT*; Reimers & Gurevych (2019), *Sentence-BERT*;
Koijen & Yogo (2019), *A Demand System Approach to Asset Pricing*; Jensen, Kelly & Pedersen
(2023), firm characteristics and returns.

---

*Prepared by WUTIS — WU Trading & Investment Society.*
