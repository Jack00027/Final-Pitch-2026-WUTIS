# =============================================================================
# live_paper_trade.R  —  run the HC reversion book on fresh prices
# =============================================================================

library(dplyr)
library(readr)
library(arrow)
library(tidyquant)
library(skmeans)

# ── Inputs and settings ──────────────────────────────────────────────────────
EMB_DIR          <- "embeddings_os"        # folder of OS-BERT q_*.parquet; uses the latest
PRICES_PARQUET   <- "prices_crsp2.parquet" # only for the cusip -> ticker map + cap ranking
FINRATIO_PARQUET <- "finratio.parquet"
GICS_PARQUET     <- "gics.parquet"
TICKER_MAP_CSV   <- "ticker_map.csv"       # optional, from wrds_pull_tickers.R

FRI   <- as.Date("2026-06-12")   # signal date: form the book at this close
MON   <- as.Date("2026-06-15")   # held day:    earn this return
TODAY <- as.Date("2026-06-16")   # today:       form the next book at Monday's close

N_CLUSTERS  <- 8      # embedding clusters
N_QUANTILES <- 20      # quintiles: long the bottom, short the top
VOL_WINDOW  <- 60     # trailing days for vol
LIQ_PCT     <- 0.8   # keep the top this fraction of HC names by market cap
SEED        <- 42

# Standardise every id to the 8-char CUSIP (issuer + issue, no check digit).
ck_from_isin  <- function(x) toupper(substr(x, 3, 10))  # US ISIN = "US" + 9-char CUSIP
ck_from_cusip <- function(x) toupper(substr(x, 1, 8))   # first 8 of a 9-char CUSIP

# ── 1. Latest embeddings quarter ─────────────────────────────────────────────
latest <- "embeddings_os/q_2025-12-31.parquet"                            
message("using embeddings: ", basename(latest))
emb      <- read_parquet(latest) |> filter(!is.na(isin)) |> mutate(ck = ck_from_isin(isin))
dim_cols <- grep("^dim_", names(emb), value = TRUE)

# ── 2. Health Care names + company names (Compustat GICS, sector 35) ─────────
fr <- read_parquet(FINRATIO_PARQUET) |>
  transmute(ck = ck_from_cusip(cusip), gvkey = as.character(gvkey)) |>
  filter(ck != "", gvkey != "") |>
  distinct(ck, .keep_all = TRUE)
gics <- read_parquet(GICS_PARQUET) |>
  transmute(gvkey = as.character(gvkey), gsector = as.character(gsector), conm)
hc <- fr |>
  inner_join(gics, by = "gvkey") |>
  filter(gsector == "35") |>                            # 35 = Health Care
  select(ck, conm)

# ── 3. Latest market cap + ticker per name (from your CRSP extract) ──────────
latest_cap <- read_parquet(PRICES_PARQUET) |>
  transmute(ck = ck_from_cusip(cusip), date = as.Date(dlycaldt),
            cap = as.numeric(dlycap), ticker = toupper(trimws(ticker))) |>
  filter(!is.na(cap), cap > 0) |>
  group_by(ck) |>
  slice_max(date, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(ck, cap, ticker)

# ── 4. Tradeable universe: HC names with an embedding, top LIQ_PCT by cap ────
universe <- emb |>
  inner_join(hc, by = "ck") |>
  inner_join(latest_cap, by = "ck") |>
  filter(cap >= quantile(cap, 1 - LIQ_PCT, na.rm = TRUE)) |>
  distinct(ck, .keep_all = TRUE)

if (file.exists(TICKER_MAP_CSV)) {                      # optional cleaner ticker map
  map <- read_csv(TICKER_MAP_CSV, show_col_types = FALSE) |>
    transmute(ck = ck_from_cusip(ck), ticker = toupper(trimws(ticker)))
  universe <- universe |> select(-ticker) |> left_join(map, by = "ck")
}
universe <- universe |> filter(!is.na(ticker), ticker != "")

# ── 5. Cluster the universe (spherical k-means on L2-normalized embeddings) ──
X <- as.matrix(universe[, dim_cols])
X <- X / sqrt(rowSums(X^2))                             # row L2 norm (rotation-invariant)
k <- max(2L, min(N_CLUSTERS, nrow(universe) %/% (2 * N_QUANTILES)))
set.seed(SEED)
universe$cluster <- skmeans(X, k)$cluster
message(sprintf("universe: %d liquid HC names in %d clusters", nrow(universe), k))

# ── 6. Live daily prices from Yahoo ──────────────────────────────────────────
message(sprintf("pulling Yahoo prices for %d tickers ...", nrow(universe)))
prices <- tq_get(universe$ticker, from = FRI - 160, to = TODAY + 1) |>
  transmute(ticker = symbol, date = as.Date(date), adjusted) |>
  filter(!is.na(adjusted), adjusted > 0) |>
  arrange(ticker, date) |>
  group_by(ticker) |>
  mutate(ret = adjusted / lag(adjusted) - 1) |>
  ungroup() |>
  inner_join(select(universe, ck, ticker, conm, cluster), by = "ticker")

# ── The book: long laggards / short leaders within each cluster ──────────────
book_for <- function(prices, signal_date) {

  # each stock's signal return on the date
  signal <- prices |>
    filter(date == signal_date) |>
    select(ck, ticker, conm, cluster, day_ret = ret)

  # trailing vol: SD of the 60 daily returns strictly before the date
  vol <- prices |>
    filter(date < signal_date, !is.na(ret)) |>
    group_by(ck) |>
    slice_tail(n = VOL_WINDOW) |>
    summarise(vol = sd(ret), have = n(), .groups = "drop") |>
    filter(have == VOL_WINDOW)

  signal |>
    inner_join(vol, by = "ck") |>
    filter(!is.na(day_ret), vol > 0) |>
    group_by(cluster) |>
    filter(n() >= 2 * N_QUANTILES) |>                   # enough names for quintiles
    mutate(z    = (day_ret - mean(day_ret)) / vol,      # residual vs cluster, vol-adjusted
           rank = ntile(z, N_QUANTILES)) |>
    filter(rank == 1 | rank == N_QUANTILES) |>          # keep only the two extreme quintiles
    mutate(side    = if_else(rank == 1, 1, -1),         # laggards long, leaders short
           n_long  = sum(side == 1),
           n_short = sum(side == -1),
           w_raw   = if_else(side == 1, 1 / n_long, -1 / n_short)) |>  # equal-weight per leg, per cluster
    ungroup() |>
    mutate(weight = w_raw / sum(abs(w_raw))) |>         # scale so gross exposure = 1
    arrange(desc(side), cluster, z) |>
    select(ck, ticker, conm, cluster, day_ret, z, side, weight)
}

show_book <- function(book) {
  if (nrow(book) == 0) { cat("  (no positions: clusters too thin)\n"); return(invisible()) }
  book |>
    transmute(leg = if_else(side == 1, "LONG", "SHORT"),
              cluster, ticker,
              company   = substr(conm, 1, 30),
              `day ret` = sprintf("%+.2f%%", 100 * day_ret),
              z         = sprintf("%+.2f", z),
              weight    = sprintf("%+.3f", weight)) |>
    as.data.frame() |> print(row.names = FALSE)
  cat(sprintf("  %d long / %d short | gross %.2f | net %+.3f\n",
              sum(book$side == 1), sum(book$side == -1),
              sum(abs(book$weight)), sum(book$weight)))
}

# ── 7. Form the books and report ─────────────────────────────────────────────
book_mon <- book_for(prices, FRI)
book_tue <- book_for(prices, MON)

cat(sprintf("\n=== MONDAY BOOK  (formed %s close, traded %s) ===\n", FRI, MON))
show_book(book_mon)

monday_ret <- prices |> filter(date == MON) |> select(ck, monday_ret = ret)
pnl <- book_mon |> inner_join(monday_ret, by = "ck") |> mutate(contribution = weight * monday_ret)

cat("\n--- Realized Monday P&L (gross, 1 day) ---\n")
cat(sprintf("  gross %+.3f%%  |  long leg %+.3f%%  |  short leg %+.3f%%\n",
            100 * sum(pnl$contribution),
            100 * sum(pnl$contribution[pnl$side == 1]),
            100 * sum(pnl$contribution[pnl$side == -1])))
cat(sprintf("  hit rate %.0f%% of %d positions\n",
            100 * mean(pnl$contribution > 0), nrow(pnl)))

cat(sprintf("\n=== TUESDAY BOOK (formed %s close, to trade %s) ===\n", MON, TODAY))
show_book(book_tue)

# ── 8. Save ──────────────────────────────────────────────────────────────────
dir.create("results_live", showWarnings = FALSE)
write_csv(book_mon, "results_live/book_monday.csv")
write_csv(book_tue, "results_live/book_tuesday.csv")
write_csv(pnl,      "results_live/pnl_monday.csv")
cat("\nsaved -> results_live/book_monday.csv, book_tuesday.csv, pnl_monday.csv\n")
