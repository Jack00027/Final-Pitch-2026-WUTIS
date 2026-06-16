# =============================================================================
# Embedding-clustered daily residual reversion — US Health Care
# =============================================================================

library(tidyverse)
library(arrow)
library(slider)    # rolling windows; install.packages("slider")
library(skmeans)   # spherical k-means; install.packages("skmeans")

# ── Parameters ───────────────────────────────────────────────────────────────
EMB_DIR          <- "embeddings_os"
PRICES_PARQUET   <- "prices_crsp2.parquet"
FINRATIO_PARQUET <- "finratio.parquet"
GICS_PARQUET     <- "gics.parquet"

TRADE_SECTOR_CODE <- "35"          # GICS sector to trade; 35 = Health Care
CLUSTER_SCOPE     <- "healthcare"  # "healthcare" = cluster WITHIN HC (self-contained)
                                   # "all"        = cluster the broad universe, trade only HC
N_CLUSTERS        <- 8             # embedding clusters (kept near the HC GICS industry count)

LAG_DAYS       <- 45    # 13F public ~45d after quarter-end -> point-in-time entry
VOL_WINDOW     <- 60    # trailing days for residual vol-adjustment
N_QUANTILES    <- 20    # long the bottom 1/N, short the top 1/N, within each cluster
REBALANCE_DAYS <- 1     # holding period in TRADING days (~21 = month, ~5 = week)
SIGNAL_WINDOW  <- 1     # residual lookback in TRADING days; keep ~= REBALANCE_DAYS for a
                        # coherent horizon (1/1 daily, 5/5 weekly). 1 = the daily strategy.

TRADING_COSTS  <- TRUE  # FALSE = gross (no costs)
COST_BPS       <- 5     # bps of traded notional per unit turnover (per rebalance)

RUN_GICS_BENCHMARK <- TRUE   # also run the GICS-grouped engine for comparison
GICS_LEVEL         <- "gind" # GICS grouping for the benchmark: "gind" or "ggroup"

LIQUIDITY_SCREEN <- FALSE    # keep only the most liquid names (cost model + shortability)
LIQ_PCT          <- 0.8      # fraction of the HC universe to keep, by market cap

SWEEP          <- FALSE              # TRUE = horizon x cost grid of NET Sharpe
SWEEP_HORIZONS <- c(1, 5, 21)        # signal = hold, in trading days
SWEEP_COSTS    <- c(5, 10, 15, 20)   # bps/side, to stress the cost assumption

SEED <- 42; set.seed(SEED)

# Standardise every id to the 8-char CUSIP (issuer + issue, no check digit).
ck_from_cusip <- function(x) toupper(substr(trimws(as.character(x)), 1, 8))  # first 8 of 9-char
ck_from_isin  <- function(x) toupper(substr(trimws(as.character(x)), 3, 10)) # US ISIN chars 3..10

# ── Small functions ──────────────────────────────────────────────────────────

# Cluster rows of an embedding matrix with spherical k-means (L2-normalize first).
assign_clusters <- function(M, k) {
  M <- M / sqrt(rowSums(M^2))                 # row L2 norm (rotation-invariant)
  k <- min(k, max(2L, nrow(M) - 1L))
  skmeans(M, k)$cluster
}

# Add each stock's trailing k-day compounded return as `cumret` (k=1 -> daily return).
add_cumret <- function(panel, window) {
  panel |>
    group_by(ck) |> arrange(date, .by_group = TRUE) |>
    mutate(cumret = slide_dbl(ret, function(r) prod(1 + r) - 1,
                              .before = window - 1, .complete = TRUE)) |>
    ungroup()
}

# Turnover at each rebalance = total absolute weight change vs the previous one,
# treating a name absent on either date as weight 0 (first rebalance = from cash).
turnover_by_date <- function(weights) {
  indexed <- weights |> mutate(r = match(date, sort(unique(date))))
  curr <- indexed |> select(r, ck, w)
  prev <- indexed |> transmute(r = r + 1, ck, w_prev = w)   # shift to align with next rebalance
  full_join(curr, prev, by = c("r", "ck")) |>
    mutate(w = coalesce(w, 0), w_prev = coalesce(w_prev, 0)) |>
    group_by(r) |>
    summarise(turnover = sum(abs(w - w_prev)), .groups = "drop") |>
    left_join(distinct(indexed, r, date), by = "r") |>
    arrange(date) |> select(date, turnover)
}

# Run the reversion engine for one grouping ("cluster" or "gics_group") and holding
# period. Returns the cost-free daily gross series + per-rebalance turnover.
run_engine <- function(panel, trading_days, group_col, hold) {
  rebal_days <- trading_days[seq(1, length(trading_days), by = hold)]

  # 1. target weights on each rebalance date: within each group, long the biggest
  #    laggards and short the biggest leaders on the vol-adjusted residual z.
  weights <- panel |>
    filter(date %in% rebal_days, !is.na(.data[[group_col]]), !is.na(cumret), vol > 0) |>
    group_by(date, across(all_of(group_col))) |>
    filter(n() >= 2 * N_QUANTILES) |>                         # balanced tails only
    mutate(z = (cumret - mean(cumret)) / vol,
           rank = ntile(z, N_QUANTILES)) |>
    filter((rank == 1 | rank == N_QUANTILES) & is_hc) |>      # extreme tails, HC only
    mutate(side    = if_else(rank == 1, 1, -1),               # laggards long, leaders short
           n_long  = sum(side == 1),
           n_short = sum(side == -1),
           w_raw   = if_else(side == 1, 1 / n_long, -1 / n_short)) |>  # equal weight per leg, per cluster
    group_by(date) |>
    mutate(w = w_raw / sum(abs(w_raw))) |>                    # gross = 1 per day
    ungroup() |>
    select(date, ck, w)
  if (nrow(weights) == 0) stop(sprintf("no positions for grouping '%s'", group_col))

  # 2. turnover per rebalance
  turnover <- turnover_by_date(weights)

  # 3. daily gross: each rebalance's weights earn returns until the next rebalance.
  #    A day is governed by the most recent rebalance STRICTLY before it.
  rebal_idx <- which(trading_days %in% rebal_days)
  gov       <- findInterval(seq_along(trading_days) - 1L, rebal_idx)
  day_gov   <- tibble(date = trading_days,
                      gov_date = if_else(gov >= 1, rebal_days[pmax(gov, 1)], as.Date(NA))) |>
    filter(!is.na(gov_date))
  returns <- panel |> distinct(date, ck, ret)
  gross <- day_gov |>
    inner_join(weights, by = c("gov_date" = "date"), relationship = "many-to-many") |>
    inner_join(returns, by = c("date", "ck")) |>
    group_by(date) |> summarise(gross = sum(w * ret), .groups = "drop") |> arrange(date)

  list(gross = gross, turnover = turnover, avg_turnover = mean(turnover$turnover, na.rm = TRUE))
}

# Apply a cost (bps per unit turnover, charged on rebalance dates) -> gross + net daily.
net_series <- function(engine, cost_bps, costs_on = TRUE) {
  cost <- engine$turnover |>
    mutate(cost = if (costs_on) (cost_bps / 1e4) * turnover else 0) |>
    select(date, cost)
  engine$gross |>
    left_join(cost, by = "date") |>
    mutate(cost = coalesce(cost, 0), net = gross - cost) |> arrange(date)
}

# Annualized performance stats from a daily return series.
daily_stats <- function(r) {
  r <- r[!is.na(r)]
  if (length(r) < 2) return(tibble(ann_ret = NA, ann_vol = NA, sharpe = NA, max_dd = NA, hit = NA))
  cum <- cumprod(1 + r)
  drawdown <- cum / cummax(cum) - 1
  tibble(ann_ret = prod(1 + r)^(252 / length(r)) - 1,
         ann_vol = sd(r) * sqrt(252),
         sharpe  = mean(r) / sd(r) * sqrt(252),
         max_dd  = min(drawdown),
         hit     = mean(r > 0))
}

# ── Load inputs ──────────────────────────────────────────────────────────────
dir.create("results", showWarnings = FALSE)
message("loading inputs ...")

# ck -> is_hc / GICS group / company name (via the finratio cusip->gvkey bridge)
bridge <- read_parquet(FINRATIO_PARQUET) |>
  transmute(ck = ck_from_cusip(cusip), gvkey = as.character(gvkey)) |>
  filter(ck != "", gvkey != "") |>
  distinct(ck, .keep_all = TRUE)
sectors <- read_parquet(GICS_PARQUET) |>
  transmute(gvkey = as.character(gvkey),
            is_hc = as.character(gsector) == TRADE_SECTOR_CODE,
            gics_group = as.character(.data[[GICS_LEVEL]]), conm) |>
  distinct(gvkey, .keep_all = TRUE)
lookup <- bridge |> left_join(sectors, by = "gvkey") |>
  transmute(ck, is_hc = coalesce(is_hc, FALSE), gics_group, conm)
hc_ck <- lookup |> filter(is_hc) |> distinct(ck)

# embeddings: one frame of (quarter, ck, dim_*)
emb_list <- list()
for (f in list.files(EMB_DIR, pattern = "^q_.*\\.parquet$", full.names = TRUE)) {
  q <- sub("^q_", "", tools::file_path_sans_ext(basename(f)))
  d <- read_parquet(f) |> filter(!is.na(isin))
  d$quarter <- q
  d$ck      <- ck_from_isin(d$isin)
  emb_list[[q]] <- d |> select(quarter, ck, starts_with("dim_"))
}
emb <- bind_rows(emb_list)
dim_cols <- grep("^dim_", names(emb), value = TRUE)

# prices: CRSP CIZ daily (assumes cusip + dlycaldt + dlyret + dlycap present)
prices <- read_parquet(PRICES_PARQUET) |>
  transmute(ck = ck_from_cusip(cusip), date = as.Date(dlycaldt),
            ret = as.numeric(dlyret), cap = as.numeric(dlycap)) |>
  filter(ck != "", !is.na(date), !is.na(ret)) |>
  arrange(ck, date) |> distinct(ck, date, .keep_all = TRUE)

# ── Trading windows: quarter q -> [as.Date(q) + LAG_DAYS, next entry) ─────────
quarters <- sort(unique(emb$quarter))
windows <- tibble(quarter = quarters, entry = as.Date(quarters) + LAG_DAYS) |>
  arrange(entry) |> mutate(exit = lead(entry, default = max(entry) + 90))
prices <- prices |>
  mutate(wi = findInterval(date, windows$entry)) |> filter(wi >= 1) |>
  mutate(quarter = windows$quarter[wi]) |> filter(date < windows$exit[wi]) |>
  select(-wi)

# trailing vol, lagged one day so the signal date can't see its own move
prices <- prices |>
  group_by(ck) |> arrange(date, .by_group = TRUE) |>
  mutate(vol = lag(slide_dbl(ret, sd, .before = VOL_WINDOW - 1, .complete = TRUE))) |>
  ungroup()

# ── Liquidity screen: keep the top LIQ_PCT of each quarter by entry market cap ─
liquid <- NULL
if (LIQUIDITY_SCREEN) {
  scope_ck <- if (CLUSTER_SCOPE == "healthcare") hc_ck$ck else unique(prices$ck)
  liquid <- prices |>
    filter(ck %in% scope_ck, !is.na(cap), cap > 0) |>
    group_by(quarter, ck) |> summarise(entry_cap = first(cap), .groups = "drop") |>  # cap at entry (PIT)
    group_by(quarter) |> filter(entry_cap >= quantile(entry_cap, 1 - LIQ_PCT)) |>
    ungroup() |> select(quarter, ck)
}

# ── Cluster per quarter (within HC, or the broad universe), auto-lowering K ───
liq_label <- if (LIQUIDITY_SCREEN) sprintf("top %d%% by cap", round(100 * LIQ_PCT)) else "off"
message(sprintf("clustering per quarter (scope %s | liquidity %s) ...", CLUSTER_SCOPE, liq_label))
cluster_list <- list()
for (q in quarters) {
  e <- emb |> filter(quarter == q) |> distinct(ck, .keep_all = TRUE)
  if (CLUSTER_SCOPE == "healthcare") e <- e |> semi_join(hc_ck, by = "ck")
  if (LIQUIDITY_SCREEN)              e <- e |> semi_join(filter(liquid, quarter == q), by = "ck")
  if (nrow(e) < 4 * N_QUANTILES) next                              # too thin for 2 clusters
  k <- min(N_CLUSTERS, max(2L, nrow(e) %/% (2 * N_QUANTILES)))     # effective K this quarter
  cluster_list[[q]] <- tibble(quarter = q, ck = e$ck,
                              cluster = assign_clusters(as.matrix(e[, dim_cols]), k))
}
clusters <- bind_rows(cluster_list)

# ── Panel: returns + cluster + GICS group + HC flag + vol ────────────────────
panel <- prices |>
  inner_join(clusters, by = c("quarter", "ck")) |>
  left_join(select(lookup, ck, gics_group, is_hc), by = "ck") |>
  mutate(is_hc = coalesce(is_hc, FALSE)) |>
  filter(!is.na(vol), vol > 0)

trading_days <- sort(unique(panel$date))
n_hc <- panel |> filter(is_hc) |> distinct(ck) |> nrow()
message(sprintf("panel: %s name-days | %d trading days | %s..%s | %d HC names traded",
                format(nrow(panel), big.mark = ","), length(trading_days),
                min(trading_days), max(trading_days), n_hc))

# ── SWEEP mode: horizon x cost grid of NET Sharpe ────────────────────────────
if (SWEEP) {
  rows <- list()
  for (h in SWEEP_HORIZONS) {
    panel_h <- add_cumret(panel, h)
    eng_emb  <- run_engine(panel_h, trading_days, "cluster", h)
    eng_gics <- if (RUN_GICS_BENCHMARK) run_engine(panel_h, trading_days, "gics_group", h) else NULL
    gross_sharpe <- daily_stats(eng_emb$gross$gross)$sharpe
    for (cst in SWEEP_COSTS) {
      d_emb    <- net_series(eng_emb, cst)
      emb_net  <- daily_stats(d_emb$net)$sharpe
      gics_net <- if (!is.null(eng_gics)) daily_stats(net_series(eng_gics, cst)$net)$sharpe else NA
      rows[[length(rows) + 1]] <- tibble(
        horizon = h, cost_bps = cst, emb_gross = gross_sharpe,
        emb_net = emb_net, gics_net = gics_net,
        edge = emb_net - gics_net, cost_drag = mean(d_emb$cost) * 252)
    }
  }
  grid <- bind_rows(rows)

  cat(sprintf("\nHorizon x cost sweep — NET Sharpe (scope %s | liquidity %s)\n\n",
              CLUSTER_SCOPE, liq_label))
  grid |> transmute(
    horizon = sprintf("%2dd", horizon), `cost(bps)` = cost_bps,
    `emb gross` = sprintf("%5.2f", emb_gross), `emb net` = sprintf("%5.2f", emb_net),
    `GICS net` = sprintf("%5.2f", gics_net), edge = sprintf("%+5.2f", edge),
    `emb drag` = sprintf("%4.1f%%", 100 * cost_drag)) |>
    as.data.frame() |> print(row.names = FALSE)
  cat("\nRead: pick (horizon, cost) where 'emb net' is healthy AND edge > 0 survives the\n")
  cat("harsher cost columns. 'emb gross' is cost-free signal quality.\n")
  write_csv(grid, "results/statarb_sweep.csv")
  cat("-> results/statarb_sweep.csv\n")
  quit(save = "no")
}

# ── SINGLE-RUN mode: one config, detailed gross/net table ────────────────────
panel_run <- add_cumret(panel, SIGNAL_WINDOW)
engines <- list(Embedding = run_engine(panel_run, trading_days, "cluster", REBALANCE_DAYS))
if (RUN_GICS_BENCHMARK)
  engines$GICS <- run_engine(panel_run, trading_days, "gics_group", REBALANCE_DAYS)

# daily gross/net/cost per engine (computed once, reused for the table and outputs)
daily <- list()
for (nm in names(engines)) daily[[nm]] <- net_series(engines[[nm]], COST_BPS, TRADING_COSTS)

cat(sprintf("\nEmbedding-clustered HC residual reversion%s\n",
            if (RUN_GICS_BENCHMARK) "  vs  GICS benchmark" else ""))
cat(sprintf("scope %s | liquidity %s | signal %dd / hold %dd | costs %s\n\n",
            CLUSTER_SCOPE, liq_label, SIGNAL_WINDOW, REBALANCE_DAYS,
            if (TRADING_COSTS) sprintf("ON (%g bps)", COST_BPS) else "OFF"))

summary_rows <- list()
for (nm in names(daily)) {
  for (leg in c("gross", "net")) {
    summary_rows[[length(summary_rows) + 1]] <-
      daily_stats(daily[[nm]][[leg]]) |> mutate(engine = nm, leg = leg, .before = 1)
  }
}
summary <- bind_rows(summary_rows)
summary |> transmute(engine, leg,
  `ann ret` = sprintf("%6.1f%%", 100 * ann_ret), `ann vol` = sprintf("%6.1f%%", 100 * ann_vol),
  Sharpe = sprintf("%6.2f", sharpe), `max DD` = sprintf("%6.1f%%", 100 * max_dd),
  `hit %` = sprintf("%5.1f%%", 100 * hit)) |>
  as.data.frame() |> print(row.names = FALSE)

cat("\nturnover / cost:\n")
for (nm in names(engines)) {
  cat(sprintf("  %-9s avg daily turnover %.2f | implied annual cost drag %.1f%%\n",
              nm, engines[[nm]]$avg_turnover, 100 * mean(daily[[nm]]$cost) * 252))
}

daily_out <- list()
for (nm in names(daily)) daily_out[[nm]] <- daily[[nm]] |> mutate(engine = nm, .before = 1)
write_parquet(bind_rows(daily_out), "results/statarb_daily.parquet")
write_csv(summary, "results/statarb_summary.csv")
cat("\n-> results/statarb_daily.parquet, results/statarb_summary.csv\n")