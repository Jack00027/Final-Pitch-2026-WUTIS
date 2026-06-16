# =============================================================================
# Embedding cluster MOMENTUM — US Health Care (long only)
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse); library(arrow); library(lubridate)
})
has_skmeans <- requireNamespace("skmeans", quietly = TRUE)   # install.packages("skmeans")

# ── Parameters ───────────────────────────────────────────────────────────────
EMB_DIR          <- "embeddings_os"
PRICES_PARQUET   <- "prices_crsp2.parquet"
FINRATIO_PARQUET <- "finratio.parquet"
GICS_PARQUET     <- "gics.parquet"
OUT_DIR          <- "results_momentum"          # separate from the reversion results/

TRADE_SECTOR   <- "Health Care"  # positions restricted to this GICS sector
CLUSTER_SCOPE  <- "healthcare"   # "healthcare" = cluster within HC; "all" = cluster broad,
                                 # rank clusters on all members, hold the HC firms in winners
N_CLUSTERS     <- 8              # embedding clusters (same as the reversion file)
CLUSTER_METHOD <- "skmeans"      # "skmeans" (spherical) or "kmeans" (fallback)
MIN_PER_CLUSTER<- 5              # drop a quarter if it can't form 2 clusters of this size
LAG_DAYS       <- 45             # 13F public ~45d after quarter-end -> point-in-time entry

# ----- the momentum trade ----------------------------------------------------
REBALANCE_DAYS   <- 21          # holding period, in TRADING days (~21 = one month, default)
MOMENTUM_WINDOW  <- REBALANCE_DAYS   # rank clusters on their return over the previous this-many
                                     # trading days (default = the rebalance period)
TOP_CLUSTER_FRAC <- 0.25         # hold firms in the top this-fraction of clusters by trailing
                                 # return (0.50 = top half; lower = more concentrated)

# ----- liquidity screen (same as the reversion file) -------------------------
LIQUIDITY_SCREEN <- TRUE         # keep only the most liquid names (market-cap ranked) so
                                 # the cost model is defensible and shorts are feasible
LIQ_PCT          <- 1

# ----- costs -----------------------------------------------------------------
TRADING_COSTS  <- TRUE           # <<< MASTER SWITCH: FALSE = gross >>>
COST_BPS       <- 5              # bps of traded notional per unit turnover

# ----- GICS benchmark --------------------------------------------------------
RUN_GICS_BENCHMARK <- TRUE
GICS_LEVEL     <- "gind"

SEED <- 42; set.seed(SEED)

GICS_NAMES <- c("10"="Energy","15"="Materials","20"="Industrials","25"="Consumer Discretionary",
                "30"="Consumer Staples","35"="Health Care","40"="Financials",
                "45"="Information Technology","50"="Communication Services",
                "55"="Utilities","60"="Real Estate")

# ── Helpers ──────────────────────────────────────────────────────────────────
dim_names <- function(df) grep("^dim_", names(df), value = TRUE)
l2norm    <- function(M) { M <- as.matrix(M); n <- sqrt(rowSums(M * M)); n[n == 0] <- 1; M / n }
ck8_cusip <- function(x) toupper(substr(trimws(as.character(x)), 1, 8))
ck8_isin  <- function(x) toupper(substr(trimws(as.character(x)), 3, 10))

# trailing k-day compounded return ending at each point (k=1 -> the daily return)
roll_cumret <- function(x, k) {
  if (k <= 1L) return(x)
  n <- length(x); if (n < k) return(rep(NA_real_, n))
  expm1(as.numeric(stats::filter(log1p(x), rep(1, k), sides = 1)))
}

cluster_assign <- function(M, k) {
  M <- l2norm(M); n <- nrow(M); k <- min(k, max(2L, n - 1L))
  if (CLUSTER_METHOD == "skmeans" && has_skmeans)
    as.integer(skmeans::skmeans(M, k)$cluster)
  else
    as.integer(stats::kmeans(M, centers = k, nstart = 5, iter.max = 50)$cluster)
}

daily_stats <- function(r) {
  r <- r[!is.na(r)]; n <- length(r)
  if (n < 2) return(tibble(days = n, ann_ret = NA, ann_vol = NA, sharpe = NA,
                           max_dd = NA, hit = NA, worst = NA, cum = NA))
  cum <- cumprod(1 + r); dd <- cum / cummax(cum) - 1
  tibble(days = n, ann_ret = prod(1 + r)^(252 / n) - 1, ann_vol = sd(r) * sqrt(252),
         sharpe = mean(r) / sd(r) * sqrt(252), max_dd = min(dd),
         hit = mean(r > 0), worst = min(r), cum = tail(cum, 1) - 1)
}

# ── Loaders (identical to the reversion strategy) ────────────────────────────
load_gics_lookup <- function() {
  fr <- read_parquet(FINRATIO_PARQUET)
  ckgv <- tibble(ck = ck8_cusip(fr$cusip), gvkey = as.character(fr$gvkey)) |>
    filter(ck != "", !is.na(gvkey), gvkey != "") |> distinct(ck, .keep_all = TRUE)
  g <- read_parquet(GICS_PARQUET) |>
    transmute(gvkey = as.character(gvkey),
              gics_sector = unname(GICS_NAMES[as.character(gsector)]),
              gics_group  = as.character(.data[[GICS_LEVEL]]), conm) |>
    distinct(gvkey, .keep_all = TRUE)
  ckgv |> left_join(g, by = "gvkey") |>
    transmute(ck, gics_sector, gics_group,
              is_hc = !is.na(gics_sector) & gics_sector == TRADE_SECTOR, conm)
}

load_emb <- function() {
  files <- list.files(EMB_DIR, pattern = "^q_.*\\.parquet$", full.names = TRUE)
  if (length(files) == 0) stop(sprintf("no q_*.parquet under %s", EMB_DIR))
  map_dfr(files, function(f) {
    lab <- sub("^q_", "", tools::file_path_sans_ext(basename(f)))
    df  <- read_parquet(f); df <- df[!is.na(df$isin), ]
    dc  <- dim_names(df); out <- df[, c("isin", dc)]
    out$quarter <- lab; out$ck <- ck8_isin(out$isin)
    out[, c("quarter", "ck", dc)]
  })
}

load_prices <- function() {
  p <- read_parquet(PRICES_PARQUET)
  ccol <- if ("cusip" %in% names(p)) "cusip" else if ("cusip9" %in% names(p)) "cusip9"
          else stop("prices file has no cusip / cusip9 column")
  tibble(ck = ck8_cusip(p[[ccol]]), date = as.Date(p$dlycaldt),
         ret = as.numeric(p$dlyret),
         cap = if ("dlycap" %in% names(p)) as.numeric(p$dlycap) else NA_real_) |>
    filter(ck != "", !is.na(date), !is.na(ret)) |>
    arrange(ck, date) |> distinct(ck, date, .keep_all = TRUE)
}

# ── Backtest plumbing: target weights on rebalance dates -> daily gross + turnover ─
# (same governing-date mechanism as the reversion engine; weights set at a
# rebalance date are held over every day it governs and earn that day's return)
backtest_weights <- function(sig, panel, tdays, rebal_dt) {
  rmap <- tibble(date = rebal_dt, ridx = seq_along(rebal_dt))
  sigr <- sig |> inner_join(rmap, by = "date")
  prev <- sigr |> transmute(ridx = ridx + 1L, ck, w_prev = w)
  turn <- full_join(sigr |> select(ridx, ck, w), prev, by = c("ridx", "ck")) |>
    mutate(w = coalesce(w, 0), w_prev = coalesce(w_prev, 0)) |>
    group_by(ridx) |> summarise(turnover = sum(abs(w - w_prev)), .groups = "drop") |>
    inner_join(rmap, by = "ridx") |> select(date, turnover)

  N <- length(tdays); rebal_idx <- match(rebal_dt, tdays)
  gov_pos <- findInterval(seq_len(N) - 1L, rebal_idx)
  daymap  <- tibble(date = tdays,
                    gov_date = as.Date(ifelse(gov_pos >= 1,
                                 as.character(tdays[rebal_idx[pmax(gov_pos, 1L)]]), NA))) |>
    filter(!is.na(gov_date))
  rday <- panel |> distinct(date, ck, ret)

  gross_daily <- daymap |>
    inner_join(sig,  by = c("gov_date" = "date"), relationship = "many-to-many") |>
    inner_join(rday, by = c("date", "ck")) |>
    group_by(date) |> summarise(gross = sum(w * ret), .groups = "drop") |> arrange(date)

  list(gross_daily = gross_daily, turn = turn, avg_turnover = mean(turn$turnover, na.rm = TRUE))
}

net_series <- function(eng, cost_bps, trading_costs = TRUE) {
  cost <- if (trading_costs) eng$turn |> transmute(date, cost = (cost_bps / 1e4) * turnover)
          else tibble(date = as.Date(character()), cost = numeric())
  eng$gross_daily |> left_join(cost, by = "date") |>
    mutate(cost = coalesce(cost, 0), net = gross - cost) |> arrange(date)
}

engine_stats <- function(eng, cost_bps, trading_costs = TRUE) {
  d <- net_series(eng, cost_bps, trading_costs)
  list(daily = d, gross = daily_stats(d$gross), net = daily_stats(d$net),
       avg_turnover = eng$avg_turnover, ann_cost_drag = mean(d$cost) * 252)
}

# ── The momentum engine ──────────────────────────────────────────────────────
# At each rebalance: rank groups by trailing MOMENTUM_WINDOW-day return (the panel's
# `cumret`), keep the top TOP_CLUSTER_FRAC of groups, hold their HC firms equal-weight.
momentum_engine <- function(panel, tdays, grp, hold) {
  rebal_dt <- tdays[seq(1L, length(tdays), by = hold)]
  sig <- panel |>
    filter(date %in% rebal_dt, !is.na(cumret), !is.na(.data[[grp]])) |>
    group_by(date, across(all_of(grp))) |>
    mutate(grp_mom = mean(cumret)) |>                       # group's trailing return (all members)
    ungroup() |>
    group_by(date) |>
    mutate(n_grp = n_distinct(.data[[grp]]),
           keep  = dense_rank(desc(grp_mom)) <= pmax(1L, ceiling(TOP_CLUSTER_FRAC * n_grp))) |>
    ungroup() |>
    filter(keep, is_hc) |>                                  # hold only HC firms in the top groups
    group_by(date) |> mutate(w = 1 / n()) |> ungroup() |>   # equal-weight, fully invested long
    select(date, ck, w)
  if (nrow(sig) == 0) stop(sprintf("no positions for grouping '%s' — check universe/clusters", grp))
  backtest_weights(sig, panel, tdays, rebal_dt)
}

# ── Main ─────────────────────────────────────────────────────────────────────
main <- function() {
  dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
  message("loading inputs ...")
  gl  <- load_gics_lookup(); emb <- load_emb(); px <- load_prices()

  qs  <- sort(unique(emb$quarter))
  win <- tibble(quarter = qs, entry = as.Date(qs) + LAG_DAYS) |> arrange(entry) |>
    mutate(exit = lead(entry, default = max(entry) + 90))
  px <- px |> mutate(wi = findInterval(date, win$entry)) |> filter(wi >= 1)
  px$quarter <- win$quarter[px$wi]
  px <- px |> filter(date < win$exit[px$wi]) |> select(-wi)

  hc_ck <- gl |> filter(is_hc) |> distinct(ck)

  # liquidity screen (identical to the reversion file): top LIQ_PCT by entry market cap
  liq <- NULL
  if (LIQUIDITY_SCREEN) {
    if (all(is.na(px$cap)))
      stop("LIQUIDITY_SCREEN=TRUE but prices file carries no market cap (dlycap). ",
           "Set LIQUIDITY_SCREEN=FALSE, or add dlycap to wrds_pull_crsp.R and re-pull.")
    uni_ck <- if (CLUSTER_SCOPE == "healthcare") hc_ck$ck else unique(px$ck)
    liq <- px |> filter(ck %in% uni_ck, !is.na(cap), cap > 0) |>
      group_by(quarter, ck) |> arrange(date, .by_group = TRUE) |>
      summarise(cap0 = dplyr::first(cap), .groups = "drop") |>
      group_by(quarter) |>
      filter(cap0 >= quantile(cap0, 1 - LIQ_PCT, na.rm = TRUE)) |>
      ungroup() |> select(quarter, ck)
  }

  # cluster per quarter (identical machinery), auto-lowering K when thin
  message(sprintf("clustering per quarter (%s, scope=%s%s) ...",
                  if (has_skmeans && CLUSTER_METHOD == "skmeans") "skmeans" else "kmeans",
                  CLUSTER_SCOPE,
                  if (LIQUIDITY_SCREEN) sprintf(", liquid top %d%%", round(100 * LIQ_PCT)) else ""))
  clus <- map_dfr(qs, function(q) {
    e <- emb |> filter(quarter == q) |> distinct(ck, .keep_all = TRUE)
    if (CLUSTER_SCOPE == "healthcare") e <- e |> inner_join(hc_ck, by = "ck")
    if (LIQUIDITY_SCREEN) e <- e |> inner_join(filter(liq, quarter == q) |> select(ck), by = "ck")
    if (nrow(e) < 2L * MIN_PER_CLUSTER) return(NULL)
    kq <- min(N_CLUSTERS, max(2L, nrow(e) %/% MIN_PER_CLUSTER))
    dc <- dim_names(e)
    tibble(quarter = q, ck = e$ck, cluster = cluster_assign(as.matrix(e[, dc]), kq))
  })

  base_panel <- px |>
    inner_join(clus, by = c("quarter", "ck")) |>
    left_join(gl |> select(ck, gics_group, is_hc), by = "ck") |>
    mutate(is_hc = coalesce(is_hc, FALSE))

  tdays   <- sort(unique(base_panel$date))
  n_names <- base_panel |> filter(is_hc) |> distinct(ck) |> nrow()
  message(sprintf("panel: %s name-days | %d trading days | %s..%s | %d HC names",
                  format(nrow(base_panel), big.mark = ","), length(tdays),
                  min(tdays), max(tdays), n_names))

  # trailing-window return for the momentum signal
  panel <- base_panel |> group_by(ck) |> arrange(date, .by_group = TRUE) |>
    mutate(cumret = roll_cumret(ret, MOMENTUM_WINDOW)) |> ungroup()

  # ── run momentum (embedding clusters, and GICS industries as benchmark) ──
  e_emb <- momentum_engine(panel, tdays, "cluster", REBALANCE_DAYS)
  res <- list(`Embedding mom` = engine_stats(e_emb, COST_BPS, TRADING_COSTS))
  if (RUN_GICS_BENCHMARK) {
    e_gics <- momentum_engine(panel, tdays, "gics_group", REBALANCE_DAYS)
    res[["GICS mom"]] <- engine_stats(e_gics, COST_BPS, TRADING_COSTS)
  }

  # passive benchmark: equal-weight liquid HC (same universe, daily)
  bench <- base_panel |> filter(is_hc) |> distinct(date, ck, ret) |>
    group_by(date) |> summarise(bench = mean(ret), .groups = "drop") |> arrange(date)
  bench_stats <- daily_stats(bench$bench)

  # ── report ──
  liq_lbl <- if (LIQUIDITY_SCREEN) sprintf("top %d%% by mktcap", round(100 * LIQ_PCT)) else "off"
  cat(sprintf("\nEmbedding cluster MOMENTUM — US Health Care (long only)\n"))
  cat(sprintf("scope: %s | liquidity %s | momentum %dd / hold %dd | top %d%% of clusters | costs: %s\n\n",
              CLUSTER_SCOPE, liq_lbl, MOMENTUM_WINDOW, REBALANCE_DAYS,
              round(100 * TOP_CLUSTER_FRAC),
              if (TRADING_COSTS) sprintf("ON (%g bps/side)", COST_BPS) else "OFF"))

  rows <- bind_rows(
    imap_dfr(res, function(r, nm) bind_rows(
      r$gross |> mutate(strategy = paste(nm, "gross"), .before = 1),
      r$net   |> mutate(strategy = paste(nm, "net"),   .before = 1))),
    bench_stats |> mutate(strategy = "EW HC (passive)", .before = 1))
  rows |>
    transmute(strategy,
              `ann ret` = sprintf("%6.1f%%", 100 * ann_ret),
              `ann vol` = sprintf("%6.1f%%", 100 * ann_vol),
              Sharpe    = sprintf("%6.2f", sharpe),
              `max DD`  = sprintf("%6.1f%%", 100 * max_dd),
              `hit %`   = sprintf("%5.1f%%", 100 * hit)) |>
    as.data.frame() |> print(row.names = FALSE)

  cat("\nturnover / cost:\n")
  iwalk(res, function(r, nm)
    cat(sprintf("  %-14s avg rebal turnover %.2f   implied annual cost drag %.1f%%\n",
                nm, r$avg_turnover, 100 * r$ann_cost_drag)))

  # does embedding momentum beat just owning the sector? (excess over EW HC)
  exc <- res[["Embedding mom"]]$daily |> inner_join(bench, by = "date") |>
    mutate(exc = net - bench)
  es <- daily_stats(exc$exc)
  cat(sprintf("\nEmbedding momentum (net) vs equal-weight HC:\n"))
  cat(sprintf("  excess return %+.1f%%/yr | info ratio %.2f | tracking error %.1f%%\n",
              100 * es$ann_ret, es$sharpe, 100 * es$ann_vol))
  cat("  (positive info ratio => the cluster tilt adds return per unit of active risk;\n")
  cat("   long-only carries HC beta, so beating EW HC is the bar, not just being positive.)\n")

  daily_out <- bind_rows(
    imap_dfr(res, function(r, nm) r$daily |> transmute(date, strategy = nm, ret = net)),
    bench |> transmute(date, strategy = "EW HC", ret = bench))
  write_parquet(daily_out, file.path(OUT_DIR, "momentum_daily.parquet"))
  write_csv(rows, file.path(OUT_DIR, "momentum_summary.csv"))
  cat(sprintf("\n-> %s/momentum_daily.parquet, %s/momentum_summary.csv\n", OUT_DIR, OUT_DIR))
  invisible(res)
}

main()
