# =============================================================================
# Embedding-clustered daily residual reversion — US Health Care
# =============================================================================
# Daily statistical-arbitrage / mean-reversion within embedding-defined peer
# clusters, with POSITIONS RESTRICTED TO HEALTH CARE (the pitch universe).
#
# Each trading period (mid-quarter to mid-quarter, ~90d) uses the PRIOR quarter's
# OS-BERT embeddings (available with the 13F 45-day lag) to cluster stocks. Each
# day: within each cluster, long the underperformers and short the outperformers
# on a vol-adjusted residual (return minus cluster mean), betting on next-day
# reversion. The embeddings only DEFINE THE PEER GROUP — the alpha is short-
# horizon residual reversion, the robust, embedding-appropriate use (cf. Carlos /
# Gabaix-Koijen: "find inefficiencies within groups", not "predict price").
#
# Headline test: run the SAME engine with GICS industry groups in place of
# embedding clusters and compare. The pitch claim is "embedding peer groups beat
# GICS for HC relative value", which survives even a modest absolute Sharpe.
#
# Inputs (your existing artifacts):
#   embeddings_os/q_*.parquet   OS-BERT (isin, quarter_end, dim_*)
#   prices_crsp2.parquet        CRSP CIZ daily (cusip, dlycaldt, dlyret)
#   finratio.parquet            cusip -> gvkey bridge
#   gics.parquet                gvkey -> GICS sector / industry
#
#   Rscript healthcare_statarb.R
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

TRADE_SECTOR   <- "Health Care"  # positions restricted to this GICS sector (the pitch)
CLUSTER_SCOPE  <- "healthcare"   # "healthcare" = cluster WITHIN HC (clean, self-contained)
                                 # "all" = cluster the broad universe, trade only HC names
                                 #         (showcases cross-sector peers; needs broad returns
                                 #          in prices_crsp2.parquet)
N_CLUSTERS     <- 8              # embedding clusters (HC has ~6 GICS industries; keep close
                                 # to the GICS group count for a fair benchmark)
CLUSTER_METHOD <- "skmeans"      # "skmeans" (spherical, principled for L2-normalized vectors)
                                 # or "kmeans" (Euclidean fallback)
MIN_CLUSTER    <- 10             # drop a cluster on a day if it has fewer members than this

LAG_DAYS       <- 45             # 13F public ~45d after quarter-end -> point-in-time entry
VOL_WINDOW     <- 60             # trailing days for residual vol-adjustment
N_QUANTILES    <- 5              # quintiles (long bottom, short top)
REBALANCE_DAYS <- 1              # 1 = daily. >1 trades every k days (the main cost lever)

# ----- costs -----------------------------------------------------------------
TRADING_COSTS  <- TRUE   
COST_BPS       <- 5              # bps of traded notional per unit turnover (per rebalance)

# ----- GICS benchmark --------------------------------------------------------
RUN_GICS_BENCHMARK <- TRUE       # also run the GICS-grouped engine for comparison
GICS_LEVEL     <- "gind"         # GICS grouping for the benchmark: "gind" (industry, ~6 in
                                 # HC) or "ggroup" (industry group, ~2 in HC)

SEED <- 42; set.seed(SEED)

GICS_NAMES <- c("10"="Energy","15"="Materials","20"="Industrials","25"="Consumer Discretionary",
                "30"="Consumer Staples","35"="Health Care","40"="Financials",
                "45"="Information Technology","50"="Communication Services",
                "55"="Utilities","60"="Real Estate")

# ── Helpers ──────────────────────────────────────────────────────────────────
dim_names <- function(df) grep("^dim_", names(df), value = TRUE)

l2norm <- function(M) { M <- as.matrix(M); n <- sqrt(rowSums(M * M)); n[n == 0] <- 1; M / n }

# Standardise every identifier to the 8-char CUSIP (issuer+issue, no check digit):
# US ISIN = "US" + 9-char CUSIP, so chars 3..10 are the 8-char CUSIP; CRSP/finratio
# CUSIPs take their first 8. 8-char is the common key across all three sources.
ck8_cusip <- function(x) toupper(substr(trimws(as.character(x)), 1, 8))
ck8_isin  <- function(x) toupper(substr(trimws(as.character(x)), 3, 10))

# Trailing sample SD over a k-window, vectorised (rolling E[x^2]-E[x]^2).
# Names with fewer than k daily observations (short-lived / recently listed /
# delisted) can't fill a window -> return all-NA so they're dropped downstream.
roll_sd <- function(x, k) {
  n <- length(x)
  if (n < k) return(rep(NA_real_, n))
  m1 <- as.numeric(stats::filter(x,   rep(1 / k, k), sides = 1))
  m2 <- as.numeric(stats::filter(x^2, rep(1 / k, k), sides = 1))
  sqrt(pmax(m2 - m1^2, 0)) * sqrt(k / (k - 1))
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
  tibble(days = n,
         ann_ret = prod(1 + r)^(252 / n) - 1,
         ann_vol = sd(r) * sqrt(252),
         sharpe  = mean(r) / sd(r) * sqrt(252),
         max_dd  = min(dd),
         hit     = mean(r > 0),
         worst   = min(r),
         cum     = tail(cum, 1) - 1)
}

# ── Loaders ──────────────────────────────────────────────────────────────────
# ck -> GICS sector / industry group / HC flag (static; one row per CUSIP).
load_gics_lookup <- function() {
  fr <- read_parquet(FINRATIO_PARQUET)
  ckgv <- tibble(ck = ck8_cusip(fr$cusip), gvkey = as.character(fr$gvkey)) |>
    filter(ck != "", !is.na(gvkey), gvkey != "") |>
    distinct(ck, .keep_all = TRUE)
  g <- read_parquet(GICS_PARQUET) |>
    transmute(gvkey      = as.character(gvkey),
              gics_sector = unname(GICS_NAMES[as.character(gsector)]),
              gics_group  = as.character(.data[[GICS_LEVEL]]),
              conm) |>
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
    df  <- read_parquet(f)
    df  <- df[!is.na(df$isin), ]
    dc  <- dim_names(df)
    out <- df[, c("isin", dc)]
    out$quarter <- lab
    out$ck      <- ck8_isin(out$isin)
    out[, c("quarter", "ck", dc)]
  })
}

load_prices <- function() {
  p <- read_parquet(PRICES_PARQUET)
  ccol <- if ("cusip" %in% names(p)) "cusip"
          else if ("cusip9" %in% names(p)) "cusip9"
          else stop("prices file has no cusip / cusip9 column")
  tibble(ck = ck8_cusip(p[[ccol]]), date = as.Date(p$dlycaldt),
         ret = as.numeric(p$dlyret)) |>
    filter(ck != "", !is.na(date), !is.na(ret)) |>
    arrange(ck, date) |> distinct(ck, date, .keep_all = TRUE)
}

# ── The reversion engine (parameterised by the grouping column) ──────────────
run_engine <- function(panel, tdays, grp) {
  N         <- length(tdays)
  rebal_idx <- seq(1L, N, by = REBALANCE_DAYS)
  rebal_dt  <- tdays[rebal_idx]

  # target weights, formed on each rebalance date from that day's residuals
  sig <- panel |>
    filter(date %in% rebal_dt, !is.na(.data[[grp]]), !is.na(vol), vol > 0) |>
    group_by(date, across(all_of(grp))) |>
    filter(n() >= 2L * N_QUANTILES) |>                      # balanced quintiles only
    mutate(cmean = mean(ret), z = (ret - cmean) / vol) |>
    filter(is.finite(z)) |>
    mutate(qt   = ntile(z, N_QUANTILES),
           side = case_when(qt == 1 ~ 1L, qt == N_QUANTILES ~ -1L, TRUE ~ 0L),
           side = ifelse(is_hc, side, 0L)) |>                # positions only in HC
    mutate(nl = sum(side == 1L), ns = sum(side == -1L),
           wraw = case_when(side ==  1L & nl > 0 ~  1 / nl,
                            side == -1L & ns > 0 ~ -1 / ns, TRUE ~ 0)) |>
    ungroup() |>
    filter(wraw != 0) |>
    group_by(date) |>
    mutate(w = wraw / sum(abs(wraw))) |>                     # gross leverage = 1 per day
    ungroup() |>
    select(date, ck, w)

  if (nrow(sig) == 0) stop(sprintf("no positions for grouping '%s' — check universe/clusters", grp))

  # turnover per rebalance date (calendar-aligned to the prior rebalance)
  rmap <- tibble(date = rebal_dt, ridx = seq_along(rebal_dt))
  sigr <- sig |> inner_join(rmap, by = "date")
  prev <- sigr |> transmute(ridx = ridx + 1L, ck, w_prev = w)
  turn <- full_join(sigr |> select(ridx, ck, w), prev, by = c("ridx", "ck")) |>
    mutate(w = coalesce(w, 0), w_prev = coalesce(w_prev, 0)) |>
    group_by(ridx) |> summarise(turnover = sum(abs(w - w_prev)), .groups = "drop") |>
    inner_join(rmap, by = "ridx") |> select(date, turnover)

  # governing rebalance date for each trading day = largest rebalance strictly before t
  gov_pos  <- findInterval(seq_len(N) - 1L, rebal_idx)       # 0 where no prior rebalance
  daymap   <- tibble(date = tdays,
                     gov_date = as.Date(ifelse(gov_pos >= 1,
                                  as.character(tdays[rebal_idx[pmax(gov_pos, 1L)]]), NA))) |>
    filter(!is.na(gov_date))

  rday <- panel |> distinct(date, ck, ret)

  port <- daymap |>
    inner_join(sig,  by = c("gov_date" = "date")) |>         # weights active today
    inner_join(rday, by = c("date", "ck")) |>                # today's realised return
    group_by(date) |> summarise(gross = sum(w * ret), .groups = "drop")

  cost <- if (TRADING_COSTS)
            turn |> transmute(date, cost = (COST_BPS / 1e4) * turnover)
          else tibble(date = as.Date(character()), cost = numeric())

  daily <- port |> left_join(cost, by = "date") |>
    mutate(cost = coalesce(cost, 0), net = gross - cost) |> arrange(date)

  list(daily = daily,
       gross = daily_stats(daily$gross),
       net   = daily_stats(daily$net),
       avg_turnover   = mean(turn$turnover, na.rm = TRUE),
       ann_cost_drag  = mean(daily$cost,   na.rm = TRUE) * 252)
}

# ── Main ─────────────────────────────────────────────────────────────────────
main <- function() {
  dir.create("results", showWarnings = FALSE)
  message("loading inputs ...")
  gl  <- load_gics_lookup()
  emb <- load_emb()
  px  <- load_prices()

  # trading windows: quarter q -> [as.Date(q) + LAG, next entry)
  qs  <- sort(unique(emb$quarter))
  win <- tibble(quarter = qs, entry = as.Date(qs) + LAG_DAYS) |> arrange(entry) |>
    mutate(exit = lead(entry, default = max(entry) + 90))

  px <- px |> mutate(wi = findInterval(date, win$entry)) |> filter(wi >= 1)
  px$quarter <- win$quarter[px$wi]
  px <- px |> filter(date < win$exit[px$wi]) |> select(-wi)

  # trailing vol + next-day return per name (no look-ahead: vol lagged one day)
  px <- px |> group_by(ck) |> arrange(date, .by_group = TRUE) |>
    mutate(vol = lag(roll_sd(ret, VOL_WINDOW))) |> ungroup()

  # cluster per window (quarter), on HC-only or the broad universe
  message(sprintf("clustering per quarter (%s, scope=%s) ...",
                  if (has_skmeans && CLUSTER_METHOD == "skmeans") "skmeans" else "kmeans",
                  CLUSTER_SCOPE))
  hc_ck <- gl |> filter(is_hc) |> distinct(ck)
  clus <- map_dfr(qs, function(q) {
    e  <- emb |> filter(quarter == q) |> distinct(ck, .keep_all = TRUE)
    if (CLUSTER_SCOPE == "healthcare") e <- e |> inner_join(hc_ck, by = "ck")
    dc <- dim_names(e)
    if (nrow(e) < max(MIN_CLUSTER, 2L * N_QUANTILES)) return(NULL)
    tibble(quarter = q, ck = e$ck, cluster = cluster_assign(as.matrix(e[, dc]), N_CLUSTERS))
  })

  # daily panel: returns + window + embedding cluster + GICS group + HC flag + vol
  panel <- px |>
    inner_join(clus, by = c("quarter", "ck")) |>
    left_join(gl |> select(ck, gics_group, is_hc), by = "ck") |>
    mutate(is_hc = coalesce(is_hc, FALSE)) |>
    filter(!is.na(vol), vol > 0)

  tdays <- sort(unique(panel$date))
  n_hc  <- panel |> filter(is_hc) |> distinct(ck) |> nrow()
  message(sprintf("panel: %s name-days | %d trading days | %s..%s | %d HC names traded",
                  format(nrow(panel), big.mark = ","), length(tdays),
                  min(tdays), max(tdays), n_hc))

  emb_res <- run_engine(panel, tdays, "cluster")
  res <- list(Embedding = emb_res)
  if (RUN_GICS_BENCHMARK) res[["GICS"]] <- run_engine(panel, tdays, "gics_group")

  # ── report ──
  cat(sprintf("\nEmbedding-clustered HC daily reversion%s\n",
              if (RUN_GICS_BENCHMARK) "  vs  GICS benchmark" else ""))
  cat(sprintf("scope: %s | clusters: %d (%s) | rebal: every %dd | costs: %s\n\n",
              CLUSTER_SCOPE, N_CLUSTERS,
              if (has_skmeans && CLUSTER_METHOD == "skmeans") "skmeans" else "kmeans",
              REBALANCE_DAYS, if (TRADING_COSTS) sprintf("ON (%g bps/side)", COST_BPS) else "OFF"))

  rows <- imap_dfr(res, function(r, nm) {
    bind_rows(
      r$gross |> mutate(engine = nm, leg = "gross", .before = 1),
      r$net   |> mutate(engine = nm, leg = "net",   .before = 1))
  })
  rows |>
    transmute(engine, leg,
              `ann ret` = sprintf("%6.1f%%", 100 * ann_ret),
              `ann vol` = sprintf("%6.1f%%", 100 * ann_vol),
              Sharpe    = sprintf("%6.2f", sharpe),
              `max DD`  = sprintf("%6.1f%%", 100 * max_dd),
              `hit %`   = sprintf("%5.1f%%", 100 * hit)) |>
    as.data.frame() |> print(row.names = FALSE)

  cat("\nturnover / cost:\n")
  iwalk(res, function(r, nm)
    cat(sprintf("  %-9s avg daily turnover %.2f   implied annual cost drag %.1f%%\n",
                nm, r$avg_turnover, 100 * r$ann_cost_drag)))

  cat("\nnotes:\n")
  cat("  - Daily rebalance => high turnover; the NET row is the honest number.\n")
  cat("  - If net Sharpe is poor, raise REBALANCE_DAYS (3, 5) to cut turnover, or\n")
  cat("    restrict to liquid names — that's the lever, not the cost rate.\n")
  cat("  - Headline claim is Embedding net Sharpe > GICS net Sharpe.\n")

  daily_out <- imap_dfr(res, function(r, nm) r$daily |> mutate(engine = nm, .before = 1))
  write_parquet(daily_out, "results/statarb_daily.parquet")
  write_csv(rows, "results/statarb_summary.csv")
  cat("\n-> results/statarb_daily.parquet, results/statarb_summary.csv\n")
  invisible(res)
}

main()
