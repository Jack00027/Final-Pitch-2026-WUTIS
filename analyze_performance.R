# =============================================================================
# Performance analysis — embedding-implied valuation L/S book
# =============================================================================
# Reads results/signal_check.csv (per-quarter long_ret, short_ret, ls_spread,
# spearman, p_value) written by WUTIS_book.R / RelativeValue_strategy.r and
# reports portfolio-level performance.
#
# The holding window is [quarter_end + 45d, +135d] and rebalance is quarterly,
# so consecutive windows are contiguous and NON-overlapping — the per-quarter
# ls_spread is a clean ~quarterly return series. Annualize by 4 (return) and
# sqrt(4)=2 (vol), the standard iid scaling.
#
#   Rscript analyze_performance.R
# =============================================================================
suppressPackageStartupMessages({ library(tidyverse) })

CHECK_CSV <- "results/signal_check.csv"
PPY       <- 4      # periods per year (quarterly rebalance)
RF_ANN    <- 0      # annual risk-free to net out; e.g. 0.02 for 2%

stopifnot(file.exists(CHECK_CSV))
chk <- read_csv(CHECK_CSV, show_col_types = FALSE) |>
  arrange(quarter_end) |>
  filter(!is.na(ls_spread))

if (nrow(chk) < 8)
  warning(sprintf("only %d quarters — annualized stats are noisy", nrow(chk)))

# ── return statistics for one return series (a leg or the L/S portfolio) ──────
ret_stats <- function(r) {
  r  <- r[!is.na(r)]; n <- length(r)
  mu <- mean(r); sdv <- sd(r)
  rf_q <- (1 + RF_ANN)^(1 / PPY) - 1
  cum  <- cumprod(1 + r)
  dd   <- cum / cummax(cum) - 1
  tibble(
    n_qtrs     = n,
    mean_qtr   = mu,
    ann_return = (1 + mu)^PPY - 1,             # geometric scale-up of the mean qtr
    cagr       = prod(1 + r)^(PPY / n) - 1,    # realized compound annual growth
    ann_vol    = sdv * sqrt(PPY),
    sharpe     = ((mu - rf_q) / sdv) * sqrt(PPY),
    t_stat     = mu / (sdv / sqrt(n)),         # H0: mean quarterly return = 0
    hit_rate   = mean(r > 0),
    best_qtr   = max(r),
    worst_qtr  = min(r),
    cum_return = tail(cum, 1) - 1,
    max_dd     = min(dd)
  )
}

perf <- bind_rows(
  ret_stats(chk$ls_spread) |> mutate(leg = "Long-Short", .before = 1),
  ret_stats(chk$long_ret)  |> mutate(leg = "Long leg",   .before = 1),
  ret_stats(chk$short_ret) |> mutate(leg = "Short leg",  .before = 1)
)

# ── rank information coefficient ──────────────────────────────────────────────
# signal = z(-ep_resid) is an OVERVALUATION score (high => short). A predictive
# signal therefore has NEGATIVE raw Spearman(signal, fwd_ret); flip the sign so
# positive IC == "signal works", consistent with ls_spread > 0 being good.
ic_series <- -chk$spearman
nic <- sum(!is.na(ic_series))
ic <- tibble(
  mean_ic     = mean(ic_series, na.rm = TRUE),
  ic_sd       = sd(ic_series, na.rm = TRUE),
  ic_ir       = mean(ic_series, na.rm = TRUE) / sd(ic_series, na.rm = TRUE),
  ic_t        = mean(ic_series, na.rm = TRUE) / (sd(ic_series, na.rm = TRUE) / sqrt(nic)),
  pct_pos     = mean(ic_series > 0, na.rm = TRUE),
  mean_pvalue = mean(chk$p_value, na.rm = TRUE)
)

# ── report ───────────────────────────────────────────────────────────────────
pct <- function(x) sprintf("%7.2f%%", 100 * x)
num <- function(x) sprintf("%7.2f",  x)

cat("\nEmbedding-implied valuation L/S — US Health Care\n")
cat(sprintf("quarters: %d   (%s  ->  %s)   rf = %.1f%%/yr\n\n",
            nrow(chk), min(chk$quarter_end), max(chk$quarter_end), 100 * RF_ANN))

perf |>
  transmute(leg,
            `mean q`  = pct(mean_qtr),
            `ann ret` = pct(ann_return),
            CAGR      = pct(cagr),
            `ann vol` = pct(ann_vol),
            Sharpe    = num(sharpe),
            `t-stat`  = num(t_stat),
            `hit %`   = pct(hit_rate),
            `best q`  = pct(best_qtr),
            `worst q` = pct(worst_qtr),
            `cum ret` = pct(cum_return),
            `max DD`  = pct(max_dd)) |>
  as.data.frame() |> print(row.names = FALSE)

cat(sprintf("\nRank IC (predictive sign), %d quarters:\n", nrow(chk)))
cat(sprintf("  mean IC       %s\n", num(ic$mean_ic)))
cat(sprintf("  IC IR         %s   (mean / sd)\n", num(ic$ic_ir)))
cat(sprintf("  IC t-stat     %s\n", num(ic$ic_t)))
cat(sprintf("  %% positive    %s\n", pct(ic$pct_pos)))
cat(sprintf("  mean p-value  %s\n", num(ic$mean_pvalue)))

cat("\nnotes:\n")
cat("  - Long-Short = long_ret - short_ret (equal-weight, dollar-neutral).\n")
cat("  - Short-leg column is the RAW return of the shorted names; you earn its\n")
cat("    negative, so a LOWER short-leg return is better.\n")
cat("  - Sharpe assumes rf = 0 unless RF_ANN is set; annualized via sqrt(4).\n")

write_csv(perf, "results/performance_summary.csv")
write_csv(ic,   "results/performance_ic.csv")
cat("\n-> results/performance_summary.csv, results/performance_ic.csv\n")
