# =============================================================================
# generate_picks_crsp.r  —  HC Long/Short from latest embeddings + CRSP prices
# =============================================================================
# Same as the live version but sources market equity from prices_crsp2.parquet
# (dlycap = market cap) instead of Yahoo. No ticker mapping, no rate limits.
# Trade-off: CRSP is not real-time, so the "current" price is the LAST date in
# your file (reported below) — not actually today.
#   Rscript generate_picks_crsp.r
# =============================================================================

library(tidyverse)
library(arrow)
library(lubridate)
library(glmnet)

EMB_DIR        <- "embeddings_os"
EMB_QUARTER    <- "2025-12-31"   # set to the actual filename date of your latest embedding
PORTFOLIO_SIZE <- 5
LIQ_PCT        <- 0.50           # keep top X% of HC by market cap (1 = no screen)

l2_normalize <- function(m) { n <- sqrt(rowSums(m^2)); n[n == 0] <- 1; m / n }
winsorize    <- function(x, p = 0.01) { b <- quantile(x, c(p, 1 - p), na.rm = TRUE); pmin(pmax(x, b[1]), b[2]) }

# ── 1. HC universe: latest book equity per name ──────────────────────────────
hc_gvkeys <- read_parquet("gics.parquet") |>
  mutate(gvkey = as.character(gvkey)) |>
  filter(substr(as.character(gsector), 1, 2) == "35") |>
  pull(gvkey)

hc_book <- read_parquet("fundamentals_merged.parquet") |>
  mutate(gvkey = as.character(gvkey)) |>
  filter(gvkey %in% hc_gvkeys, book_equity > 0) |>
  group_by(ck) |> slice_max(quarter, n = 1, with_ties = FALSE) |> ungroup() |>
  select(ck, book_equity)

# ── 1b. US universe: filter to US ISINs from the latest embeddings ───────────
us_cks <- read_parquet(file.path(EMB_DIR, paste0("q_", EMB_QUARTER, ".parquet")), col_select = "isin") |>
  filter(substr(isin, 1, 2) == "US") |>
  mutate(ck = toupper(substr(isin, 3, 10))) |>
  pull(ck)

hc_book <- hc_book |> filter(ck %in% us_cks)

# ── 2. Latest CRSP market cap + ticker per name (no Yahoo) ────────────────────
crsp_latest <- read_parquet("prices_crsp2.parquet") |>
  transmute(ck = toupper(substr(cusip, 1, 8)), date = as.Date(dlycaldt),
            ticker, market_equity = as.numeric(dlycap)) |>
  filter(ck %in% hc_book$ck, market_equity > 0) |>
  group_by(ck) |> slice_max(date, n = 1, with_ties = FALSE) |> ungroup()

message(sprintf("CRSP prices as of: %s (this is the 'current' date, not today)",
                max(crsp_latest$date)))

val <- hc_book |> inner_join(crsp_latest, by = "ck")

# ── 2b. Liquidity screen ─────────────────────────────────────────────────────
if (LIQ_PCT < 1) {
  val <- val |> filter(market_equity >= quantile(market_equity, 1 - LIQ_PCT, na.rm = TRUE))
  message(sprintf("liquidity screen: top %d%% of US HC by market cap -> %d names",
                  round(100 * LIQ_PCT), nrow(val)))
}

# ── 3. Valuation residual on the US HC cross-section ─────────────────────────
val <- val |> mutate(log_me = winsorize(log(market_equity)), log_denom = winsorize(log(book_equity)))
fit_v <- lm(log_me ~ log_denom, data = val)
val$actual_p_perp <- as.numeric(residuals(fit_v))
message(sprintf("Priced %d US HC names | gamma %.2f", nrow(val), coef(fit_v)[["log_denom"]]))

# ── 4. Ridge: embeddings -> p_perp (US HC only, in-sample) ───────────────────
emb <- read_parquet(file.path(EMB_DIR, paste0("q_", EMB_QUARTER, ".parquet"))) |>
  mutate(ck = toupper(substr(isin, 3, 10))) |>
  select(-isin)

# (val is already pre-filtered to US in step 1b, so this inner_join enforces it)
d <- val |> inner_join(emb, by = "ck")
dim_cols <- grep("^dim_", names(d), value = TRUE)
X <- l2_normalize({ m <- as.matrix(d[, dim_cols]); mode(m) <- "numeric"; m })
y <- d$actual_p_perp

ridge <- cv.glmnet(X, y, alpha = 0, standardize = FALSE, nfolds = 10)
d$ai_predicted_p_perp <- as.numeric(predict(ridge, s = "lambda.min", newx = X))
d$mispricing <- d$actual_p_perp - d$ai_predicted_p_perp
message(sprintf("Ridge on %d US HC names with embeddings | lambda %.4f", nrow(d), ridge$lambda.min))

# ── 5. Picks ─────────────────────────────────────────────────────────────────
longs <- d |> arrange(mispricing) |> head(PORTFOLIO_SIZE) |>
  transmute(side = "LONG  (undervalued)", ticker, ck, mispricing = round(mispricing, 3))
shorts <- d |> arrange(desc(mispricing)) |> head(PORTFOLIO_SIZE) |>
  transmute(side = "SHORT (overvalued)", ticker, ck, mispricing = round(mispricing, 3))

picks <- bind_rows(longs, shorts)
message(sprintf("\n=== US HC Long/Short as of %s (embeddings: %s) ===", max(crsp_latest$date), EMB_QUARTER))
print(picks, n = Inf)
write_csv(picks, "picks_crsp.csv")
message("\n-> picks_crsp.csv saved.")