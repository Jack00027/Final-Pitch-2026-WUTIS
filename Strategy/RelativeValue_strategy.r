# =============================================================================
# Relative Valuation Benchmark  (Gabaix-Koijen-Richmond-Yogo, Section 4.1)
# =============================================================================

library(tidyverse)
library(arrow)
library(lubridate)

# ── Choices ──────────────────────────────────────────────────────────────────
VALUATION <- "market_to_book"   # "market_to_book" -> regress log ME on log book equity
                                # "price_earnings" -> regress log ME on log earnings

LIQUIDITY_SCREEN <- TRUE        # TRUE  = estimate on liquid firms only
                                # FALSE = full cross-section (for the paper's R^2)
LIQ_PCT          <- 0.50        # fraction of each quarter to keep, by market cap

predictions_list <- list()

# ── Inputs ───────────────────────────────────────────────────────────────────
EMB_DIR             <- "embeddings_os"               # OS-BERT q_*.parquet (defines the firm universe)
FUNDAMENTALS_MERGED <- "fundamentals_merged.parquet"

# Standardise every id to the 8-char CUSIP used across the project.
ck_from_isin  <- function(x) toupper(substr(x, 3, 10))   # US ISIN chars 3..10

# Canonical quarter-end: 07-01 -> 06-30, 10-01 -> 09-30, others unchanged.
canon_q <- function(d) ceiling_date(as.Date(d) - days(1), "quarter") - days(1)

# Define the winsorization function
winsorize <- function(x, pct = 0.01) {
  bounds <- quantile(x, probs = c(pct, 1 - pct), na.rm = TRUE)
  case_when(
    x < bounds[1] ~ bounds[1],
    x > bounds[2] ~ bounds[2],
    TRUE ~ x
  )
}

# ── Step 1. Universe: every (quarter, US firm) that has an embedding ─────────
message("Loading embedding universe (US only)...")
emb_files <- list.files(EMB_DIR, pattern = "^q_.*\\.parquet$", full.names = TRUE)
emb_index <- tibble(
  file     = emb_files,
  emb_date = as.Date(sub("^q_", "", tools::file_path_sans_ext(basename(emb_files))))
) |>
  mutate(quarter = canon_q(emb_date))

universe_list <- list()
for (i in seq_len(nrow(emb_index))) {
  isins <- read_parquet(emb_index$file[i], col_select = "isin")$isin
  isins <- isins[substr(isins, 1, 2) == "US"]            # US stocks only
  universe_list[[i]] <- tibble(quarter = emb_index$quarter[i], ck = ck_from_isin(isins))
}
universe <- bind_rows(universe_list) |> filter(ck != "", !is.na(ck)) |> distinct(quarter, ck)
message(sprintf("universe: %d firm-quarters across %d quarters",
                nrow(universe), n_distinct(universe$quarter)))

# ── Step 2. Load the Cleaned Fundamentals Panel ──────────────────────────────
message("Loading merged fundamentals...")
panel <- read_parquet(FUNDAMENTALS_MERGED) |>
  mutate(
    # Lag the fundamentals by 1 quarter (3 months) to avoid look-ahead bias
    quarter = quarter %m+% months(3),
    quarter = ceiling_date(quarter, "quarter") - days(1)
  ) |>
  # Keep only the rows that exist in our embeddings universe
  inner_join(universe, by = c("quarter", "ck")) |>
  # Ensure clean numeric formats just in case
  mutate(
    market_equity = as.numeric(market_equity),
    book_equity   = as.numeric(book_equity),
    earnings      = as.numeric(earnings)
  )

# ── Step 3. Liquidity screen ─────────────────────────────────────────────────
if (LIQUIDITY_SCREEN) {
  panel <- panel |>
    group_by(quarter) |>
    filter(market_equity >= quantile(market_equity, 1 - LIQ_PCT, na.rm = TRUE)) |>
    ungroup()
  message(sprintf("liquidity screen: top %d%% by market cap -> %d firm-quarters",
                  round(100 * LIQ_PCT), nrow(panel)))
}

# ── Step 4. Pick the denominator (the if-switch) ─────────────────────────────
if (VALUATION == "market_to_book") {
  panel <- panel |> mutate(denominator = book_equity)
} else if (VALUATION == "price_earnings") {
  panel <- panel |> mutate(denominator = earnings)
} else {
  stop("VALUATION must be 'market_to_book' or 'price_earnings'")
}

# logs need positive values (this drops negative book equity / negative earnings and missing ME)
panel <- panel |>
  filter(market_equity > 0, denominator > 0, !is.na(market_equity), !is.na(denominator)) |>
  mutate(log_me = log(market_equity), log_denom = log(denominator))

# ── Step 5. Per-quarter cross-sectional regression -> valuation residual ─────
message("Running cross-sectional regressions...")
valuation_list <- list()

for (q in sort(unique(panel$quarter))) {
  d <- panel |> filter(quarter == q)
  if (nrow(d) < 10) next                          # skip quarters with too few firms

  d$log_me <- winsorize(d$log_me, 0.01)
  d$log_denom <- winsorize(d$log_denom, 0.01)

  fit <- lm(log_me ~ log_denom, data = d)

  d$gamma     <- coef(fit)[["log_denom"]]
  d$alpha     <- coef(fit)[["(Intercept)"]]
  d$valuation <- as.numeric(residuals(fit))

  valuation_list[[as.character(q)]] <- d
}

valuation <- bind_rows(valuation_list)

# ── Step 6. Inspect and save ─────────────────────────────────────────────────
message(sprintf("%s: %d firm-quarters | mean gamma %.2f",
                VALUATION, nrow(valuation), mean(valuation$gamma)))
out <- valuation |> select(quarter, ck, log_me, log_denom, gamma, alpha, valuation)

write_parquet(out, "valuation_metric.parquet")
message("-> valuation_metric.parquet successfully saved.")


# =============================================================================
# Step 7. Assign firms to K cross-fitting folds (firm-level, fixed across time)
# =============================================================================
library(glmnet)

K_FOLDS <- 5
message(sprintf("Assigning firms to %d cross-fitting folds...", K_FOLDS))

valuation <- read_parquet("valuation_metric.parquet")

unique_firms <- unique(valuation$ck)
set.seed(42)
fold_of_firm <- tibble(ck = unique_firms,
                       fold = sample(rep_len(1:K_FOLDS, length(unique_firms))))

valuation <- valuation |> inner_join(fold_of_firm, by = "ck")
message(sprintf("%d firms across %d folds (~%d firms/fold)",
                length(unique_firms), K_FOLDS, round(length(unique_firms) / K_FOLDS)))

# =============================================================================
# Step 8. Per-quarter K-fold CROSS-FITTED ridge (every firm predicted out-of-fold)
# =============================================================================
message("Running cross-fitted Ridge per quarter (this is ~K times slower)...")

quarters <- as.character(sort(unique(valuation$quarter)))
results_list     <- list()
predictions_list <- list()

l2_normalize <- function(mat) {
  norms <- sqrt(rowSums(mat^2))
  norms[norms == 0] <- 1
  return(mat / norms)
}

for (q in quarters) {

  emb_file <- emb_index$file[emb_index$quarter == as.Date(q)]
  if (length(emb_file) == 0 || !file.exists(emb_file[1])) next

  emb_data <- read_parquet(emb_file[1]) |>
    filter(substr(isin, 1, 2) == "US") |>
    mutate(ck = toupper(substr(isin, 3, 10))) |>
    select(-isin)

  d <- valuation |>
    filter(quarter == as.Date(q)) |>
    inner_join(emb_data, by = "ck")

  if (nrow(d) < 50) next

  dim_cols <- grep("^dim_", names(d), value = TRUE)
  X_all <- l2_normalize({ m <- as.matrix(d[, dim_cols]); mode(m) <- "numeric"; m })
  y_all <- d$valuation

  # Out-of-fold predictions: train on the other folds, predict this fold.
  preds_oof <- rep(NA_real_, nrow(d))
  for (k in sort(unique(d$fold))) {
    tr <- which(d$fold != k)
    te <- which(d$fold == k)
    if (length(tr) < 50 || length(te) == 0) next
    cv_fit <- cv.glmnet(X_all[tr, , drop = FALSE], y_all[tr],
                        alpha = 0, nfolds = 10, standardize = FALSE)
    preds_oof[te] <- as.numeric(predict(cv_fit, s = "lambda.min",
                                        newx = X_all[te, , drop = FALSE]))
  }

  ok <- !is.na(preds_oof)
  if (sum(ok) < 10) next

  predictions_list[[as.character(q)]] <- tibble(
    quarter = as.Date(q),
    ck = d$ck[ok],
    actual_p_perp = y_all[ok],
    ai_predicted_p_perp = preds_oof[ok]
  )

  # OOS variance components on the pooled out-of-fold predictions (full universe)
  results_list[[as.character(q)]] <- tibble(
    quarter    = as.Date(q),
    var_error  = var(y_all[ok] - preds_oof[ok]),
    var_target = var(y_all[ok]),
    n          = sum(ok)
  )
}

ai_predictions <- bind_rows(predictions_list)

# =============================================================================
# Step 9. Compute Final Out-of-Sample R^2 (full-universe cross-validated)
# =============================================================================
message("Calculating final Out-Of-Sample R^2...")

results_df <- bind_rows(results_list)

# We average the variance ratios across all quarters (T), then subtract from 1.
RV_metric <- 1 - mean(results_df$var_error / results_df$var_target)

message(sprintf("==========================================="))
message(sprintf("Relative Valuation Benchmark OOS R^2: %.4f", RV_metric))
message(sprintf("ai_predictions: %d firm-quarters across %d names (full universe)",
                nrow(ai_predictions), n_distinct(ai_predictions$ck)))
message(sprintf("==========================================="))

write_parquet(results_df, "ridge_oos_results.parquet")
write_parquet(ai_predictions, "ai_predictions.parquet")