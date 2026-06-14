#!/usr/bin/env Rscript
# =============================================================================
# WUTIS_book.R  —  Embedding-implied valuation, Health-Care long-short book
# =============================================================================
# One self-contained script. Starts from your existing artifacts:
#   * OS-BERT asset embeddings   embeddings_os/q_*.parquet
#                                (issuer_id, isin, quarter_end, dim_000..dim_0NN)
#   * Bloomberg valuation sheet  Combined_US_Stocks.xlsx
#                                (ISIN, P/E & Market Cap for Q / Q-1 / Q-2, GICS)
#
# Idea: for each quarter, regress (or kNN-comp) earnings yield E/P = 1/(P/E) on
# the OS-BERT embeddings to get an *implied* (peer-fair) yield; the residual is
# the mispricing. Long the cheapest names (trading below implied yield), short
# the richest, within US Health Care. The two sources join on ISIN.
#
# Because the sheet is a 3-quarter snapshot, this is a cross-sectional
# stock-selection deliverable (the book) plus a light signal check, not a
# long-horizon backtest.
#
# Dependencies: tidyverse, arrow, readxl  (ridge & kNN use base R only).
# Run a no-data demo (synthesises OS-BERT-format embeddings for the real ISINs):
#   DEMO <- TRUE  (below), then  Rscript WUTIS_book.R
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(readxl)
})

# ── Parameters (edit here) ───────────────────────────────────────────────────
EMB_DIR        <- "embeddings_os"            # OS-BERT output dir
VALUATION_XLSX <- "Combined_US_Stocks.xlsx"  # Bloomberg export
CACHE_DIR      <- "cache"
RESULTS_DIR    <- "results"

# The sheet's Q / Q-1 / Q-2 columns are quarter-end snapshots, pulled 2026-06-12:
# Q = 2026-03-31 (latest completed quarter), Q-1 = Dec-2025, Q-2 = Sep-2025.
# IMPORTANT: the VALUES below must equal your embedding FILE labels, q_<value>.parquet
# (the loader keys off the filename). Most files are labelled by quarter-end, but the
# Sep-2025 file is labelled 2025-10-01, so Q-2 uses that exact string.
QUARTER_MAP <- c("Q" = "2026-03-31", "Q-1" = "2025-12-31", "Q-2" = "2025-10-01")

SECTOR_FILTER  <- "Health Care"   # tradable book; NA to trade the whole market
FIT_UNIVERSE   <- "all"           # "all" = fit implied-E/P on full cross-section
                                  # then trade the sector; "sector" = fit within it
MIN_FIT_NAMES  <- 30
MIN_MARKET_CAP <- 0               # off: the >=10-institutional-owner pruning in
                                  # WUTIS_Data.r already defines the universe, and
                                  # the ISIN join enforces it, so a size floor is
                                  # redundant. Set >0 only for a sensitivity check.

L2_NORMALIZE   <- TRUE
WINSOR         <- c(0.01, 0.99)   # per-quarter E/P winsorisation

# P/E source for the valuation signal:
#   "finratio" = WRDS Financial Ratios firm-level table (clean, standardised,
#                history to 1970; run wrds_pull_finratio.R first). Recommended.
#   "bloomberg" = the P/E columns in Combined_US_Stocks.xlsx (3 quarters only).
PE_SOURCE        <- "finratio"
FINRATIO_PARQUET <- "finratio.parquet"   # output of wrds_pull_finratio.R
# Which Financial Ratios P/E column to use. pe_exi = diluted, excl. extraordinary
# items (standard). pe_op_dil = operating earnings (robust to one-off items, e.g.
# the Cigna trailing-EPS artifact). capei = Shiller CAPE. (Univariate signal:
# only this one measure drives E/P; add others later if you want a composite.)
PE_VAR           <- "pe_exi"

# Universe source:
#   "bloomberg"  = the xlsx (current quarters in QUARTER_MAP); GICS sector from
#                  the sheet. Survivorship-biased for history, but fine for the
#                  current pitch (every name is live). This is the default.
#   "embeddings" = ALL quarters in EMB_DIR — the point-in-time, survivorship-free
#                  universe (includes delisted names). Forces P/E from Financial
#                  Ratios and GICS from Compustat (GICS_PARQUET). Use this for the
#                  full historical backtest. Needs RETURN_SOURCE = "crsp", and the
#                  CRSP / Financial-Ratios pulls widened to your full date range.
UNIVERSE_SOURCE <- "embeddings"
GICS_PARQUET    <- "gics.parquet"   # gvkey -> GICS sector, from wrds_pull_gics.R

# GICS sector code -> name (Compustat gsector). Health Care = 35.
GICS_NAMES <- c("10" = "Energy", "15" = "Materials", "20" = "Industrials",
                "25" = "Consumer Discretionary", "30" = "Consumer Staples",
                "35" = "Health Care", "40" = "Financials",
                "45" = "Information Technology", "50" = "Communication Services",
                "55" = "Utilities", "60" = "Real Estate")

PROJECTION     <- "knn"         # "ridge" (GCV) or "knn"
RIDGE_LAMBDAS  <- 10 ^ seq(-2, 4, by = 0.5)
KNN_K          <- 30

TOP_N          <- 10              # names per side
USE_QUANTILE   <- FALSE
QUANTILE_FRAC  <- 0.20

# Forward-return source for the signal check / benchmark:
#   "crsp"   = real daily TOTAL returns + S&P 500 from the CRSP CIZ v2 file
#              (run wrds_pull_crsp.R first to produce PRICES_CRSP)
#   "mktcap" = crude quarter-over-quarter market-cap-ratio proxy (no extra data)
RETURN_SOURCE   <- "crsp"
PRICES_CRSP     <- "prices_crsp2.parquet" 
FILING_LAG_DAYS <- 0            # 13F public ~45 days after quarter-end; trade after
HOLDING_DAYS    <- 90            # holding window for the forward total return

SEED           <- 42
DEMO           <- FALSE           # TRUE -> synthetic embeddings for real ISINs

set.seed(SEED)

# ── Small helpers ────────────────────────────────────────────────────────────
dim_names <- function(df) grep("^dim_", names(df), value = TRUE)

winsorize <- function(x, lo, hi) {
  qs <- quantile(x, c(lo, hi), na.rm = TRUE, names = FALSE)
  pmin(pmax(x, qs[1]), qs[2])
}

zscore <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(x * 0)
  (x - mean(x, na.rm = TRUE)) / s
}

l2_normalize_rows <- function(M) {
  n <- sqrt(rowSums(M * M))
  n[n == 0] <- 1
  M / n
}

# Ridge with leave-one-out / generalised cross-validation over lambda, via SVD.
# Mirrors sklearn RidgeCV (default GCV): standardise X, choose lambda by GCV,
# return the in-sample fitted values (= implied E/P).
gcv_ridge_fitted <- function(X, y, lambdas = RIDGE_LAMBDAS) {
  n  <- nrow(X)
  Xs <- scale(X)                                   # centre + unit-variance columns
  sds <- attr(Xs, "scaled:scale")
  Xs[, sds == 0 | is.na(sds)] <- 0                 # guard constant columns
  yc  <- mean(y); yctr <- y - yc
  sv  <- svd(Xs)                                   # Xs = U diag(d) V'
  U <- sv$u; d <- sv$d
  Uty <- crossprod(U, yctr)                        # U'y_centred
  best_gcv <- Inf; best_fit <- rep(yc, n)
  for (lam in lambdas) {
    filt   <- d^2 / (d^2 + lam)                    # H eigenvalues
    fit_c  <- as.numeric(U %*% (filt * Uty))       # X(X'X+lam I)^-1 X' y_centred
    trH    <- sum(filt)
    rss    <- sum((yctr - fit_c)^2)
    gcv    <- (rss / n) / ((1 - trH / n)^2)
    if (is.finite(gcv) && gcv < best_gcv) {
      best_gcv <- gcv; best_fit <- fit_c + yc
    }
  }
  best_fit
}

# kNN comps: implied E/P = median E/P of the k nearest peers by cosine similarity
# (self excluded). Rows of V are assumed L2-normalised, so cosine = dot product.
knn_implied <- function(V, y, k = KNN_K) {
  nrw <- nrow(V)
  k   <- min(k, nrw - 1)
  S   <- tcrossprod(V)                             # n x n cosine similarity
  diag(S) <- -Inf                                  # exclude self
  vapply(seq_len(nrw), function(i) {
    nbr <- order(S[i, ], decreasing = TRUE)[seq_len(k)]
    median(y[nbr])
  }, numeric(1))
}

# ── Load OS-BERT embeddings ──────────────────────────────────────────────────
load_embeddings <- function(emb_dir = EMB_DIR) {
  files <- list.files(emb_dir, pattern = "^q_.*\\.parquet$", full.names = TRUE)
  if (length(files) == 0)
    stop(sprintf("No q_*.parquet under %s (point EMB_DIR at the OS-BERT output).",
                 emb_dir))
  # bloomberg mode: only the QUARTER_MAP quarters. embeddings mode: every quarter.
  wanted <- if (UNIVERSE_SOURCE == "embeddings") NULL else unname(QUARTER_MAP)
  out <- map_dfr(files, function(f) {
    label <- sub("^q_", "", tools::file_path_sans_ext(basename(f)))
    if (!is.null(wanted) && !(label %in% wanted)) return(NULL)
    df <- read_parquet(f)
    df$quarter_end <- label          # authoritative: key off the FILE label, so
                                     # QUARTER_MAP matches filenames regardless of any
                                     # in-file date value (handles off-by-one labels)
    dcols <- dim_names(df)
    if (L2_NORMALIZE) df[dcols] <- as.data.frame(l2_normalize_rows(as.matrix(df[dcols])))
    df[, c("quarter_end", "issuer_id", "isin", dcols)]
  })
  if (is.null(out) || nrow(out) == 0)
    stop(sprintf("No embedding files matched QUARTER_MAP (%s). Fix QUARTER_MAP.",
                 paste(wanted, collapse = ", ")))
  out <- out |> filter(!is.na(isin))
  message(sprintf("  embeddings: %s asset-quarters, %d quarter(s), %d dims",
                  format(nrow(out), big.mark = ","),
                  n_distinct(out$quarter_end), length(dim_names(out))))
  out
}

# ── P/E from WRDS Financial Ratios (firm level) ──────────────────────────────
# Map a quarter label to the Financial-Ratios public_date (a calendar quarter-end,
# month-end). Quarter-end labels pass through; a quarter-START label (e.g. your
# 2025-10-01 for the Sep-2025 quarter) snaps back to the prior quarter-end.
finratio_qend <- function(qlabel) {
  d <- as.Date(qlabel)
  if (lubridate::day(d) == 1L && lubridate::month(d) %in% c(1L, 4L, 7L, 10L))
    return(lubridate::floor_date(d, "quarter") - 1)     # prior quarter-end
  lubridate::ceiling_date(d, "quarter") - 1             # end of the containing quarter
}

# Join P/E from FINRATIO_PARQUET onto the Bloomberg universe, matched by CUSIP
# (auto 8/9-char from the table) and the quarter-end public_date. ISIN bridge:
# US ISIN = "US" + 9-char CUSIP + check digit, so the CUSIP sits at chars 3..(2+L).
attach_pe_finratio <- function(panel) {
  if (!file.exists(FINRATIO_PARQUET))
    stop(sprintf("FINRATIO_PARQUET not found (%s). Run wrds_pull_finratio.R first, or set PE_SOURCE='bloomberg'.",
                 FINRATIO_PARQUET))
  fr <- read_parquet(FINRATIO_PARQUET)
  fr$public_date <- as.Date(fr$public_date)
  fr$cusip <- toupper(trimws(as.character(fr$cusip)))
  L <- as.integer(names(sort(table(nchar(fr$cusip[fr$cusip != ""])),
                             decreasing = TRUE))[1])     # modal CUSIP length (8 or 9)
  fr <- fr |> transmute(cusip, public_date, pe = .data[[PE_VAR]])

  qmap <- tibble(quarter_end = unique(panel$quarter_end))
  qmap$fr_date <- as.Date(vapply(qmap$quarter_end,
                                 function(q) as.character(finratio_qend(q)), character(1)))

  panel |>
    mutate(cusip_key = toupper(substr(isin, 3, 2 + L))) |>
    left_join(qmap, by = "quarter_end") |>
    left_join(fr, by = c("cusip_key" = "cusip", "fr_date" = "public_date")) |>
    mutate(ep = ifelse(!is.na(pe) & pe > 0, 1 / pe, NA_real_)) |>
    select(-cusip_key, -fr_date, -pe_bbg)
}

# ── Load Bloomberg valuation sheet ───────────────────────────────────────────
load_valuation <- function(xlsx = VALUATION_XLSX) {
  cols <- c("ticker_full", "short_name", "isin", "ticker_short",
            "mcap_Q", "mcap_Qm1", "mcap_Qm2",
            "pe_Q", "pe_Qm1", "pe_Qm2",
            "gics_sector", "gics_grp", "gics_ind", "gics_subind")
  # rows 1-2 metadata, row 3 header, data from row 4 -> skip 3, supply our names
  raw <- read_excel(xlsx, skip = 3, col_names = cols)

  num <- function(x) suppressWarnings(as.numeric(x))
  suf <- c("Q" = "Q", "Q-1" = "Qm1", "Q-2" = "Qm2")    # map keys -> column suffix

  panel <- map_dfr(names(QUARTER_MAP), function(key) {
    s  <- suf[[key]]
    tibble(
      quarter_end = QUARTER_MAP[[key]],
      isin        = str_trim(as.character(raw$isin)),
      ticker      = str_trim(str_remove(as.character(raw$ticker_full), " Equity")),
      short_name  = str_trim(as.character(raw$short_name)),
      gics_sector = str_trim(as.character(raw$gics_sector)),
      market_cap  = num(raw[[paste0("mcap_", s)]]),
      pe_bbg      = num(raw[[paste0("pe_", s)]])         # Bloomberg P/E (fallback source)
    )
  }) |>
    filter(!is.na(isin), isin != "")

  # P/E source: WRDS Financial Ratios (clean, standardised) or Bloomberg columns.
  panel <- if (PE_SOURCE == "finratio") attach_pe_finratio(panel)
           else mutate(panel, pe = pe_bbg,
                       ep = ifelse(pe_bbg > 0, 1 / pe_bbg, NA_real_)) |> select(-pe_bbg)
  panel <- panel |> filter(!is.na(ep))

  if (MIN_MARKET_CAP > 0)
    panel <- panel |> filter(market_cap >= MIN_MARKET_CAP)

  panel <- panel |>                                    # one primary line per ISIN
    arrange(desc(market_cap)) |>
    distinct(isin, quarter_end, .keep_all = TRUE) |>
    group_by(quarter_end) |>
    mutate(ep = winsorize(ep, WINSOR[1], WINSOR[2])) |>
    ungroup()

  message(sprintf("  valuation: %s ticker-quarters with E/P (%d quarters; %s HC rows)",
                  format(nrow(panel), big.mark = ","),
                  n_distinct(panel$quarter_end),
                  format(sum(panel$gics_sector == "Health Care", na.rm = TRUE),
                         big.mark = ",")))
  panel
}

# ── Valuation from WRDS (embeddings universe, survivorship-free) ─────────────
# Build the valuation panel for ALL embedding (quarter, isin) pairs from the
# survivorship-free sources: P/E + gvkey + ticker from Financial Ratios (matched
# by CUSIP + quarter-end public_date), GICS sector + company name from Compustat
# (by gvkey). No Bloomberg dependency, so delisted names are kept.
build_valuation_from_embeddings <- function(embeddings) {
  if (!file.exists(FINRATIO_PARQUET))
    stop(sprintf("FINRATIO_PARQUET not found (%s). Run wrds_pull_finratio.R first.", FINRATIO_PARQUET))
  if (!file.exists(GICS_PARQUET))
    stop(sprintf("GICS_PARQUET not found (%s). Run wrds_pull_gics.R first.", GICS_PARQUET))

  fr <- read_parquet(FINRATIO_PARQUET)
  fr$public_date <- as.Date(fr$public_date)
  fr$cusip <- toupper(trimws(as.character(fr$cusip)))
  L <- as.integer(names(sort(table(nchar(fr$cusip[fr$cusip != ""])), decreasing = TRUE))[1])
  frx <- fr |> transmute(cusip, public_date,
                         gvkey = as.character(gvkey), ticker, pe = .data[[PE_VAR]])

  gics <- read_parquet(GICS_PARQUET) |>
    mutate(gvkey = as.character(gvkey),
           gics_sector = unname(GICS_NAMES[as.character(gsector)])) |>
    distinct(gvkey, .keep_all = TRUE) |>
    select(gvkey, gics_sector, conm)

  uni  <- embeddings |> distinct(quarter_end, isin)
  qmap <- tibble(quarter_end = unique(uni$quarter_end))
  qmap$fr_date <- as.Date(vapply(qmap$quarter_end,
                                 function(q) as.character(finratio_qend(q)), character(1)))

  val <- uni |>
    mutate(cusip_key = toupper(substr(isin, 3, 2 + L))) |>
    left_join(qmap, by = "quarter_end") |>
    left_join(frx,  by = c("cusip_key" = "cusip", "fr_date" = "public_date")) |>
    left_join(gics, by = "gvkey") |>
    transmute(quarter_end, isin, ticker,
              short_name  = conm,
              gics_sector,
              market_cap  = NA_real_,        # not needed (CRSP returns; equal-weight)
              pe,
              ep = ifelse(!is.na(pe) & pe > 0, 1 / pe, NA_real_)) |>
    filter(!is.na(ep)) |>
    distinct(quarter_end, isin, .keep_all = TRUE) |>
    group_by(quarter_end) |>
    mutate(ep = winsorize(ep, WINSOR[1], WINSOR[2])) |>
    ungroup()

  message(sprintf("  valuation (embeddings universe): %s firm-quarters with E/P, %d quarters; %s HC rows",
                  format(nrow(val), big.mark = ","), n_distinct(val$quarter_end),
                  format(sum(val$gics_sector == "Health Care", na.rm = TRUE), big.mark = ",")))
  val
}

# ── Mispricing signal ────────────────────────────────────────────────────────
build_signal <- function(embeddings, valuation) {
  dcols  <- dim_names(embeddings)
  merged <- inner_join(embeddings, valuation, by = c("quarter_end", "isin"))

  out <- list()
  for (q in sort(unique(merged$quarter_end))) {
    pq <- merged |> filter(quarter_end == q)
    fit <- if (FIT_UNIVERSE == "sector" && !is.na(SECTOR_FILTER)) {
      pq |> filter(gics_sector == SECTOR_FILTER)
    } else pq
    if (nrow(fit) < MIN_FIT_NAMES) {
      message(sprintf("  %s: only %d fitting names (<%d), skipped",
                      q, nrow(fit), MIN_FIT_NAMES)); next
    }
    X <- as.matrix(fit[, dcols]); y <- fit$ep
    implied <- if (PROJECTION == "ridge") gcv_ridge_fitted(X, y)
               else if (PROJECTION == "knn") knn_implied(X, y)
               else stop(sprintf("Unknown PROJECTION %s", PROJECTION))
    fit <- fit |>
      mutate(implied_ep = implied,
             ep_resid   = ep - implied_ep,          # >0 => cheaper than peers
             signal     = zscore(-ep_resid))         # high => overvalued => short
    out[[q]] <- fit
    hc <- if (!is.na(SECTOR_FILTER)) sum(fit$gics_sector == SECTOR_FILTER, na.rm = TRUE) else nrow(fit)
    message(sprintf("  %s: fit on %s names (%s); %s in target sector",
                    q, format(nrow(fit), big.mark = ","), PROJECTION,
                    format(hc, big.mark = ",")))
  }
  if (length(out) == 0)
    stop("No quarter produced a signal — check the ISIN join and QUARTER_MAP.")
  bind_rows(out)
}

# ── Build the long-short book ────────────────────────────────────────────────
book_cols <- c("quarter_end", "side", "rank", "ticker", "short_name",
               "gics_sector", "market_cap", "pe", "ep", "implied_ep",
               "ep_resid", "signal")

select_legs <- function(pq) {
  s <- pq |> arrange(signal)                          # cheap first
  if (USE_QUANTILE) {
    lo <- quantile(s$signal, QUANTILE_FRAC,     names = FALSE)
    hi <- quantile(s$signal, 1 - QUANTILE_FRAC, names = FALSE)
    longs  <- s |> filter(signal <= lo)
    shorts <- s |> filter(signal >= hi) |> arrange(desc(signal))
  } else {
    n <- min(TOP_N, nrow(s) %/% 2)
    longs  <- s |> slice_head(n = n)
    shorts <- s |> slice_tail(n = n) |> arrange(desc(signal))
  }
  bind_rows(
    longs  |> mutate(side = "LONG",  rank = row_number()),
    shorts |> mutate(side = "SHORT", rank = row_number())
  )
}

build_book <- function(signal_panel) {
  df <- signal_panel
  if (!is.na(SECTOR_FILTER)) df <- df |> filter(gics_sector == SECTOR_FILTER)
  books <- list()
  for (q in sort(unique(df$quarter_end))) {
    pq <- df |> filter(quarter_end == q)
    if (nrow(pq) < 4) { message(sprintf("  %s: too few names for a book", q)); next }
    bq <- select_legs(pq)
    books[[q]] <- bq[, intersect(book_cols, names(bq))]
    message(sprintf("  %s: %d long / %d short",
                    q, sum(bq$side == "LONG"), sum(bq$side == "SHORT")))
  }
  if (length(books) == 0) stop("No book produced — check SECTOR_FILTER / universe.")
  bind_rows(books)
}

# ── Forward returns ──────────────────────────────────────────────────────────
# (A) crude market-cap-ratio proxy: quarter t -> t+1 change in market cap.
forward_returns_mktcap <- function(valuation) {
  valuation |>
    select(quarter_end, isin, market_cap) |>
    arrange(isin, quarter_end) |>
    group_by(isin) |>
    mutate(fwd_ret = lead(market_cap) / market_cap - 1) |>
    ungroup() |>
    filter(!is.na(fwd_ret)) |>
    select(quarter_end, isin, fwd_ret)
}

# (B) real forward TOTAL return from CRSP CIZ daily, plus the S&P 500 over the
# same window. For each name: compound dlyret over [qe + lag, qe + lag + horizon].
# ISIN -> CRSP bridge is length-agnostic: matches the prices file's CUSIP (8-char
# `cusip` from dsf_v2, or 9-char `cusip9` from the merged view) to substr(isin,3,*).
forward_returns_crsp <- function(valuation, prices_path = PRICES_CRSP,
                                 lag = FILING_LAG_DAYS, horizon = HOLDING_DAYS) {
  if (!file.exists(prices_path))
    stop(sprintf("PRICES_CRSP not found (%s). Run wrds_pull_crsp.R first, or set RETURN_SOURCE='mktcap'.",
                 prices_path))
  prices <- read_parquet(prices_path)
  prices$dlycaldt <- as.Date(prices$dlycaldt)

  qs <- sort(unique(valuation$quarter_end))

  # S&P 500 window return per quarter — only if the prices file carries sprtrn.
  # (If you didn't pull a benchmark, this is skipped and the book just omits the
  # S&P columns; signal_check handles their absence.)
  has_spx  <- "sprtrn" %in% names(prices)
  spx_rows <- if (has_spx) {
    spx <- prices |> distinct(dlycaldt, sprtrn) |> arrange(dlycaldt)
    map_dfr(qs, function(q) {
      entry <- as.Date(q) + lag; exit <- entry + horizon
      w <- spx |> filter(dlycaldt >= entry, dlycaldt <= exit, !is.na(sprtrn))
      tibble(quarter_end = q,
             spx_ret = if (nrow(w)) prod(1 + w$sprtrn) - 1 else NA_real_)
    })
  } else NULL

  # CUSIP bridge, length-agnostic: the prices file may carry a 9-char `cusip9`
  # (merged view) or an 8-char `cusip` (raw dsf_v2). Standardise to `ck` and derive
  # the matching ISIN substring from its modal length.
  ccol <- if ("cusip9" %in% names(prices)) "cusip9" else "cusip"
  prices$ck <- toupper(trimws(as.character(prices[[ccol]])))
  L <- as.integer(names(sort(table(nchar(prices$ck[prices$ck != ""])),
                             decreasing = TRUE))[1])

  uni <- valuation |> distinct(quarter_end, isin) |>
    mutate(ck = toupper(substr(isin, 3, 2 + L)))
  px <- prices |>
    filter(ck %in% unique(uni$ck),
           dlycaldt >= (min(as.Date(qs)) + lag),
           dlycaldt <= (max(as.Date(qs)) + lag + horizon),
           !is.na(dlyret)) |>
    select(ck, dlycaldt, dlyret)

  # Per-quarter compounding. A single inner_join(uni, px, by = "ck") forms the
  # cross-product of every (name, quarter) against EVERY daily price row that name
  # has over the whole sample, pruning by date only afterwards — on a multi-year
  # panel that is hundreds of millions of rows and exhausts memory. Scoping each
  # quarter to its own [entry, exit] window keeps the join at one row per name.
  stock <- map_dfr(qs, function(q) {
    entry <- as.Date(q) + lag; exit <- entry + horizon
    p_q <- px |> filter(dlycaldt >= entry, dlycaldt <= exit)
    if (nrow(p_q) == 0L) return(NULL)
    u_q <- uni |> filter(quarter_end == q) |> distinct(isin, ck)
    p_q |>
      group_by(ck) |>
      summarise(fwd_ret = prod(1 + dlyret) - 1, n_days = n(), .groups = "drop") |>
      inner_join(u_q, by = "ck") |>
      transmute(quarter_end = q, isin, fwd_ret, n_days)
  })

  if (is.null(spx_rows)) return(stock)
  stock |>
    left_join(spx_rows, by = "quarter_end") |>
    mutate(excess_ret = fwd_ret - spx_ret)
}

signal_check <- function(signal_panel, fwd) {
  df <- signal_panel
  if (!is.na(SECTOR_FILTER)) df <- df |> filter(gics_sector == SECTOR_FILTER)
  j <- inner_join(df, fwd, by = c("quarter_end", "isin"))
  has_spx <- "spx_ret" %in% names(j)
  rows <- list()
  for (q in sort(unique(j$quarter_end))) {
    g <- j |> filter(quarter_end == q, !is.na(fwd_ret))
    if (nrow(g) < 2 * TOP_N) next
    ct <- suppressWarnings(cor.test(g$signal, g$fwd_ret, method = "spearman"))
    s  <- g |> arrange(signal); n <- min(TOP_N, nrow(s) %/% 2)
    lr <- mean(head(s$fwd_ret, n)); sr <- mean(tail(s$fwd_ret, n))
    row <- tibble(quarter_end = q, n = nrow(g),
                  spearman = unname(ct$estimate), p_value = ct$p.value,
                  long_ret = lr, short_ret = sr, ls_spread = lr - sr)
    if (has_spx) {
      spx_q <- g$spx_ret[1]                       # same for all names in the quarter
      row <- row |> mutate(spx_ret = spx_q,
                           long_excess  = lr - spx_q,
                           short_excess = sr - spx_q)
    }
    rows[[q]] <- row
  }
  if (length(rows) == 0) return(tibble())
  bind_rows(rows)
}

# ── Demo: synthetic OS-BERT-format embeddings for the real ISINs ─────────────
# E/P is partially encoded along one direction so the ridge has real explanatory
# content and the residual is the unexplained part. Forward returns are NOT
# encoded, so the demo's signal check is ~random by construction.
write_demo_embeddings <- function(valuation, d = 64L, emb_dir = EMB_DIR) {
  dir.create(emb_dir, showWarnings = FALSE, recursive = TRUE)
  for (q in unique(valuation$quarter_end)) {
    g <- valuation |> filter(quarter_end == q) |> distinct(isin, .keep_all = TRUE)
    z <- as.numeric(scale(g$ep)); z[is.na(z)] <- 0
    u <- rnorm(d); u <- u / sqrt(sum(u^2))            # E/P direction
    V <- outer(z, u) + 0.6 * matrix(rnorm(nrow(g) * d), nrow(g), d)
    colnames(V) <- sprintf("dim_%03d", 0:(d - 1))
    emb <- bind_cols(
      tibble(issuer_id   = sprintf("E%06d", seq_len(nrow(g)) - 1L),
             isin        = g$isin,
             quarter_end = q),
      as_tibble(V))
    write_parquet(emb, file.path(emb_dir, sprintf("q_%s.parquet", q)))
  }
  message(sprintf("  [demo] wrote synthetic embeddings for %d quarter(s) -> %s",
                  n_distinct(valuation$quarter_end), emb_dir))
}

# ── Main ─────────────────────────────────────────────────────────────────────
main <- function() {
  dir.create(CACHE_DIR,   showWarnings = FALSE, recursive = TRUE)
  dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

  if (UNIVERSE_SOURCE == "embeddings") {
    message("[1/4] OS-BERT asset embeddings (universe = all quarters)")
    embeddings <- load_embeddings()
    message("[2/4] valuation from Financial Ratios + Compustat GICS (survivorship-free)")
    valuation  <- build_valuation_from_embeddings(embeddings)
  } else {
    message("[1/4] valuation panel (Bloomberg xlsx)")
    valuation <- load_valuation()
    if (DEMO) write_demo_embeddings(valuation)
    message("[2/4] OS-BERT asset embeddings")
    embeddings <- load_embeddings()
  }

  message("[3/4] embedding-implied E/P -> mispricing signal")
  signal <- build_signal(embeddings, valuation)

  message("[4/4] long-short book + signal check")
  book <- build_book(signal)
  fwd  <- if (RETURN_SOURCE == "crsp") forward_returns_crsp(valuation)
          else forward_returns_mktcap(valuation)
  chk  <- signal_check(signal, fwd)

  write_parquet(signal, file.path(CACHE_DIR, "signal_panel.parquet"))
  write_parquet(book,   file.path(RESULTS_DIR, "book_all.parquet"))
  for (q in unique(book$quarter_end))
    write_csv(book |> filter(quarter_end == q),
              file.path(RESULTS_DIR, sprintf("book_%s.csv", q)))
  if (nrow(chk) > 0) write_csv(chk, file.path(RESULTS_DIR, "signal_check.csv"))

  for (q in sort(unique(book$quarter_end))) {
    cat(sprintf("\n===== %s book — %s (%s) =====\n",
                ifelse(is.na(SECTOR_FILTER), "ALL", SECTOR_FILTER), q, PROJECTION))
    book |>
      filter(quarter_end == q) |>
      transmute(side, rank, ticker, short_name,
                ep = round(ep, 4), implied_ep = round(implied_ep, 4),
                signal = round(signal, 4)) |>
      as.data.frame() |> print(row.names = FALSE)
  }

  if (nrow(chk) > 0) {
    src <- if (RETURN_SOURCE == "crsp") "CRSP total returns vs S&P 500"
           else "market-cap-derived fwd returns"
    cat(sprintf("\n----- signal check (%s) -----\n", src))
    chk |> mutate(across(where(is.numeric), ~ round(.x, 4))) |>
      as.data.frame() |> print(row.names = FALSE)
    if (DEMO) cat("(demo: embeddings synthetic -> these numbers are ~random)\n")
  }
  cat(sprintf("\nbook + check -> %s\n", RESULTS_DIR))
  invisible(list(book = book, check = chk))
}

# Run the pipeline. (Previously guarded by sys.nframe() == 0L, which fires under
# `Rscript file.R` but NOT under `source(file)` — source() adds a stack frame, so
# sys.nframe() is not 0. Call main() unconditionally so source() runs it too.)
main()