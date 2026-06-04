# Data pipeline for asset embeddings from investor holdings.

library(tidyverse)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)


# ── Parameters ─────────────────────────────────────────────────
START_QUARTER  <- ymd("2005-01-01")
END_QUARTER    <- ymd("2025-12-31")
MIN_STOCKS     <- 20     # per investor-quarter
MIN_INVESTORS  <- 20     # per stock-quarter
MAX_TOP1_PCT   <- 0.75   # max single-holding weight
REPORT_WINDOW  <- 5      # days before quarter-end for fund reports
CONTEXT_WINDOW <- 62     # OS-BERT max sequence length

out_dir <- "data_wutis"     # data directory

# ── Test mode ─────────────────────────────────────────────────
TEST_MODE <- TRUE   # set to FALSE for full run; TRUE runs 2 quarters for quick testing

if (TEST_MODE) {
  START_QUARTER <- ymd("2019-07-01")
  END_QUARTER   <- ymd("2019-12-31")
  out_dir <- "data_wutis/test"
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)


# ── WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

tbl_13f      <- tbl(wrds, in_schema("factset_own", "wrds_own_13f"))
tbl_fund     <- tbl(wrds, in_schema("factset_own", "wrds_own_fund"))
tbl_ent_fund <- tbl(wrds, in_schema("factset_own", "own_ent_funds"))
tbl_sec_map  <- tbl(wrds, in_schema("factset_own", "own_sec_entity_eq"))

quarter_ends <- seq.Date(
  ceiling_date(START_QUARTER, "quarter") - days(1),
  ceiling_date(END_QUARTER,   "quarter") - days(1),
  by = "quarter"
)

# Lazy reference for issuer mapping
sec_map_lazy <- tbl_sec_map |>
  filter(!is.na(factset_entity_id)) |>
  select(fsym_id, issuer_id = factset_entity_id)

fund_map <- tbl_ent_fund |> filter(!is.na(fund_type)) |>
                            select(factset_fund_id, fund_type)


# ── Sequence chunking helper ──────────────────────────────────
chunk_seq <- function(tokens, n, ctx = CONTEXT_WINDOW) {
  if (n <= ctx) return(list(tokens))
  k <- ceiling(n / ctx)
  split(tokens, ceiling(seq_along(tokens) / ceiling(n / k)))
}


# ── Per-quarter loop ──────────────────────────────────────────
message(length(quarter_ends), " quarters from ",
        first(quarter_ends), " to ", last(quarter_ends))

for (i in seq_along(quarter_ends)) {
  qe       <- quarter_ends[i]
  q_label  <- as.character(qe)
  out_file <- file.path(out_dir, sprintf("q_%s.parquet", q_label))

  message("[", i, "/", length(quarter_ends), "] ", q_label)

  if (file.exists(out_file)) {
    message("   already exists, skipping")
    next
  }

  t0 <- Sys.time()

# Precompute date bounds
  q_13f_lo  <- qe - 7
  q_13f_hi  <- qe + 7
  q_fund_lo <- qe - REPORT_WINDOW
  q_fund_hi <- qe + REPORT_WINDOW


  # ── 1. 13F holdings 
  holdings_13f <- tbl_13f |>
    filter(entity_sub_type == "HF",
           report_date >= q_13f_lo,
           report_date <= q_13f_hi,
           adj_mv > 0) |>
    inner_join(sec_map_lazy, by = "fsym_id") |>
    select(investor_id = factset_entity_id, issuer_id, report_date, adj_mv) |>
    collect() |>
    mutate(report_date = as.Date(report_date), quarter_end = qe) |>
    group_by(investor_id, quarter_end, issuer_id) |>
    summarise(adj_mv = sum(as.numeric(adj_mv)), investor_type = "HF",
              .groups = "drop")


  # ── 2. Fund holdings
  holdings_fund <- tbl_fund |>
    inner_join(sec_map_lazy, by = "fsym_id") |>
    inner_join(fund_map, by = "factset_fund_id") |>
    filter(fund_type %in% c("OEF", "ETF", "CEF", "VAR"),
           report_date >= q_fund_lo,
           report_date <= q_fund_hi,
           adj_mv > 0) |>
    select(investor_id = factset_fund_id, report_date, adj_mv,
           investor_type = fund_type, issuer_id) |>
    collect() |>
    filter(!is.na(issuer_id)) |>
    mutate(report_date = as.Date(report_date), quarter_end = qe) |>
    arrange(investor_id, issuer_id, desc(report_date)) |>
    group_by(investor_id, issuer_id) |>
    slice_head(n = 1) |>
    ungroup() |>
    group_by(investor_id, investor_type, quarter_end, issuer_id) |>
    summarise(adj_mv = sum(as.numeric(adj_mv)), .groups = "drop")


  # ── 3. Combine ── 
  holdings <- bind_rows(holdings_13f, holdings_fund)
  rm(holdings_13f, holdings_fund)

  if (nrow(holdings) == 0) {
    message("   no holdings, skipping")
    next
  }


  # ── 4. Concentration filter + bipartite pruning 
  holdings <- holdings |>
    group_by(investor_id, quarter_end) |>
    filter(max(adj_mv) / sum(adj_mv) <= MAX_TOP1_PCT) |>
    ungroup()

  repeat {
    n0 <- nrow(holdings)
    holdings <- holdings |>
      group_by(investor_id, quarter_end) |> filter(n() >= MIN_STOCKS) |> ungroup() |>
      group_by(issuer_id, quarter_end)   |> filter(n() >= MIN_INVESTORS) |> ungroup()
    if (nrow(holdings) == n0) break
  }

  if (nrow(holdings) == 0) {
    message("   nothing survived pruning, skipping")
    next
  }



  