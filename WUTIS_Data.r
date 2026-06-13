# Data pipeline for asset embeddings from investor holdings.

library(tidyverse)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)


# ── Parameters ─────────────────────────────────────────────────
START_QUARTER  <- ymd("2005-01-01")
END_QUARTER    <- ymd("2026-03-31")
MIN_STOCKS     <- 10     # per investor-quarter
MIN_INVESTORS  <- 10     # per stock-quarter
MAX_TOP1_PCT   <- 0.70   # max single-holding weight
CONTEXT_WINDOW <- 62     # OS-BERT max sequence length

out_dir <- "data_wutis"     # data directory

# ── Test mode ─────────────────────────────────────────────────
TEST_MODE <- FALSE   # set to FALSE for full run; TRUE runs 2 quarters for quick testing

# ── Overqrite existing files? ───────────────────────────────────────
OVERWRITE <- TRUE   # set to FALSE to skip quarters with existing output files

if (TEST_MODE) {
  START_QUARTER <- ymd("2025-10-01")
  END_QUARTER   <- ymd("2026-03-31")
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

  if (file.exists(out_file) && !OVERWRITE) {
    message("   already exists, skipping")
    next
  }

  t0 <- Sys.time()

  # Precompute date bounds
  q_13f_lo  <- qe - 7
  q_13f_hi  <- qe + 7


  # ── 1. 13F holdings (collected at security grain so we can keep ISIN)

  raw_13f <- tbl_13f |>
    filter(report_date >= q_13f_lo,
           report_date <= q_13f_hi,
           adj_mv > 0) |>
    inner_join(sec_map_lazy, by = "fsym_id") |>
    select(investor_id = factset_entity_id, issuer_id, report_date, adj_mv, isin) |>
    collect() |>
    mutate(report_date = as.Date(report_date), quarter_end = qe)

  # Holdings aggregated to (investor, issuer): the asset is the issuer entity.
  holdings_13f <- raw_13f |>
    group_by(investor_id, quarter_end, issuer_id) |>
    summarise(adj_mv = sum(as.numeric(adj_mv)), investor_type = "INST",
              .groups = "drop")

  # Representative ISIN per issuer-quarter: the security with the largest value.
  # (An issuer entity can span several securities/share classes, each with its
  #  own ISIN, so we pick the primary one to keep exactly one ISIN per asset.)
  isin_map <- raw_13f |>
    group_by(quarter_end, issuer_id, isin) |>
    summarise(mv = sum(as.numeric(adj_mv)), .groups = "drop_last") |>
    slice_max(mv, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(quarter_end, issuer_id, isin)


  # ── 2. Holdings (13F only)

  holdings <- holdings_13f
  rm(holdings_13f, raw_13f)

  if (nrow(holdings) == 0) {
    message("   no holdings, skipping")
    next
  }


  # ── 3. Concentration filter + bipartite pruning
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


  # ── 5. Ownership → investor token sequences ──
  #    Group by ASSET; order its investors by descending ownership share.
  #    Within an asset-quarter every investor holds at the same price, so
  #    descending adj_mv == descending ownership share.

  sequences <- holdings |>
    group_by(issuer_id, quarter_end) |>
    mutate(s = adj_mv / sum(adj_mv)) |>
    arrange(desc(s), .by_group = TRUE) |>
    summarise(tokens      = list(as.character(investor_id)),
              n_investors = n(),
              .groups     = "drop") |>
    mutate(chunks = map2(tokens, n_investors, chunk_seq)) |>
    unnest(chunks) |>
    mutate(tokens   = chunks,
           n_tokens = map_int(tokens, length)) |>
    left_join(isin_map, by = c("quarter_end", "issuer_id")) |>
    select(quarter_end, issuer_id, isin,
           tokens, n_tokens, n_investors_full = n_investors)


  # ── 6. Save and free memory ──
  write_parquet(sequences, out_file)

  elapsed_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  message(sprintf("   %s holdings, %s assets, %s sequences (%.1f min)",
                  format(nrow(holdings), big.mark = ","),
                  format(n_distinct(holdings$issuer_id), big.mark = ","),
                  format(nrow(sequences), big.mark = ","),
                  elapsed_min))

  rm(holdings, sequences, isin_map); gc(verbose = FALSE)
}

dbDisconnect(wrds)
message("Done.")