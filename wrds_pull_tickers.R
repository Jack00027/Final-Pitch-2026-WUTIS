#!/usr/bin/env Rscript
# =============================================================================
# wrds_pull_tickers.R  —  CUSIP -> Yahoo ticker map for live_paper_trade.R
# =============================================================================
# Pulls current tickers from Compustat (comp.security), keys them to the 8-char
# CUSIP used everywhere in the pipeline, fixes share-class suffixes for Yahoo,
# and writes ticker_map.csv (columns: ck, ticker).
#
#   Rscript wrds_pull_tickers.R
# =============================================================================

library(RPostgres)
library(DBI)
library(dplyr)
library(dbplyr)
library(readr)

# ── WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

# ── 2. Pull every security that has a ticker ─────────────────────────────────
# comp.security is one row per issue, carrying the 9-char cusip and ticker (tic).
security <- tbl(wrds, in_schema("comp", "security")) |>
  select(gvkey, cusip, tic) |>
  filter(!is.na(tic), tic != "", !is.na(cusip)) |>
  collect()

dbDisconnect(wrds)

# ── 3. Build the map ─────────────────────────────────────────────────────────
# ck     = first 8 chars of the cusip (issuer + issue, no check digit), the key
#          used across the pipeline.
# ticker = tic with class-share dots turned into dashes ("BRK.A" -> "BRK-A"),
#          which is how Yahoo writes them.
ticker_map <- security |>
  mutate(ck     = toupper(substr(cusip, 1, 8)),
         ticker = gsub(".", "-", tic, fixed = TRUE)) |>
  filter(nchar(ck) == 8) |>
  distinct(ck, .keep_all = TRUE) |>
  select(ck, ticker)

# ── 4. Save ──────────────────────────────────────────────────────────────────
write_csv(ticker_map, "ticker_map.csv")
message(sprintf("ticker_map.csv: %d CUSIP -> ticker rows", nrow(ticker_map)))
