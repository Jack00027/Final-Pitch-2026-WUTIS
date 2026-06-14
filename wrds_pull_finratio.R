#!/usr/bin/env Rscript
# =============================================================================
# wrds_pull_finratio.R  —  P/E (and other valuation ratios) from WRDS
#                          "Financial Ratios Firm Level by WRDS"
# =============================================================================
# Produces  finratio.parquet  for WUTIS_book.R (PE_SOURCE = "finratio").
# One row per firm-month (public_date, month-end), keyed by permno/gvkey/cusip/
# ticker. WUTIS_book.R matches it to each quarter-end by CUSIP (derived from the
# ISIN) and public_date.
#
# Ratios kept (pe_exi is the default in WUTIS_book.R; the rest let you switch
# PE_VAR or build a composite later without re-pulling):
#   pe_exi    P/E, diluted, excl. extraordinary items   (standard trailing P/E)
#   pe_op_dil P/E on operating earnings (robust to one-off items)
#   pe_inc    P/E incl. extraordinary items
#   capei     Shiller cyclically-adjusted P/E
#   ps, ptb, bm, pcf, divyield, evm, peg_trailing       (other valuation ratios)
# =============================================================================

suppressPackageStartupMessages({
  library(RPostgres); library(DBI); library(dplyr); library(dbplyr); library(arrow)
})

# ── WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

# Your "Variable Descriptions" page: Library = wrdsapps, File = finratiofirm.
FR_SCHEMA <- "wrdsapps_finratio"
FR_TABLE  <- "firm_ratio"

# Pull from your embedding start through the table's coverage (ends 2025-12-31).
DATE_FROM <- as.Date("2005-01-01")
DATE_TO   <- as.Date("2025-12-31")

message(sprintf("Querying %s.%s for %s..%s ...", FR_SCHEMA, FR_TABLE, DATE_FROM, DATE_TO))

fr <- tbl(wrds, in_schema(FR_SCHEMA, FR_TABLE)) |>
  filter(public_date >= DATE_FROM, public_date <= DATE_TO) |>
  select(permno, gvkey, cusip, ticker, public_date,
         pe_exi, pe_op_dil, pe_inc, pe_op_basic, capei,
         ps, ptb, bm, pcf, divyield, evm, peg_trailing) |>
  collect()

fr <- fr |> filter(!is.na(cusip), cusip != "")
write_parquet(fr, "finratio.parquet")
dbDisconnect(wrds)

message(sprintf("finratio.parquet: %s firm-months, %s firms, %s..%s (CUSIP len: %s)",
                format(nrow(fr), big.mark = ","),
                format(dplyr::n_distinct(fr$permno), big.mark = ","),
                min(fr$public_date), max(fr$public_date),
                paste(sort(unique(nchar(fr$cusip))), collapse = "/")))
