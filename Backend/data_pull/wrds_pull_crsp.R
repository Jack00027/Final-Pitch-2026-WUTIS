# =============================================================================
# wrds_pull_crsp.R  —  daily TOTAL returns + S&P 500 from the CRSP CIZ v2 file
# =============================================================================
# Produces  prices_crsp.parquet  for WUTIS_book.R (RETURN_SOURCE = "crsp").
# Pulls the raw CIZ daily stock file (crsp_a_stock.dsf_v2). No benchmark: the S&P
# 500 (sprtrn) is not in this file and the CIZ index tables aren't in this license,
# so the book treats the S&P column as optional. Add your own daily series later.
#
# Fields kept:
#   permno, dlycaldt, cusip, hdrcusip, ticker, siccd   (id / date)
#   dlyret   = daily TOTAL return (dividends included)  <- the real forward return
#   dlyretx  = daily price return (ex-dividends)
#   dlyprc, dlyclose, dlycap, shrout              (price / size, for reference)
#   (no sprtrn / benchmark — not pulled here)
#
# The ISIN -> CRSP bridge lives in WUTIS_book.R, length-agnostic: US ISIN = "US" +
# 9-char CUSIP + check digit, so the 8-char `cusip` = substr(isin, 3, 10) and the
# 9-char form = substr(isin, 3, 11); the book auto-detects which the file carries.
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

# ── Table ────────────────────────────────────────────────────────────────────
# Raw CIZ daily stock file crsp_a_stock.dsf_v2 (8-char `cusip`; no cusip9). It does
# not carry sprtrn (the S&P 500 return), and the CIZ index tables aren't in this
# license, so no benchmark is pulled here — add your own daily S&P series later.
CRSP_SCHEMA <- "crsp_a_stock"
CRSP_TABLE  <- "dsf_v2"

# ── Date window: cover every forward holding window you need ─────────────────
# entry = quarter_end + 45d, exit = entry + 90d. The table tops out around the
# last monthly update (~2026-04-30 in your snapshot).
#   - Current pitch (UNIVERSE_SOURCE = "bloomberg", recent quarters): the narrow
#     default below is enough.
#   - Full backtest (UNIVERSE_SOURCE = "embeddings"): set DATE_FROM to your
#     embedding start (e.g. 2005-01-01). NOTE: large multi-GB daily pull.
DATE_FROM <- as.Date("2005-01-01")
DATE_TO   <- as.Date("2026-04-30")

message(sprintf("Querying %s.%s (stock) for %s..%s ...", CRSP_SCHEMA, CRSP_TABLE, DATE_FROM, DATE_TO))

dsf <- tbl(wrds, in_schema(CRSP_SCHEMA, CRSP_TABLE)) |>
  filter(dlycaldt >= DATE_FROM, dlycaldt <= DATE_TO) |>
  select(permno, dlycaldt, cusip, hdrcusip, ticker, siccd,
         dlyret, dlyretx, dlyprc, dlyclose, dlycap, shrout) |>
  collect()

# Keep rows usable for matching (need a CUSIP to bridge to ISIN).
dsf <- dsf |> filter(!is.na(cusip), cusip != "")

write_parquet(dsf, "prices_crsp2.parquet")
dbDisconnect(wrds)

message(sprintf("prices_crsp2.parquet: %s rows, %s securities, %s..%s",
                format(nrow(dsf), big.mark = ","),
                format(dplyr::n_distinct(dsf$permno), big.mark = ","),
                min(dsf$dlycaldt), max(dsf$dlycaldt)))