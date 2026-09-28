# =============================================================================
# wrds_pull_fundamentals.R
# =============================================================================
# Book equity, earnings, and market equity — all from ONE table: the
# CRSP/Compustat Merged Fundamentals Quarterly file (ccmfundq). Writes
# fundamentals.parquet with columns: ck, quarter, book_equity, earnings,
# market_equity.
#
#   Rscript wrds_pull_fundamentals.R
# =============================================================================

library(RPostgres)
library(DBI)
library(dplyr)
library(dbplyr)
library(lubridate)
library(arrow)

# ── 1. WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

# ── 2. Pull the needed columns from raw Compustat ────────────────────────────
# Removed the CRSP link filters because those variables do not exist here.
raw <- tbl(wrds, in_schema("comp_na_daily_all", "fundq")) |>
  filter(
    curncdq == "USD",
    !is.na(cusip),
    datadate >= "2005-01-01"  
  ) |>
  select(
    gvkey, cusip, datadate,
    seqq, ceqq, pstkq, atq, ltq, txditcq,   # book equity inputs
    ibq,                                    # earnings
    mkvaltq                                 # market value
  ) |>
  collect()

dbDisconnect(wrds)

# ── 3. Build the fields ──────────────────────────────────────────────────────
fundamentals <- raw |>
  mutate(
    pref = coalesce(pstkq, 0),
    dtax = coalesce(txditcq, 0),
    se   = coalesce(seqq, ceqq + pref, atq - ltq),   # shareholders' equity, with fallbacks
    book_equity   = se - pref + dtax,
    earnings      = ibq,
    market_equity = mkvaltq,
    ck      = toupper(substr(cusip, 1, 8)),
    quarter = ceiling_date(as.Date(datadate), "quarter") - days(1)   # calendar quarter-end
  ) |>
  filter(!is.na(ck), ck != "") |>
  select(gvkey, ck, quarter, book_equity, earnings, market_equity) |>  # <--- Add gvkey here
  distinct(gvkey, quarter, .keep_all = TRUE)

# ── 4. Save ──────────────────────────────────────────────────────────────────
write_parquet(fundamentals, "fundamentals.parquet")
message(sprintf("fundamentals.parquet: %d firm-quarters | %s..%s",
                nrow(fundamentals), min(fundamentals$quarter), max(fundamentals$quarter)))
