# =============================================================================
# merge_fundamentals_crsp_ccm.R
# =============================================================================

library(arrow)
library(dplyr)
library(lubridate)
library(RPostgres)
library(DBI)

# ── 1. Load Local Data ───────────────────────────────────────────────────────
message("Loading local datasets...")
funds <- read_parquet("fundamentals.parquet")
crsp  <- read_parquet("prices_crsp2.parquet")

# ── 2. Connect to WRDS & Pull Link Table ─────────────────────────────────────
message("Pulling CCM Link Table from WRDS...")
wrds <- dbConnect(
  Postgres(), host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

# Pull the official gvkey -> permno bridge
link_table <- tbl(wrds, in_schema("crsp", "ccmxpf_linktable")) |>
  filter(linktype %in% c("LC", "LU"), linkprim %in% c("P", "C")) |>
  select(gvkey, lpermno, linkdt, linkenddt) |>
  collect() |>
  mutate(
    # Handle missing end dates (means the link is active today)
    linkenddt = coalesce(as.Date(linkenddt), Sys.Date())
  )

dbDisconnect(wrds)

# ── 3. Prep CRSP (Convert to Millions & Quarters) ────────────────────────────
message("Prepping CRSP data...")
crsp_quarterly <- crsp |>
  mutate(quarter = ceiling_date(as.Date(dlycaldt), "quarter") - days(1)) |>
  group_by(permno, quarter) |>
  slice_max(order_by = dlycaldt, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(permno, quarter, crsp_me = dlycap) |>
  mutate(crsp_me = crsp_me / 1000) # Fix the Thousands -> Millions unit mismatch!

# ── 4. The Grand Merge ───────────────────────────────────────────────────────
message("Merging using permanent IDs and date ranges...")

# A. Join fundamentals to the link table based on gvkey
funds_linked <- funds |>
  left_join(link_table, by = "gvkey", relationship = "many-to-many") |>
  # B. Filter so the quarter falls exactly within the active link window
  filter(quarter >= linkdt, quarter <= linkenddt) |>
  select(-linkdt, -linkenddt)

# C. Join to CRSP using the permno we just attached
funds_merged <- funds_linked |>
  left_join(crsp_quarterly, by = c("lpermno" = "permno", "quarter")) |>
  mutate(
    market_equity = coalesce(crsp_me, market_equity)
  ) |>
  select(-crsp_me, -lpermno) |>
  distinct(gvkey, quarter, .keep_all = TRUE) # Clean up any duplicates

# ── 5. Save ──────────────────────────────────────────────────────────────────
write_parquet(funds_merged, "fundamentals_merged.parquet")
message("Saved robust dataset to fundamentals_merged.parquet")