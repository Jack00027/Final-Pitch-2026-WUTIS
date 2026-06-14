# =============================================================================
# wrds_pull_gics.R  —  static gvkey -> GICS sector (+ company name) from Compustat
# =============================================================================
# Produces  gics.parquet  for WUTIS_book.R (UNIVERSE_SOURCE = "embeddings").
#
# comp.company is one row per gvkey (header file) covering ACTIVE and INACTIVE
# (delisted) firms, so the static GICS classification it carries is survivorship-
# free — exactly what a historical backtest needs. GICS sector membership is
# stable enough that a single (latest) label per firm is a sound approximation;
# the point of using Compustat rather than the Bloomberg snapshot is COVERAGE of
# the names that have since delisted, not point-in-time precision.
#
# WUTIS_book.R joins this to the Financial Ratios table by gvkey, and maps the
# 2-digit gsector code to a name (35 = Health Care) via its GICS_NAMES table.
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

# Compustat company header. gsector = GICS sector, ggroup/gind/gsubind = finer
# GICS levels (kept for reference), conm = company name.
gics <- tbl(wrds, in_schema("comp", "company")) |>
  select(gvkey, conm, gsector, ggroup, gind, gsubind) |>
  collect()

gics <- gics |> filter(!is.na(gsector), gsector != "")
write_parquet(gics, "gics.parquet")
dbDisconnect(wrds)

message(sprintf("gics.parquet: %s gvkeys with a GICS sector (sectors: %s)",
                format(nrow(gics), big.mark = ","),
                paste(sort(unique(gics$gsector)), collapse = ",")))
