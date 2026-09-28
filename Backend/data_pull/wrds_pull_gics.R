# =============================================================================
# wrds_pull_gics.R  —  static gvkey -> GICS sector (+ company name) from Compustat
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
