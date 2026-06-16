# upload hc_clusters_2025-12-31.parquet file and transform into csv 

library(arrow)
library(dplyr)
# Read the Parquet file
parquet_file <- "hc_clusters_2025-12-31.parquet"
csv_file     <- sub("\\.parquet$", ".csv", parquet_file)

df <- read_parquet(parquet_file)

# Write out as CSV
write.csv(df, csv_file, row.names = FALSE)

cat(sprintf("Wrote %d rows x %d cols -> %s\n", nrow(df), ncol(df), csv_file))

