# find out what the last observations are in the prices_crsp2.parquet file
library(arrow)
library(dplyr)

last_obs <- read_parquet("prices_crsp2.parquet") %>%
  tail(1) %>%
  print()