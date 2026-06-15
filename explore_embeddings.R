# explore_embeddings.R  —  Health Care clusters (validates the stat-arb peer groups)
# ----------------------------------------------------------------------------
# Refocused on the peer structure healthcare_statarb.R actually trades: cluster
# the OS-BERT asset embeddings WITHIN Health Care (K matched to the strategy),
# and contrast the embedding clusters against the GICS HC sub-industries. This
# is the visual / quantitative companion to the backtest's "embeddings beat
# GICS" result — it shows *how* the embedding regroups HC vs the obvious
# benchmark. Company names come from the Compustat bridge, so the neighbour tour
# shows real firms.
#
# Inputs: embeddings_os/q_*.parquet, finratio.parquet, gics.parquet
# ----------------------------------------------------------------------------
library(arrow); library(tidyverse); library(skmeans); library(cluster)

# ── Parameters ────────────────────────────────────────────────
EMB_FILE         <- "embeddings_os/q_2025-12-31.parquet"  # representative quarter — point at
                                                          # the SAME file the strategy clusters
                                                          # (your file used embeddings_os/test/;
                                                          # use whichever holds 2025-12-31)
FINRATIO_PARQUET <- "finratio.parquet"   # cusip -> gvkey  (wrds_pull_finratio.R)
GICS_PARQUET     <- "gics.parquet"        # gvkey -> GICS   (wrds_pull_gics.R)
TRADE_SECTOR     <- "Health Care"
K                <- 8        # embedding clusters — match N_CLUSTERS in healthcare_statarb.R
N_NEIGHBORS      <- 8
OUT_DIR          <- "exhibits"
SEED             <- 42
SIL_SAMPLE       <- 2000     # subsample for silhouette (full NxN is large; HC is small)

set.seed(SEED)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

GICS_NAMES <- c("10"="Energy","15"="Materials","20"="Industrials","25"="Consumer Discretionary",
                "30"="Consumer Staples","35"="Health Care","40"="Financials",
                "45"="Information Technology","50"="Communication Services",
                "55"="Utilities","60"="Real Estate")
# GICS HC industries (gind, 6-digit) -> readable names; raw code kept as fallback.
HC_GIND_NAMES <- c("351010"="HC Equipment & Supplies","351020"="HC Providers & Services",
                   "351030"="HC Technology","352010"="Biotechnology",
                   "352020"="Pharmaceuticals","352030"="Life Sciences Tools")

# 8-char CUSIP is the common key: US ISIN = "US" + 9-char CUSIP -> chars 3..10.
ck8_cusip <- function(x) toupper(substr(trimws(as.character(x)), 1, 8))
ck8_isin  <- function(x) toupper(substr(trimws(as.character(x)), 3, 10))


# ── Load embeddings ───────────────────────────────────────────
emb_raw  <- read_parquet(EMB_FILE)
q_label  <- basename(EMB_FILE) |> str_remove("^q_") |> str_remove("\\.parquet$")
dim_cols <- grep("^dim_", names(emb_raw), value = TRUE)


# ── GICS lookup: ck8 -> sector / HC industry / company name ───
fr   <- read_parquet(FINRATIO_PARQUET)
ckgv <- tibble(ck = ck8_cusip(fr$cusip), gvkey = as.character(fr$gvkey)) |>
  filter(ck != "", !is.na(gvkey), gvkey != "") |> distinct(ck, .keep_all = TRUE)
g <- read_parquet(GICS_PARQUET) |>
  transmute(gvkey = as.character(gvkey),
            gics_sector = unname(GICS_NAMES[as.character(gsector)]),
            gind = as.character(gind), conm) |>
  distinct(gvkey, .keep_all = TRUE)
gics_lkp <- ckgv |> left_join(g, by = "gvkey") |>
  transmute(ck, gics_sector,
            gics_industry = coalesce(unname(HC_GIND_NAMES[gind]), gind),
            conm)


# ── Join + filter to Health Care ──────────────────────────────
emb <- emb_raw |>
  mutate(ck = ck8_isin(isin)) |>
  left_join(gics_lkp, by = "ck") |>
  filter(gics_sector == TRADE_SECTOR, !is.na(gics_industry))

if (nrow(emb) < 2 * K)
  stop(sprintf("only %d HC names matched the GICS bridge — check the CUSIP join", nrow(emb)))

ids  <- emb$issuer_id
nm   <- coalesce(emb$conm, emb$issuer_id)        # real company names where available
X    <- as.matrix(emb[dim_cols])
l2 <- sqrt(rowSums(X^2)); l2[l2 == 0] <- 1
Xn <- X / l2                                      # rows on the unit sphere

cat(sprintf("%s Health Care assets x %d dims (quarter %s)\n",
            format(nrow(Xn), big.mark = ","), length(dim_cols), q_label))


# ── 1. Spherical k-means WITHIN Health Care ───────────────────
sk          <- skmeans(Xn, k = K)
labels      <- sk$cluster
emb$cluster <- labels
gics_grp    <- as.integer(factor(emb$gics_industry))   # GICS partition (for the contrast)


# ── 2. Cohesion + the GICS contrast ───────────────────────────
# On unit-norm rows tcrossprod is cosine similarity, so 1 - tcrossprod is exact
# cosine distance. Silhouette of the embedding clusters vs the GICS partition vs
# a random placebo, all measured in embedding space.
sil_score <- function(mat, lab, n = SIL_SAMPLE) {
  idx <- sample(seq_len(nrow(mat)), min(n, nrow(mat)))
  d   <- as.dist(1 - tcrossprod(mat[idx, , drop = FALSE]))
  mean(silhouette(lab[idx], d)[, "sil_width"])
}
sil_emb  <- sil_score(Xn, labels)
sil_gics <- sil_score(Xn, gics_grp)
Xp <- matrix(rnorm(length(Xn)), nrow = nrow(Xn)); Xp <- Xp / sqrt(rowSums(Xp^2))
sil_p <- sil_score(Xp, skmeans(Xp, k = K)$cluster)

# Adjusted Rand index between the embedding clusters and the GICS industries:
# low => the embedding regroups HC differently from GICS.
adj_rand <- function(a, b) {
  tab <- table(a, b)
  si  <- sum(choose(as.numeric(rowSums(tab)), 2))
  sj  <- sum(choose(as.numeric(colSums(tab)), 2))
  sij <- sum(choose(as.numeric(tab), 2))
  ex  <- si * sj / choose(sum(tab), 2)
  (sij - ex) / ((si + sj) / 2 - ex)
}
ari_val <- adj_rand(labels, gics_grp)

cat(sprintf("cosine silhouette   embeddings=%.3f   GICS industries=%.3f   random=%.3f\n",
            sil_emb, sil_gics, sil_p))
cat(sprintf("embedding-vs-GICS adjusted Rand = %.3f  (low => embeddings carve HC differently)\n",
            ari_val))

# How each embedding cluster is composed of GICS industries (the cross-tab that
# shows clusters cutting across / merging GICS buckets).
comp <- emb |> count(cluster, gics_industry) |>
  group_by(cluster) |> mutate(share = n / sum(n)) |> ungroup()
write_csv(comp, file.path(OUT_DIR, sprintf("cluster_gics_composition_%s.csv", q_label)))


# ── 3. 2D projection — coloured by embedding cluster AND by GICS ──
project_2d <- function(mat) {
  if (requireNamespace("uwot", quietly = TRUE)) {
    list(coords = uwot::umap(mat, n_components = 2, metric = "cosine"), method = "UMAP")
  } else if (requireNamespace("Rtsne", quietly = TRUE)) {
    ts <- Rtsne::Rtsne(mat, dims = 2, perplexity = min(30, floor((nrow(mat)-1)/3)),
                       check_duplicates = FALSE)
    list(coords = ts$Y, method = "t-SNE")
  } else {
    list(coords = prcomp(mat, rank. = 2)$x[, 1:2], method = "PCA")
  }
}
proj <- project_2d(Xn)
emb$x2d <- proj$coords[, 1]; emb$y2d <- proj$coords[, 2]

base_theme <- theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank())

# (a) coloured by embedding cluster — the groups the strategy trades
p_clu <- ggplot(emb, aes(x2d, y2d, color = factor(cluster))) +
  geom_point(size = 1.2, alpha = 0.8) +
  scale_color_discrete(guide = "none") +
  labs(title = sprintf("HC embedding clusters — %s", q_label),
       subtitle = sprintf("%s HC firms, K=%d (skmeans); silhouette %.2f vs %.2f random",
                          format(nrow(emb), big.mark = ","), K, sil_emb, sil_p),
       x = NULL, y = NULL) + base_theme

# (b) SAME map coloured by GICS industry — the benchmark partition, for contrast
p_gics <- ggplot(emb, aes(x2d, y2d, color = gics_industry)) +
  geom_point(size = 1.2, alpha = 0.8) +
  labs(title = sprintf("Same firms, coloured by GICS industry — %s", q_label),
       subtitle = sprintf("GICS silhouette %.2f; adj. Rand vs embedding clusters %.2f",
                          sil_gics, ari_val),
       color = "GICS industry", x = NULL, y = NULL) + base_theme

ggsave(file.path(OUT_DIR, sprintf("hc_clusters_embedding_%s.png", q_label)),
       p_clu,  width = 9, height = 7, dpi = 160)
ggsave(file.path(OUT_DIR, sprintf("hc_clusters_gics_%s.png", q_label)),
       p_gics, width = 9.6, height = 7, dpi = 160)

emb |> select(issuer_id, isin, conm, gics_industry, cluster, x2d, y2d) |>
  write_parquet(file.path(OUT_DIR, sprintf("hc_clusters_%s.parquet", q_label)))

cat("\ncluster sizes:\n"); print(table(labels))
cat("saved maps + assignments ->", OUT_DIR, "\n")


# ── 4. Nearest neighbours (cosine), annotated with GICS industry ──
# The pitch artifact: a firm's embedding neighbours often span GICS industries —
# the cross-industry peer grouping GICS can't produce.
ind_of <- setNames(emb$gics_industry, emb$issuer_id)
neighbors <- function(id, k = N_NEIGHBORS) {
  i <- match(id, ids); if (is.na(i)) return(NULL)
  sims <- as.vector(Xn %*% Xn[i, ])
  ord  <- order(sims, decreasing = TRUE); ord <- ord[ids[ord] != id][seq_len(k)]
  tibble(name = nm[ord], industry = emb$gics_industry[ord], cosine = sims[ord])
}

# representative firm per cluster = closest to the cluster's mean direction
reps <- vapply(sort(unique(labels)), function(c) {
  m <- which(labels == c)
  ctr <- colMeans(Xn[m, , drop = FALSE]); ctr <- ctr / sqrt(sum(ctr^2))
  m[which.max(Xn[m, , drop = FALSE] %*% ctr)]
}, integer(1))

cat("\n— representative HC firm per cluster + nearest neighbours —\n")
for (j in seq_along(reps)) {
  qi <- reps[j]
  cat(sprintf("\n  cluster %d | %s  [%s]\n", labels[qi], nm[qi], emb$gics_industry[qi]))
  nb <- neighbors(ids[qi])
  for (r in seq_len(nrow(nb)))
    cat(sprintf("      %.3f  %-28s %s\n", nb$cosine[r], str_trunc(nb$name[r], 28), nb$industry[r]))
}

cat("\nDone.\n")