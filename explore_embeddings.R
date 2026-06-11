# explore_embeddings.R

library(arrow)
library(tidyverse)
library(skmeans)
library(cluster)    

# ── Parameters ────────────────────────────────────────────────
EMB_FILE     <- "embeddings_os/test/q_2025-12-31.parquet"
K            <- 12        # number of spherical-kmeans clusters
NAMES_FILE   <- NA        # optional CSV with columns issuer_id,name (NA to skip)
NEIGHBORS_OF <- NA        # issuer_id to show neighbors for (NA = auto-pick per cluster)
N_NEIGHBORS  <- 8
OUT_DIR      <- "exhibits"
SEED         <- 42
SIL_SAMPLE   <- 2000      # subsample size for silhouette (full NxN is large)

set.seed(SEED)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


# ── Load + L2-normalize ───────────────────────────────────────
emb     <- read_parquet(EMB_FILE)
q_label <- basename(EMB_FILE) |> str_remove("^q_") |> str_remove("\\.parquet$")

dim_cols <- grep("^dim_", names(emb), value = TRUE)
ids <- emb$issuer_id
X   <- as.matrix(emb[dim_cols])

l2 <- sqrt(rowSums(X^2)); l2[l2 == 0] <- 1
Xn <- X / l2                                   # rows on the unit sphere

cat(sprintf("%s assets x %d dims (quarter %s)\n",
            format(nrow(Xn), big.mark = ","), length(dim_cols), q_label))


# ── 1. Spherical k-means ──────────────────────────────────────
sk     <- skmeans(Xn, k = K)
labels <- sk$cluster
emb$cluster <- labels


# ── 3. Cosine silhouette vs random placebo ────────────────────
# On unit-norm rows, tcrossprod is the cosine-similarity matrix, so
# (1 - tcrossprod) is exact cosine distance.
sil_score <- function(mat, lab, n = SIL_SAMPLE) {
  idx <- sample(seq_len(nrow(mat)), min(n, nrow(mat)))
  d   <- as.dist(1 - tcrossprod(mat[idx, , drop = FALSE]))
  mean(silhouette(lab[idx], d)[, "sil_width"])
}
sil <- sil_score(Xn, labels)

Xp <- matrix(rnorm(length(Xn)), nrow = nrow(Xn))
Xp <- Xp / sqrt(rowSums(Xp^2))
lp <- skmeans(Xp, k = K)$cluster
sil_p <- sil_score(Xp, lp)

cat(sprintf("cosine silhouette: embeddings=%.3f  vs  random placebo=%.3f\n",
            sil, sil_p))


# ── 2. 2D projection ──────────────────────────────────────────
project_2d <- function(mat) {
  if (requireNamespace("uwot", quietly = TRUE)) {
    list(coords = uwot::umap(mat, n_components = 2, metric = "cosine"),
         method = "UMAP")
  } else if (requireNamespace("Rtsne", quietly = TRUE)) {
    ts <- Rtsne::Rtsne(mat, dims = 2, perplexity = 30, check_duplicates = FALSE)
    list(coords = ts$Y, method = "t-SNE")
  } else {
    pc <- prcomp(mat, rank. = 2)
    list(coords = pc$x[, 1:2], method = "PCA")
  }
}
proj <- project_2d(Xn)
emb$x2d <- proj$coords[, 1]
emb$y2d <- proj$coords[, 2]

p <- ggplot(emb, aes(x2d, y2d, color = factor(cluster))) +
  geom_point(size = 0.6, alpha = 0.7) +
  scale_color_discrete(guide = "none") +
  labs(title    = sprintf("OS-BERT asset embeddings — %s", q_label),
       subtitle = sprintf("%s firms, %d clusters (%s); silhouette %.2f vs %.2f placebo",
                          format(nrow(emb), big.mark = ","), K, proj$method, sil, sil_p),
       x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        panel.grid = element_blank())

png_path <- file.path(OUT_DIR, sprintf("clusters_%s.png", q_label))
ggsave(png_path, p, width = 9, height = 7, dpi = 160)
cat("saved map ->", png_path, "\n")

# save assignments + coords for custom plots / joining names later
emb |>
  select(issuer_id, quarter_end, cluster, x2d, y2d) |>
  write_parquet(file.path(OUT_DIR, sprintf("clusters_%s.parquet", q_label)))

cat("cluster sizes:\n"); print(table(labels))


# ── 4. Nearest-neighbor "similar firms" ───────────────────────
names_map <- NULL
if (!is.na(NAMES_FILE)) {
  nm <- read_csv(NAMES_FILE, show_col_types = FALSE)
  names_map <- setNames(nm$name, nm$issuer_id)
}
show_name <- function(id) {
  if (!is.null(names_map) && id %in% names(names_map)) names_map[[id]] else id
}

neighbors <- function(id, k = N_NEIGHBORS) {
  i <- match(id, ids)
  if (is.na(i)) return(NULL)
  sims <- as.vector(Xn %*% Xn[i, ])            # cosine sim to every firm
  ord  <- order(sims, decreasing = TRUE)
  ord  <- ord[ids[ord] != id][seq_len(k)]
  tibble(neighbor = ids[ord],
         name     = vapply(ids[ord], show_name, character(1)),
         cosine   = sims[ord])
}

queries <- if (!is.na(NEIGHBORS_OF)) NEIGHBORS_OF else {
  # representative firm per cluster = closest to the cluster's mean direction
  reps <- vapply(sort(unique(labels)), function(c) {
    m <- which(labels == c)
    centroid <- colMeans(Xn[m, , drop = FALSE])
    centroid <- centroid / sqrt(sum(centroid^2))
    m[which.max(Xn[m, , drop = FALSE] %*% centroid)]
  }, integer(1))
  ids[reps][seq_len(min(5, length(reps)))]     # a short tour
}

cat("\n— nearest neighbors (cosine) —\n")
for (qid in queries) {
  cat(sprintf("  %s:\n", show_name(qid)))
  nb <- neighbors(qid)
  if (is.null(nb)) { cat("    not found\n"); next }
  for (r in seq_len(nrow(nb)))
    cat(sprintf("      %.3f  %s\n", nb$cosine[r], nb$name[r]))
}

cat("\nDone.\n")
