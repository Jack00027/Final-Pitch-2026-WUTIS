# =============================================================================
# Visualize Out-of-Sample R^2 Over Time
# =============================================================================

library(tidyverse)
library(arrow)
library(scales) # Needed to format the Y-axis as percentages

# ── 1. Load and Prep the Data ────────────────────────────────────────────────
message("Loading ridge regression results...")
results <- read_parquet("ridge_oos_results.parquet")

# Calculate the quarter-by-quarter Out-of-Sample R^2
# Formula: 1 - (Variance of Error / Variance of Target)
plot_data <- results |>
  mutate(
    quarterly_r2 = 1 - (var_error / var_target)
  ) |>
  arrange(quarter)

# ── 2. Build the Pitch-Deck Chart ────────────────────────────────────────────
message("Generating chart...")

p <- ggplot(plot_data, aes(x = quarter, y = quarterly_r2)) +
  
  # A. The raw quarterly data (Faint grey line and points)
  geom_line(color = "gray75", linewidth = 0.5) +
  geom_point(color = "gray60", size = 1.5, alpha = 0.7) +
  
  # B. The Smoothed Trendline (Bold Navy Blue)
  # 'span = 0.35' controls how sensitive the curve is to local bumps. 
  geom_smooth(method = "loess", span = 0.35, 
              color = "#003366", fill = "#6699CC", alpha = 0.2, linewidth = 1.2) +
  
  # C. Axis Formatting
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  
  # D. Institutional Styling
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18, margin = margin(b = 8)),
    plot.subtitle = element_text(color = "gray30", size = 12, margin = margin(b = 20)),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(), # Cleaner look for time series
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  
  # E. Labels
  labs(
    title = "AI Valuation Predictive Power Over Time",
    subtitle = "Out-of-Sample R² of the OS-BERT Ridge Regression vs. Raw Market-to-Book Premium",
    x = NULL,
    y = "Out-of-Sample R²",
    caption = "Data: CRSP/Compustat CCM Linked | Embeddings: OS-BERT | Valuation: p_perp"
  )

# ── 3. Display and Save ──────────────────────────────────────────────────────
print(p)

# Save a high-resolution PNG perfect for a PowerPoint/Keynote slide
ggsave("AI_Valuation_Trend.png", plot = p, width = 10, height = 6, dpi = 300)
message("Saved high-resolution chart to AI_Valuation_Trend.png")