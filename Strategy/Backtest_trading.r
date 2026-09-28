# =============================================================================
# AI Valuation Strategy Backtest (Long/Short Top 10)
# =============================================================================
# Strategy:
#   - LONG the Top 10 most undervalued firms (Lowest p_perp)
#   - SHORT the Top 10 most overvalued firms (Highest p_perp)
#   - Equal weight, rebalanced quarterly.
# =============================================================================

library(tidyverse)
library(arrow)
library(lubridate)
library(scales)

LAST_10_YEARS <- TRUE          # TRUE = backtest on last 10 years only; FALSE = full history   
PORTFOLIO_SIZE <- 5


# ── 1. Load Data ─────────────────────────────────────────────────────────────
message("Loading Market Data and AI Predictions...")

# Load the raw market data (for calculating forward returns)
market_data <- read_parquet("valuation_metric.parquet") |>
  select(quarter, ck, log_me)

# Load the AI predictions we just saved
ai_preds <- read_parquet("ai_predictions.parquet") |>
  # Calculate the AI Mispricing Signal
  mutate(
    mispricing = actual_p_perp - ai_predicted_p_perp
  )

# Merge them together
signals <- ai_preds |>
  inner_join(market_data, by = c("quarter", "ck")) |>
  arrange(ck, quarter)

# ── 1b. Keep Healthcare firms only (GICS Sector 35) ──────────────────────────
# ck -> gvkey via fundamentals_merged; gvkey -> sector via gics.parquet.
# (substr(.,1,2) == "35" works whether gsector is "35" or an 8-digit subindustry.)
hc_gvkeys <- read_parquet("gics.parquet") |>
  mutate(gvkey = as.character(gvkey)) |>
  filter(substr(as.character(gsector), 1, 2) == "35") |>
  pull(gvkey)

hc_tickers <- read_parquet("fundamentals_merged.parquet") |>
  mutate(gvkey = as.character(gvkey)) |>
  filter(gvkey %in% hc_gvkeys) |>
  distinct(ck) |>
  pull(ck)

signals <- signals |> filter(ck %in% hc_tickers)
message(sprintf("Healthcare universe: %d firm-quarters across %d names.",
                nrow(signals), n_distinct(signals$ck)))

# ── 2. Calculate Forward Returns ─────────────────────────────────────────────
message("Calculating forward quarterly returns...")

signals <- signals |>
  group_by(ck) |>
  mutate(
    next_me = lead(exp(log_me), 1),
    fwd_return = (next_me / exp(log_me)) - 1
  ) |>
  ungroup() |>
  filter(!is.na(fwd_return))

if (LAST_10_YEARS) {
  cutoff_date <- max(signals$quarter) - years(7)
  signals <- signals |> filter(quarter >= cutoff_date)
}

# ── 3. Portfolio Sorting (The AI Engine) ─────────────────────────────────────
message("Sorting portfolios based on AI Mispricing...")

PORTFOLIO_SIZE <- 60

portfolios <- signals |>
  group_by(quarter) |>
  # Rank by mispricing: 
  # Lowest values = Actual is lower than AI prediction (AI says it's CHEAP)
  # Highest values = Actual is higher than AI prediction (AI says it's EXPENSIVE)
  mutate(ai_rank = rank(mispricing, ties.method = "first")) |>
  mutate(
    position = case_when(
      ai_rank <= PORTFOLIO_SIZE ~ "LONG",                                 
      ai_rank > (n() - PORTFOLIO_SIZE) ~ "SHORT",                         
      TRUE ~ "NEUTRAL"
    )
  ) |>
  filter(position != "NEUTRAL") |>
  ungroup()

# ── 4. Calculate Strategy Returns ────────────────────────────────────────────
message("Simulating quarterly AI rebalancing...")

strategy_returns <- portfolios |>
  group_by(quarter, position) |>
  summarise(bucket_return = mean(fwd_return, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = position, values_from = bucket_return) |>
  mutate(strategy_return = LONG - SHORT) |>
  arrange(quarter)

strategy_returns <- strategy_returns |>
  mutate(
    cum_long = cumprod(1 + LONG),
    cum_short = cumprod(1 + SHORT), 
    cum_strategy = cumprod(1 + strategy_return)
  )

# Print Summary Stats
ann_ret <- (tail(strategy_returns$cum_strategy, 1) ^ (4 / nrow(strategy_returns))) - 1
message(sprintf("==========================================="))
message(sprintf("AI Stat-Arb Annualized Return: %.2f%%", ann_ret * 100))
message(sprintf("Total Cumulative Gain: %.2fX Initial Capital", tail(strategy_returns$cum_strategy, 1)))
message(sprintf("==========================================="))

# ── 5. Visualize the AI Backtest ─────────────────────────────────────────────
message("Generating AI Equity Curve chart...")

p <- ggplot(strategy_returns, aes(x = quarter, y = cum_strategy)) +
  geom_line(color = "#008080", linewidth = 1.2) + # Distinct Teal color for the AI Strategy
  geom_ribbon(aes(ymin = 1, ymax = cum_strategy), fill = "#008080", alpha = 0.15) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  scale_y_continuous(labels = dollar_format(prefix = "$")) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18, margin = margin(b = 8)),
    plot.subtitle = element_text(color = "gray30", size = 12, margin = margin(b = 20)),
    panel.grid.minor = element_blank()
  ) +
  labs(
    title = "AI Textual Arbitrage: Cumulative Equity Curve",
    subtitle = sprintf("Long Top %d AI-Undervalued / Short Top %d AI-Overvalued", PORTFOLIO_SIZE, PORTFOLIO_SIZE),
    x = NULL, y = "Growth of $1.00",
    caption = "Signal: (Actual Market-to-Book Residual) minus (OS-BERT Predicted Residual)"
  )

print(p)
ggsave("AI_True_Arbitrage_Backtest.png", plot = p, width = 10, height = 6, dpi = 300)

# ── 4. Calculate Strategy Returns & Performance Metrics ──────────────────────
message("Simulating quarterly AI rebalancing and calculating risk metrics...")

strategy_returns <- portfolios |>
  group_by(quarter, position) |>
  summarise(bucket_return = mean(fwd_return, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = position, values_from = bucket_return) |>
  mutate(strategy_return = LONG - SHORT) |>
  arrange(quarter)

# Calculate Cumulative Wealth and Drawdowns
strategy_returns <- strategy_returns |>
  mutate(
    cum_long = cumprod(1 + LONG),
    cum_short = cumprod(1 + SHORT), 
    cum_strategy = cumprod(1 + strategy_return),
    # Drawdown math: Track the highest peak so far, and measure how far we fall from it
    rolling_peak = cummax(cum_strategy),
    drawdown = (cum_strategy / rolling_peak) - 1
  )

# Calculate Institutional Risk/Return Metrics
# Since we rebalance quarterly, our annualization factor is 4.
ann_factor <- 4 
n_quarters <- nrow(strategy_returns)

ann_ret  <- (tail(strategy_returns$cum_strategy, 1) ^ (ann_factor / n_quarters)) - 1
ann_vol  <- sd(strategy_returns$strategy_return, na.rm = TRUE) * sqrt(ann_factor)
sharpe   <- ann_ret / ann_vol  # Assuming a 0% risk-free rate for a dollar-neutral L/S spread

# Sortino: like Sharpe but penalizes only downside (target return = 0%).
# Downside deviation is the root lower-partial-moment over ALL quarters, annualized like vol.
downside_dev <- sqrt(mean(pmin(strategy_returns$strategy_return, 0)^2, na.rm = TRUE)) * sqrt(ann_factor)
sortino  <- ann_ret / downside_dev

max_dd   <- min(strategy_returns$drawdown, na.rm = TRUE)
win_rate <- mean(strategy_returns$strategy_return > 0, na.rm = TRUE)

# Print the Institutional Tear Sheet
message(sprintf("=================================================="))
message(sprintf("     AI STAT-ARB PERFORMANCE TEAR SHEET           "))
message(sprintf("=================================================="))
message(sprintf("Total Cumulative Gain: %.2fX Initial Capital", tail(strategy_returns$cum_strategy, 1)))
message(sprintf("Annualized Return:     %.2f%%", ann_ret * 100))
message(sprintf("Annualized Volatility: %.2f%%", ann_vol * 100))
message(sprintf("Sharpe Ratio:          %.2f", sharpe))
message(sprintf("Sortino Ratio:         %.2f", sortino))
message(sprintf("Maximum Drawdown:      %.2f%%", max_dd * 100))
message(sprintf("Win Rate (Quarterly):  %.1f%%", win_rate * 100))
message(sprintf("=================================================="))