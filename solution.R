# Load required libraries
library(tidyverse)
library(lubridate)
library(plotly)

print(paste("---> R Script Start", Sys.time()))

# Set up parameters
start_train <- as.Date("2017-01-01")
end_train <- as.Date("2023-11-30")
start_test <- as.Date("2024-01-01")
end_test <- as.Date("2024-06-30")

n_buys <- 10
lookback_period <- 20  # Number of days to look back for momentum calculation

# Load data
df_sectors <- read.csv('data/data0.csv')
df_data <- read.csv('data/data1.csv')
df_data$date <- as.Date(df_data$date)

# Function to calculate momentum
calculate_momentum <- function(df) {
  df %>%
    group_by(security) %>%
    arrange(date) %>%
    mutate(momentum = (price / lag(price, lookback_period) - 1) * 100) %>%
    ungroup()
}

# Prepare data
df_momentum <- df_data %>%
  select(date, security, price) %>%
  calculate_momentum()

# Modified generate_buy_signals function
generate_buy_signals <- function(df, n_buys) {
  df %>%
    arrange(desc(momentum)) %>%
    slice_head(n = n_buys) %>%
    mutate(signal = 1) %>%
    select(security, signal)
}

# Generate signals for all test dates
df_signals <- df_momentum %>%
  filter(date >= start_test, date <= end_test) %>%
  group_by(date) %>%
  group_modify(~ generate_buy_signals(.x, n_buys)) %>%
  ungroup()

# Create buy matrix
df_buys <- df_signals %>%
  pivot_wider(names_from = security, values_from = signal, values_fill = 0)

# Plot heatmap of buy signals
fig_pixel <- plot_ly(z = as.matrix(df_buys[,-1]), x = colnames(df_buys)[-1], y = df_buys$date, type = "heatmap")
fig_pixel

# Load returns data
df_returns <- read.csv('data/returns.csv')
df_returns$date <- as.Date(df_returns$date)
df_returns <- df_returns %>% 
  filter(date >= start_test) %>%
  pivot_wider(names_from = security, values_from = return1)

# Updated calculate_portfolio_performance function
calculate_portfolio_performance <- function(df_buys, df_returns) {
  # Ensure df_buys and df_returns have the same columns, excluding 'date'
  common_cols <- setdiff(intersect(names(df_buys), names(df_returns)), "date")
  print("Common columns (excluding 'date'):")
  print(common_cols)
  
  df_buys <- df_buys[, c("date", common_cols)]
  df_returns <- df_returns[, c("date", common_cols)]
  
  # Aggregate data by date to ensure uniqueness
  df_buys <- df_buys %>%
    group_by(date) %>%
    summarise(across(everything(), sum)) %>%
    ungroup()
  print("Aggregated df_buys:")
  print(head(df_buys))
  
  df_returns <- df_returns %>%
    group_by(date) %>%
    summarise(across(everything(), mean)) %>%
    ungroup()
  print("Aggregated df_returns:")
  print(head(df_returns))
  
  # Align dates
  df_combined <- inner_join(df_buys, df_returns, by = "date", suffix = c("_buys", "_returns"))
  print("Combined data frame:")
  print(head(df_combined))
  
  # Calculate daily portfolio return
  df_combined$portfolio_return <- rowSums(df_combined[, paste0(common_cols, "_buys")] * 
                                            df_combined[, paste0(common_cols, "_returns")], na.rm = TRUE) / n_buys
  print("Data frame with portfolio return:")
  print(head(df_combined))
  
  # Calculate cumulative return
  df_combined$cumulative_return <- cumprod(1 + df_combined$portfolio_return)
  print("Data frame with cumulative return:")
  print(head(df_combined))
  
  return(df_combined)
}

# Use the updated function
df_performance <- calculate_portfolio_performance(df_buys, df_returns)

# Plot performance
fig_performance <- plot_ly(df_performance, x = ~date, y = ~cumulative_return, type = 'scatter', mode = 'lines')
fig_performance

# Print final performance
final_return <- tail(df_performance$cumulative_return, 1)
print(paste("---> Total return for the momentum strategy between", 
            min(df_performance$date), "and", max(df_performance$date), 
            "is", round((final_return - 1) * 100, 2), "%"))

print(paste("---> R Script End", Sys.time()))
print(paste("---> Total time taken", difftime(Sys.time(), as.POSIXct(start_train), units = "secs")))


# Load required libraries
library(tidyverse)
library(lubridate)
library(plotly)

print(paste("---> R Script Start", Sys.time()))

# Set up parameters
start_train <- as.Date("2017-01-01")
end_train <- as.Date("2023-11-30")
start_test <- as.Date("2024-01-01")
end_test <- as.Date("2024-06-30")

n_buys <- 10
lookback_periods <- c(10, 20, 30, 40, 50)  # Different lookback periods to compare

# Load data
df_data <- read.csv('data/data1.csv')
df_data$date <- as.Date(df_data$date)

# Function to calculate momentum
calculate_momentum <- function(df, lookback_period) {
  df %>%
    group_by(security) %>%
    arrange(date) %>%
    mutate(momentum = (price / lag(price, lookback_period) - 1) * 100) %>%
    ungroup()
}

# Initialize a list to store performance data for each lookback period
performance_list <- list()

# Loop through different lookback periods
for (lookback_period in lookback_periods) {
  # Prepare data
  df_momentum <- df_data %>%
    select(date, security, price) %>%
    calculate_momentum(lookback_period)
  
  # Generate signals for all test dates
  df_signals <- df_momentum %>%
    filter(date >= start_test, date <= end_test) %>%
    group_by(date) %>%
    group_modify(~ generate_buy_signals(.x, n_buys)) %>%
    ungroup()
  
  # Create buy matrix
  df_buys <- df_signals %>%
    pivot_wider(names_from = security, values_from = signal, values_fill = 0)
  
  # Load returns data
  df_returns <- read.csv('data/returns.csv')
  df_returns$date <- as.Date(df_returns$date)
  df_returns <- df_returns %>% 
    filter(date >= start_test) %>%
    pivot_wider(names_from = security, values_from = return1)
  
  # Calculate portfolio performance
  df_performance <- calculate_portfolio_performance(df_buys, df_returns)
  
  # Store cumulative return for the current lookback period
  final_return <- tail(df_performance$cumulative_return, 1)
  performance_list[[as.character(lookback_period)]] <- (final_return - 1) * 100  # Store as percentage
}

# Prepare data for plotting
df_performance_comparison <- data.frame(
  Lookback_Period = lookback_periods,
  Total_Return = unlist(performance_list)
)

# Plotting chunk
{
  # Plot the performance comparison
  fig_comparison <- plot_ly(df_performance_comparison, 
                            x = ~Lookback_Period, 
                            y = ~Total_Return, 
                            type = 'bar', 
                            name = 'Total Return (%)',
                            marker = list(color = 'rgba(255, 0, 0, 0.6)'))
  
  fig_comparison <- fig_comparison %>%
    layout(title = "Total Return Comparison by Lookback Period",
           xaxis = list(title = "Lookback Period (Days)"),
           yaxis = list(title = "Total Return (%)"))
  
  fig_comparison
}

print(paste("---> R Script End", Sys.time()))
