# Load necessary libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(randomForest)
library(tidyr)
library(pacman)
# pacman::p_load(randomForest)

print('---> R Script Start')

# Define parameters
start_train <- as.Date("2017-01-01")
end_train <- as.Date("2023-11-30")
start_test <- as.Date("2024-01-01")
end_test <- as.Date("2024-06-30")

n_buys <- 10
verbose <- FALSE

print('---> initial data set up')

# Load sector data
df_sectors <- read.csv('data/data0.csv')

# Load price and financial data
df_data <- read.csv('data/data1.csv')
df_data$date <- as.Date(df_data$date)

df_x <- df_data %>% select(date, security, price, return30, ratio_pe, ratio_pcf, ratio_de, ratio_roe, ratio_roa)
df_y <- df_data %>% select(date, security, label)

list_vars1 <- c('price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa')

# Create signals DataFrame
df_signals <- data.frame(date = unique(df_x$date[df_x$date >= start_test & df_x$date <= end_test]))
df_signals <- df_signals %>% arrange(date)

# Initialize an empty list for storing accuracy results
df_signals$acc_total <- NA
df_signals$acc_current <- NA

for (i in seq_len(nrow(df_signals))) {
  
  if (verbose) print(paste('---> doing', df_signals$date[i]))
  
  # Training set
  df_trainx <- df_x %>% filter(date < df_signals$date[i])
  df_trainx <- df_trainx %>% filter(date != max(date))
  
  df_trainy <- df_y %>% filter(date < df_signals$date[i])
  df_trainy <- df_trainy %>% filter(date != max(date))
  
  # Test set
  df_testx <- df_x %>% filter(date >= df_signals$date[i])
  df_testy <- df_y %>% filter(date >= df_signals$date[i])
  
  # Scale data
  for (col in list_vars1) {
    scaler <- function(x) (x - min(x)) / (max(x) - min(x)) * 2 - 1
    df_trainx[[col]] <- scaler(df_trainx[[col]])
    df_testx[[col]] <- scaler(df_testx[[col]])
  }
  
  df_trainy <- df_trainy %>% mutate(label = as.factor(label)) 
  
  # Fit the Random Forest classifier
  if (i == 1) {
    clf <- randomForest(x = df_trainx[list_vars1], y = df_trainy$label, ntree = 10, mtry = sqrt(length(list_vars1)), nodesize = 1000)
  }
  
  
  # Predictions and accuracy
  pred_probs <- predict(clf, newdata = df_testx[list_vars1], type = "prob")
  df_testy$signal <- pred_probs[, 2]
  df_testy$pred <- ifelse(pred_probs[, 2] > 0.5, 1, 0)
  df_testy$count <- 1
  
  df_current <- df_testy %>% filter(date == df_signals$date[i])
  
  acc_total <- mean(df_testy$label == df_testy$pred)
  acc_current <- mean(df_current$label == df_current$pred)
  
  print(paste('---> accuracy test set', round(acc_total, 2), ', accuracy current date', round(acc_current, 2)))
  
  # Add accuracy and signal to dataframe
  df_signals$acc_total[i] <- acc_total
  df_signals$acc_current[i] <- acc_current
  
  df_signals[i, df_current$security] <- df_current$signal
}

# Create buy matrix for payoff plot
df_signals$`10th` <- apply(df_signals[df_sectors$security], 1, function(x) sort(x, decreasing = TRUE)[n_buys])
df_index <- df_signals[df_sectors$security] >= df_signals$`10th`

# Set 1 for top 10 strongest signals
df_buys <- as.data.frame(matrix(0, nrow = nrow(df_signals), ncol = length(df_sectors$security)))
colnames(df_buys) <- df_sectors$security
df_buys[df_index] <- 1

# keep first 10 obs
process_row <- function(row) {
  ones_indices <- which(row == 1)  # Get the indices of 1's
  if(length(ones_indices) > 10) {
    row[ones_indices[11:length(ones_indices)]] <- 0  # Set ones after the first 10 to 0
  }
  return(row)
}
df_buys <- t(apply(df_buys, 1, process_row))

# add dates
library(zoo)
# df_buys <- cbind(date = as.Date(df_signals$date), df_buys)
df_buys <- cbind(date = df_signals$date, df_buys)


# df_buys[,-1] %>% rowSums()


# Plot signals
ggplot(df_signals, aes(x = date)) + geom_line(aes(y = AAPL)) + ggtitle("AAPL Signals")
image(t(df_buys[-1]))  # You might need to use other visualization for the buy signals

# Create return matrix
df_returns <- read.csv('data/returns.csv')
df_returns$date <- as.Date(df_returns$date)
df_returns <- df_returns %>% filter(date >= start_test)
df_returns <- df_returns %>% pivot_wider(names_from = security, values_from = return1)

plot_payoff <- function(df_buys) {
  
  df <- df_buys[,-1]
  
  if (sum(rowSums(df) == 10) != nrow(df)) {
    stop("---> must have exactly 10 buys each day")
  }
  
  # df_payoff <- df[, 1, drop = FALSE]
  df <- df * (1 / n_buys)  # equally weighted

  df_payoff <- NULL
  arr_ret <- as.matrix(df_returns[-1] + 1)
  df_payoff$payoff <- diag(df %*% t(arr_ret-1))
  df_payoff$tri <- cumprod(1+df_payoff$payoff)
  
  df_payoff$date <- df_returns$date
  df_payoff <- df_payoff %>% as_tibble()
  
  
  payoff_result <- (tail(df_payoff$tri, 1) - 1) * 100
  print(paste("---> payoff for these buys between period", min(df_payoff$date), "and", max(df_payoff$date), "is", round(payoff_result, 2), "%"))
  
  return(df_payoff)
}

df_payoff <- plot_payoff(df_buys)
ggplot(df_payoff, aes(x = date, y = tri)) + geom_line() + ggtitle("Payoff Over Time")


# Install and load necessary libraries
library(h2o)

# 1. Initialize the H2O cluster
h2o.init()

# 2. Prepare the time series data
# Assume 'df_data' is your time series data with 'date' and 'price' columns (or whatever variable you're predicting)
# Example: df_data <- data.frame(date = as.Date('2017-01-01') + 0:1000, price = rnorm(1001, 100, 5))

# Convert date to numerical format for modeling
df_data$time_index <- as.numeric(as.Date(df_data$date))

# Split the data into training and test sets
train_data <- df_data %>% filter(date <= as.Date('2023-11-30'))
test_data <- df_data %>% filter(date >= as.Date('2024-01-01'))

# Convert training and testing data to H2O format
train_h2o <- as.h2o(train_data)
test_h2o <- as.h2o(test_data)

# Define the target variable (e.g., 'price') and feature(s) (e.g., 'time_index')
response <- "price"  # The target variable you want to forecast
predictors <- "time_index"  # The feature to use for modeling

# 3. Run H2O AutoML to train models
aml <- h2o.automl(
  x = predictors,                  # Feature(s)
  y = response,                    # Target variable
  training_frame = train_h2o,      # Training data
  max_runtime_secs = 120,          # Time limit for training (adjust as needed)
  seed = 42,                       # Ensure reproducibility
  project_name = "time_series_automl"
)

# 4. View the leaderboard of models trained
lb <- aml@leaderboard
print(lb)

# 5. Get the best model from the AutoML leaderboard
best_model <- h2o.get_best_model(aml)

# Summary of the best model
print(best_model)

# 6. Make predictions on the test set (future dates)
predictions <- h2o.predict(best_model, test_h2o)

# Convert predictions to R data frame
predictions_df <- as.data.frame(predictions)

# 7. Combine predictions with the actual test data for visualization
result <- cbind(test_data, predictions_df)
names(result)[ncol(result)] <- "predicted_price"

# 8. Plot the actual vs predicted prices
library(ggplot2)

ggplot(result, aes(x = date)) +
  geom_line(aes(y = price, color = "Actual")) +
  geom_line(aes(y = predicted_price, color = "Predicted")) +
  labs(title = "Actual vs Predicted Stock Prices",
       x = "Date", y = "Price") +
  theme_minimal()

# 9. Shut down the H2O cluster (optional)
h2o.shutdown(prompt = FALSE)

