# Load necessary libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(randomForest)
library(tidyr)
library(zoo)
library(pacman)
library(caret)

# Ensure required libraries are installed and loaded
pacman::p_load(dplyr, lubridate, ggplot2, randomForest, tidyr, zoo, caret)

print('---> R Script Start')

# Define parameters
start_train <- as.Date("2017-01-01")
end_train <- as.Date("2023-11-30")
start_test <- as.Date("2024-01-01")
end_test <- as.Date("2024-06-30")

n_buys <- 10
verbose <- FALSE

print('---> Initial data set up')

# Load sector data
df_sectors <- read.csv('data0.csv')

# Load price and financial data
df_data <- read.csv('data1.csv')
df_data$date <- as.Date(df_data$date)

# Select relevant features
df_x <- df_data %>% select(date, security, price, return30, ratio_pe, ratio_pcf, ratio_de, ratio_roe, ratio_roa)
df_y <- df_data %>% select(date, security, label)

list_vars1 <- c('price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa')

# Create signals DataFrame
df_signals <- data.frame(date = unique(df_x$date[df_x$date >= start_test & df_x$date <= end_test]))
df_signals <- df_signals %>% arrange(date)

# Initialize columns for accuracy results
df_signals$acc_total <- NA
df_signals$acc_current <- NA

# Scale data using training set parameters
scaler <- function(x) (x - min(x)) / (max(x) - min(x)) * 2 - 1
df_x[list_vars1] <- lapply(df_x[list_vars1], scaler)

# Fit the Random Forest classifier with hyperparameter tuning
df_trainx <- df_x %>% filter(date < start_test)
df_trainy <- df_y %>% filter(date < start_test) %>% mutate(label = as.factor(label))

# Define a grid of hyperparameters for tuning
tune_grid <- expand.grid(mtry = c(1:5), 
                         splitrule = "gini", 
                         min.node.size = c(1, 5, 10))

# Train control settings
train_control <- trainControl(method = "cv", number = 10)

# Train Random Forest with tuning
clf_tuned <- train(label ~ ., 
                   data = df_trainx %>% select(all_of(list_vars1)), 
                   method = "ranger", 
                   trControl = train_control, 
                   tuneGrid = tune_grid)

print(paste("Best mtry:", clf_tuned$bestTune$mtry))

# Predictions and evaluation
for (i in seq_len(nrow(df_signals))) {
  
  if (verbose) print(paste('---> Processing date', df_signals$date[i]))
  
  # Test set
  df_testx <- df_x %>% filter(date == df_signals$date[i])
  df_testy <- df_y %>% filter(date == df_signals$date[i])
  
  # Predictions and accuracy
  pred_probs <- predict(clf_tuned, newdata = df_testx %>% select(all_of(list_vars1)), type = "prob")
  df_testy$signal <- pred_probs[, 2]
  df_testy$pred <- ifelse(pred_probs[, 2] > 0.5, 1, 0)
  
  # Calculate accuracies
  acc_total <- mean(df_y$label == predict(clf_tuned, newdata = df_x %>% select(all_of(list_vars1))))
  acc_current <- mean(df_testy$label == df_testy$pred)
  
  print(paste('---> Accuracy: total =', round(acc_total, 2), ', current date =', round(acc_current, 2)))
  
  # Store accuracy results
  df_signals$acc_total[i] <- acc_total
  df_signals$acc_current[i] <- acc_current
  
  df_signals[i, df_testy$security] <- df_testy$signal
}

# Create buy matrix for payoff plot
df_signals$`10th` <- apply(df_signals[df_sectors$security], 1, function(x) sort(x, decreasing = TRUE)[n_buys])
df_index <- df_signals[df_sectors$security] >= df_signals$`10th`

# Set 1 for top 10 strongest signals
df_buys <- as.data.frame(matrix(0, nrow = nrow(df_signals), ncol = length(df_sectors$security)))
colnames(df_buys) <- df_sectors$security
df_buys[df_index] <- 1

# Ensure exactly 10 buys per day
process_row <- function(row) {
  ones_indices <- which(row == 1)
  if(length(ones_indices) > 10) {
    row[ones_indices[11:length(ones_indices)]] <- 0
  }
  return(row)
}
df_buys <- t(apply(df_buys, 1, process_row))

# Add dates
df_buys <- cbind(date = df_signals$date, df_buys)

# Plot signals
ggplot(df_signals, aes(x = date)) + geom_line(aes(y = AAPL)) + ggtitle("AAPL Signals")
image(t(df_buys[-1]))  # Visualize buy signals (consider improving visualization)

# Create return matrix
df_returns <- read.csv('returns.csv')
df_returns$date <- as.Date(df_returns$date)
df_returns <- df_returns %>% filter(date >= start_test)
df_returns <- df_returns %>% pivot_wider(names_from = security, values_from = return1)

# Function to calculate and plot payoff
plot_payoff <- function(df_buys) {
  
  df <- df_buys[,-1]
  
  if (sum(rowSums(df) == 10) != nrow(df)) {
    stop("---> Must have exactly 10 buys each day")
  }
  
  df <- df * (1 / n_buys)  # Equally weighted
  
  df_payoff <- NULL
  arr_ret <- as.matrix(df_returns[-1] + 1)
  df_payoff$payoff <- diag(df %*% t(arr_ret - 1))
  df_payoff$tri <- cumprod(1 + df_payoff$payoff)
  
  df_payoff$date <- df_returns$date
  df_payoff <- df_payoff %>% as_tibble()
  
  payoff_result <- (tail(df_payoff$tri, 1) - 1) * 100
  print(paste("---> Payoff for these buys between", min(df_payoff$date), "and", max(df_payoff$date), "is", round(payoff_result, 2), "%"))
  
  return(df_payoff)
}

df_payoff <- plot_payoff(df_buys)
ggplot(df_payoff, aes(x = date, y = tri)) + geom_line() + ggtitle("Payoff Over Time")

print('---> R Script End')
