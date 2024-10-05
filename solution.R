# Load necessary libraries
rm(list=ls())
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)
library(h2o)


# Initialize the H2O cluster
h2o.init(nthreads = -1)

# Define parameters
start_train <- as.Date("2017-01-01")
end_train <- as.Date("2023-11-30")
start_test <- as.Date("2024-01-01")
end_test <- as.Date("2024-06-30")

n_buys <- 10
verbose <- FALSE



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



# Load necessary libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)
library(h2o)

print('---> R Script Start')

# Initialize the H2O cluster
h2o.init(nthreads = -1)

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
  
  # Scale data (optional, h2o handles scaling internally)
  for (col in list_vars1) {
    scaler <- function(x) (x - min(x)) / (max(x) - min(x)) * 2 - 1
    df_trainx[[col]] <- scaler(df_trainx[[col]])
    df_testx[[col]] <- scaler(df_testx[[col]])
  }
  
  df_trainy <- df_trainy %>% mutate(label = as.factor(label)) 
  
  # Convert to h2o frames
  train_h2o <- as.h2o(cbind(df_trainx[list_vars1], label = df_trainy$label))
  test_h2o <- as.h2o(df_testx[list_vars1])
  
  # Build and train the H2O deep learning model trains every 10 days 
  if (i %in% seq(1, nrow(df_signals), 10)) {
    clf <- h2o.deeplearning(
      x = list_vars1,
      y = "label",
      training_frame = train_h2o,
      activation = "RectifierWithDropout",
      hidden = c(64, 32),
      epochs = 20,
      input_dropout_ratio = 0.2,
      hidden_dropout_ratios = c(0.5, 0.5),
      balance_classes = TRUE,
      stopping_metric = "misclassification",
      stopping_rounds = 5,
      verbose = verbose
    )
  }
  
  # Predictions and accuracy
  pred_probs <- h2o.predict(clf, newdata = test_h2o)
  pred_probs <- as.data.frame(pred_probs)
  df_testy$signal <- pred_probs$p1
  df_testy$pred <- ifelse(pred_probs$p1 > 0.5, 1, 0)
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

# Continue with your signal plotting, buy matrix, and payoff calculations

# Shutdown H2O cluster after completion
h2o.shutdown(prompt = FALSE)

print('---> R Script End')

# Continue with your signal plotting, buy matrix, and payoff calculations

# Shutdown H2O cluster after completion
h2o.shutdown(prompt = FALSE)


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


