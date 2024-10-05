set.seed(123)

library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)  # For confusionMatrix

params <- list(
  train_start_date = as.Date("2017-01-01"),
  train_end_date = as.Date("2023-12-31"),
  test_start_date = as.Date("2024-01-01"),
  test_end_date = as.Date("2024-12-31"),
  max_buys = 10,
  verbose = FALSE
)

sector_data <- read.csv('data0.csv')
financial_data <- read.csv('data1.csv') %>%
  mutate(date = as.Date(date))

returns_data <- read.csv('returns.csv') %>%
  mutate(date = as.Date(date)) %>%
  # only use data before 2024
  filter(date >= params$test_start_date) %>%
  pivot_wider(names_from = security, values_from = return1)

features <- c('price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa')
features_data <- financial_data %>% select(date, security, all_of(features))
labels_data <- financial_data %>% select(date, security, label)

signal_dates <- financial_data %>%
  filter(date >= params$test_start_date & date <= params$test_end_date) %>%
  distinct(date) %>%
  arrange(date) %>%
  mutate(total_accuracy = NA, current_accuracy = NA)

normalize_features <- function(df) {
  df %>%
    mutate(across(all_of(features), ~ (.-min(.))/(max(.)-min(.)) * 2 - 1))
}

for (index in seq_len(nrow(signal_dates))) {
  
  training_filter <- features_data$date < signal_dates$date[index]
  
  training_features <- features_data %>%
    filter(training_filter) %>%
    filter(date != max(date)) %>%
    normalize_features()
  
  training_labels <- labels_data %>%
    filter(training_filter) %>%
    filter(date != max(date)) %>%
    mutate(label = as.factor(label))
  
  testing_features <- features_data %>%
    filter(date >= signal_dates$date[index]) %>%
    normalize_features()
  
  testing_labels <- labels_data %>%
    filter(date >= signal_dates$date[index])
  
  # logistic model
  if (index == 1) {
    model <- glm(label ~ ., data = training_features %>% mutate(label = training_labels$label), family = binomial)
  }
  
  predicted_probabilities <- predict(model, newdata = testing_features, type = "response")
  testing_labels <- testing_labels %>%
    mutate(signal_strength = predicted_probabilities,
           predicted_label = ifelse(predicted_probabilities > 0.5, 1, 0))
  
  current_day_labels <- testing_labels %>% filter(date == signal_dates$date[index])
  
  signal_dates$total_accuracy[index] <- mean(testing_labels$label == testing_labels$predicted_label)
  signal_dates$current_accuracy[index] <- mean(current_day_labels$label == current_day_labels$predicted_label)
  
  
  signal_dates[index, current_day_labels$security] <- current_day_labels$signal_strength
}

buy_thresholds <- apply(signal_dates[sector_data$security], 1, function(x) sort(x, decreasing = TRUE)[params$max_buys])
buy_indices <- signal_dates[sector_data$security] >= buy_thresholds

buy_matrix <- as.data.frame(ifelse(buy_indices, 1, 0), stringsAsFactors = FALSE)
colnames(buy_matrix) <- sector_data$security

buy_matrix <- t(apply(buy_matrix, 1, function(row) {
  if (sum(row) > params$max_buys) {
    row[order(row, decreasing = TRUE)[(params$max_buys + 1):length(row)]] <- 0
  }
  return(row)
}))

buy_matrix <- cbind(date = signal_dates$date, buy_matrix)

calculate_payoff <- function(buy_matrix, returns_data) {
  normalized_buys <- buy_matrix[,-1] * (1 / params$max_buys)
  returns_array <- as.matrix(returns_data[-1] + 1)
  
  total_payoff <- diag(normalized_buys %*% t(returns_array - 1))
  cumulative_growth <- cumprod(1 + total_payoff)
  
  payoffs <- tibble(date = returns_data$date, total_payoff = total_payoff, cumulative_growth = cumulative_growth)
  
  final_payoff_percentage <- (tail(payoffs$cumulative_growth, 1) - 1) * 100
  
  return(payoffs)
}

payoff_results <- calculate_payoff(buy_matrix, returns_data)

ggplot(payoff_results, aes(x = date, y = cumulative_growth)) +
  geom_line() +
  ggtitle("Predicted Payoff Over 6 Months") +
  xlab("Date") +
  ylab("Predicted Payoff")
