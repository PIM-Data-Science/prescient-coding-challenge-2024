# Load necessary libraries
library(dplyr)
library(randomForest)
library(ggplot2)
library(caret)
library(xgboost)

# Load data
sectors <- read.csv("data/data0.csv")
data <- read.csv("data/data1.csv")
returns <- read.csv("data/returns.csv")

# Convert date columns
data$date <- as.Date(data$date)
returns$date <- as.Date(returns$date)

# Feature engineering 
# Relationship between the market’s valuation of a company's cash flow and its efficiency
data <- data %>%
  mutate(
    pe_to_roa = ratio_pe / ratio_roa,
    pcf_to_roe = ratio_pcf / ratio_roe
  )

# Split data into training and test sets
train_data <- data %>% filter(date >= "2017-01-01" & date <= "2023-11-30")
test_data <- data %>% filter(date >= "2024-01-01" & date <= "2024-06-30")

# Convert the label to a factor for classification
train_data$label <- as.factor(train_data$label)
test_data$label <- as.factor(test_data$label)

# Train the Random Forest model (Model 1)
set.seed(24)
rf_model <- randomForest(label ~ price + return30 + pe_to_roa + pcf_to_roe,
                         data = train_data,
                         ntree = 20)

# Train the Logistic Regression model (Model 2)
logistic_model <- glm(label ~ price + return30 + pe_to_roa + pcf_to_roe,
                      data = train_data,
                      family = binomial)

# Train the Gradient Boosting model (Model 3)
# Prepare the data for xgboost (XGBoost requires a matrix format for features and labels)
train_matrix <- as.matrix(train_data %>% select(price, return30, pe_to_roa, pcf_to_roe))
train_label <- as.numeric(as.character(train_data$label))
xgb_model <- xgboost(data = train_matrix, label = train_label, nrounds = 50, objective = "binary:logistic")




# Get predictions for the test data from Random Forest (Model 1)
test_data$rf_predicted <- predict(rf_model, newdata = test_data, type = "response")

# Get predictions for the test data from Logistic Regression (Model 2)
test_data$logistic_predicted <- predict(logistic_model, newdata = test_data, type = "response")

# Get predictions for the test data from Gradient Boosting (Model 3)
test_matrix <- as.matrix(test_data %>% select(price, return30, pe_to_roa, pcf_to_roe))
test_data$xgb_predicted <- predict(xgb_model, newdata = test_matrix)




# Apply majority voting (if at least two out of three models predict a buy, we assign a buy = 1)
## Convert rf_predicted from factor to numeric (0 and 1)

test_data$rf_predicted <- as.numeric(as.character(test_data$rf_predicted))

test_data <- test_data %>%
  mutate(vote_predicted = ifelse((rf_predicted > 0.5) + (logistic_predicted > 0.5) + (xgb_predicted > 0.5) >= 2, 1, 0))


# For each day, select the top 10 stocks based on the voting mechanism
buy_matrix <- test_data %>%
  group_by(date) %>%
  arrange(desc(vote_predicted)) %>%
  slice_head(n = 10) %>%
  mutate(buy = 1) %>%
  ungroup()

# Create a wide buy matrix (1s and 0s) with spread()
buy_matrix <- buy_matrix %>%
  spread(key = security, value = buy, fill = 0)

# Check structure of buy_matrix
str(buy_matrix)



# Merge the buy matrix with returns data on 'date'
buy_matrix_with_returns <- merge(buy_matrix, returns, by = "date", all.x = TRUE)

# Convert all buy signal columns to numeric
buy_matrix_with_returns <- buy_matrix_with_returns %>%
  mutate(across(-c(date, return1), as.numeric))

# Multiply buy signals by returns to calculate daily payoff
buy_matrix_with_returns <- buy_matrix_with_returns %>%
  mutate(daily_payoff = rowSums(select(buy_matrix_with_returns, -date, -return1) * buy_matrix_with_returns$return1, na.rm = TRUE))

# Aggregate the payoff for each day
daily_payoff <- buy_matrix_with_returns %>%
  group_by(date) %>%
  summarise(total_payoff = sum(daily_payoff, na.rm = TRUE))

# Cumulative payoff calculation
daily_payoff <- daily_payoff %>%
  mutate(cumulative_payoff = cumsum(total_payoff))

# Plot the cumulative payoff over time
ggplot(daily_payoff, aes(x = date, y = cumulative_payoff)) +
  geom_line() +
  labs(title = "Cumulative Payoff Over Time", x = "Date", y = "Cumulative Payoff")


