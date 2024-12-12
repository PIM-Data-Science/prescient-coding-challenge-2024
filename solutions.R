#BRAVURA


# Loading essential libraries
library(data.table)
library(dplyr)
library(lubridate)
library(tidyr)
library(ggplot2)
library(xgboost)
library(caret)

# Setting the working directory
setwd("C:/Users/ellio/Downloads")

# Loading the data
data1 <- fread("data1.csv")
data0 <- fread("data0.csv")
returns_data <- fread("returns.csv")

# Merging data1 and returns_data on 'date' and 'security'
merged_data <- merge(data1, returns_data, by = c("date", "security"), all.x = TRUE)

# Merging the result with data0 on 'security'
merged_data <- merge(merged_data, data0[, .(security, sector)], by = "security", all.x = TRUE)

# Converting data types
merged_data <- merged_data %>%
  mutate(
    date = as.Date(date),
    security = as.factor(security),
    sector = as.factor(sector),
    label = as.factor(label)
  )

# Removing duplicates
merged_data <- merged_data %>% distinct()

# Identifying missing values
missing_values <- colSums(is.na(merged_data))
print("Missing values per column:")
print(missing_values)

# Time series plot of price for all stocks on the same plot
ggplot(merged_data, aes(x = date, y = price, color = security, group = security)) +
  geom_line(alpha = 0.8) +  
  labs(title = "Price Time Series for All Stocks", x = "Date", y = "Price") +
  theme_minimal() +
  theme(legend.position = "none")  

# Identifying numeric features
numeric_features <- c(
  "price", "return1", "return30", "ratio_pe", "ratio_pcf", "ratio_de",
  "ratio_roe", "ratio_roa"
)

# Scaling numerical features
scaled_features <- merged_data %>%
  select(all_of(numeric_features)) %>%
  mutate_all(scale) %>%
  as.data.frame()

# Combining scaled features back into the dataset
merged_data <- merged_data %>%
  select(-all_of(numeric_features)) %>%
  bind_cols(scaled_features)

# Splitting the data
train_data <- merged_data %>% filter(date < as.Date("2024-01-01"))
test_data <- merged_data %>% filter(date >= as.Date("2024-01-01") & date <= as.Date("2024-06-30"))

# Preparing training data
train_matrix <- train_data %>%
  select(-date, -security, -sector, -label) %>%
  as.matrix()
train_label <- as.numeric(as.character(train_data$label))

# Preparing testing data
test_matrix <- test_data %>%
  select(-date, -security, -sector, -label) %>%
  as.matrix()

# Creating DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

# Setting parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 6
)

# Training the model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)

# Generating predictions
xgb_predictions <- predict(xgb_model, test_matrix)

# Adding predictions to test_data
test_data <- test_data %>%
  mutate(predictions = xgb_predictions)

# Creating buy signals: select top 10 stocks per day
test_data <- test_data %>%
  group_by(date) %>%
  arrange(desc(predictions)) %>%
  mutate(buy_signal = ifelse(row_number() <= 10, 1, 0)) %>%
  ungroup()

# Generating the buy matrix
buy_matrix <- test_data %>%
  select(date, security, buy_signal) %>%
  pivot_wider(names_from = security, values_from = buy_signal, values_fill = 0)

# Viewing the buy matrix
head(buy_matrix)

# Calculating portfolio performance
payoff_matrix <- test_data %>%
  filter(buy_signal == 1) %>%
  group_by(date) %>%
  summarise(total_payoff = sum(return1, na.rm = TRUE))

# Printing the payoff matrix for inspection
print(payoff_matrix)

# Plotting the payoff over time
ggplot(payoff_matrix, aes(x = date, y = total_payoff)) +
  geom_line(color = "blue") +
  labs(title = "Total Payoff Over Time", x = "Date", y = "Total Payoff") +
  theme_minimal()
