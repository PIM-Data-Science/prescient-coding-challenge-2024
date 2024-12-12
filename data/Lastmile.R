# Load necessary libraries
library(dplyr)

# Define date ranges
start_train <- as.Date("2017-01-01")
end_train <- as.Date("2023-12-30")
start_test <- as.Date("2024-01-01")
end_test <- as.Date("2024-06-30")

# Set working directory
setwd("C:\\Users\\siyot\\OneDrive\\Desktop\\Khuliso\\prescient-coding-challenge-2024\\data")

# Load data
data0 <- read.csv("data0.csv")
data1 <- read.csv("data1.csv")
returns <- read.csv("returns.csv")

# Initialize momentum column in data1
data1$momentum <- NA

# Loop through each unique security to calculate momentum
unique_securities <- unique(data1$security)
for (security in unique_securities) {
  # Get the price data for the current security
  security_data <- data1[data1$security == security, ]
  
  # Calculate momentum: (price / lag(price) - 1) * 100
  security_data$momentum <- c(NA, diff(security_data$price) / head(security_data$price, -1) * 100)
  
  # Assign the calculated momentum back to the original data1
  data1[data1$security == security, "momentum"] <- security_data$momentum
}

# Filter training data
train_data <- data1[data1$date >= start_train & data1$date <= end_train, ]
# Filter testing data
test_data <- data1[data1$date >= start_test & data1$date <= end_test, ]

# Function to generate buy signals for a given data
generate_buy_signals <- function(data) {
  # Initialize buy matrix with zeros
  buy_matrix <- matrix(0, nrow = length(unique(data$date)), ncol = length(unique(data$security)))
  rownames(buy_matrix) <- unique(data$date)
  colnames(buy_matrix) <- unique(data$security)
  
  # Loop through each date in the data
  for (date in unique(data$date)) {
    daily_data <- data[data$date == date, ]
    
    # Check if daily_data has any records for that date
    if (nrow(daily_data) > 0) {
      # Order securities by momentum and select top 10
      top_stocks <- head(daily_data[order(-daily_data$momentum), "security"], 10)
      
      # Mark buy signals in the matrix
      buy_matrix[date, top_stocks] <- 1
    }
  }
  
  return(buy_matrix)
}

# Generate buy signals for training data
train_buy_matrix <- generate_buy_signals(train_data)

# Generate buy signals for testing data
test_buy_matrix <- generate_buy_signals(test_data)

# Function to calculate total payoff
calculate_payoff <- function(buy_matrix, returns) {
  # Initialize a payoff vector
  payoff <- rep(0, nrow(buy_matrix))
  
  # Loop through each date in the buy_matrix
  for (i in 1:nrow(buy_matrix)) {
    date <- rownames(buy_matrix)[i]
    
    # Get the returns for the current date
    daily_returns <- returns[returns$date == date, ]
    
    # Check if there are any returns for the given date
    if (nrow(daily_returns) > 0) {
      # Calculate the payoff for that day
      for (j in 1:ncol(buy_matrix)) {
        stock <- colnames(buy_matrix)[j]
        if (buy_matrix[i, stock] == 1) {
          # Check if the stock exists in daily_returns to avoid NA issues
          stock_return <- daily_returns[daily_returns$security == stock, "return1"]
          if (length(stock_return) > 0) { # Ensure we have a return value
            payoff[i] <- payoff[i] + stock_return
          }
        }
      }
    }
  }
  
  # Return cumulative payoff
  return(cumsum(payoff))
}

# Calculate the total payoff based on the training buy matrix and returns
train_total_payoff <- calculate_payoff(train_buy_matrix, returns)

# Calculate the total payoff based on the testing buy matrix and returns
test_total_payoff <- calculate_payoff(test_buy_matrix, returns)

# Plot the total payoff over time for both training and testing
plot(as.Date(rownames(train_buy_matrix)), train_total_payoff, type = "l", 
     main = "Total Payoff Over Time (Training Period)", xlab = "Date", 
     ylab = "Cumulative Total Payoff", col = "blue", lwd = 2)

plot(as.Date(rownames(test_buy_matrix)), test_total_payoff, type = "l", 
     main = "Total Payoff Over Time (Testing Period)", xlab = "Date", 
     ylab = "Cumulative Total Payoff", col = "red", lwd = 2)

# Assuming the following variables are already defined and available:
# - buy_matrix: The matrix with buy signals (1s and 0s)
# - returns: The dataframe containing daily returns (with columns: date, security, return1)

# Initialize total investment and total value
initial_investment <- 1  # Amount invested in rand
total_value <- initial_investment  # Start with the initial investment
n_dates_test <- nrow(test_buy_matrix)  # Number of days in the test period
daily_returns_vector <- numeric(n_dates_test)  # Vector to store daily returns

# Loop through each date in the test period to calculate daily returns based on buy signals
for (i in 1:n_dates_test) {
  date <- rownames(test_buy_matrix)[i]  # Get the current date
  daily_buy_signals <- test_buy_matrix[i, ]  # Get buy signals for the current date
  
  # Get the corresponding daily returns for the stocks bought
  daily_returns <- returns[returns$date == date, ]
  
  # Calculate the return for the stocks bought
  for (security in colnames(test_buy_matrix)[which(daily_buy_signals == 1)]) {
    # Find the return for the current security
    security_return <- daily_returns[daily_returns$security == security, "return1"]
    
    # If the security return exists, calculate the impact on total value
    if (length(security_return) > 0) {
      # Update total value based on the daily return
      total_value <- total_value * (1 + security_return / 10)  # Each stock is 10% of the portfolio
    }
  }
  
  # Store the cumulative value for that day
  daily_returns_vector[i] <- total_value
}

# Final total return from the initial investment for testing period
total_return_test <- total_value - initial_investment

# Display results
cat("Final value of the investment after the test period: R", round(total_value, 2), "\n")
cat("Total return from the investment: R", round(total_return_test, 2), "\n")
