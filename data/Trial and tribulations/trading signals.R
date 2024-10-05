# Set working directory
setwd("C:\\Users\\siyot\\OneDrive\\Desktop\\Khuliso\\prescient-coding-challenge-2024\\data")

# Load necessary data
data0 <- read.csv("data0.csv")
data1 <- read.csv("data1.csv")
returns <- read.csv("returns.csv")

# Define the training and testing periods
start_train <- as.Date("2017-01-01")
end_train <- as.Date("2023-12-30")
start_test <- as.Date("2024-01-01")
end_test <- as.Date("2024-06-30")

# Split the data into in-sample (training) and out-of-sample (testing) periods
in_sample_data <- data1[data1$date >= start_train & data1$date <= end_train, ]
out_of_sample_data <- data1[data1$date >= start_test & data1$date <= end_test, ]

# Calculate momentum for each security based on daily price changes (in-sample)
in_sample_data$momentum <- NA

# Loop through each unique security in the in-sample data
unique_securities <- unique(in_sample_data$security)
for (security in unique_securities) {
  # Get the price data for the current security
  security_data <- in_sample_data[in_sample_data$security == security, ]
  
  # Calculate momentum: (price / lag(price) - 1) * 100
  security_data$momentum <- c(NA, diff(security_data$price) / head(security_data$price, -1) * 100)
  
  # Assign the calculated momentum back to the original in-sample data
  in_sample_data[in_sample_data$security == security, "momentum"] <- security_data$momentum
}

# Generate buy signals for in-sample data
generate_buy_signals <- function(data) {
  # Initialize buy matrix with zeros
  buy_matrix <- matrix(0, nrow = length(unique(data$date)), ncol = length(unique(data$security)))
  rownames(buy_matrix) <- unique(data$date)
  colnames(buy_matrix) <- unique(data$security)
  
  # Loop through each date in the in-sample data
  for (date in unique(data$date)) {
    daily_data <- data[data$date == date, ]
    
    # Order securities by momentum and select top 10
    top_stocks <- head(daily_data[order(-daily_data$momentum), "security"], 10)
    
    # Mark buy signals in the matrix
    buy_matrix[date, top_stocks] <- 1
  }
  
  return(buy_matrix)
}


# Generate buy signals for in-sample data
buy_matrix_in_sample <- generate_buy_signals(in_sample_data)

# Function to calculate total payoff for in-sample
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

# Calculate the total payoff based on the buy matrix and returns for in-sample
total_payoff_in_sample <- calculate_payoff(buy_matrix_in_sample, returns)

# Plot the total payoff over time for in-sample
plot(as.Date(rownames(buy_matrix_in_sample)), total_payoff_in_sample, type = "l", 
     main = "Total Payoff Over Time (In-Sample)", xlab = "Date", ylab = "Cumulative Total Payoff", 
     col = "blue", lwd = 2)

# For out-of-sample testing, calculate momentum similarly
out_of_sample_data$momentum <- NA
for (security in unique_securities) {
  security_data <- out_of_sample_data[out_of_sample_data$security == security, ]
  security_data$momentum <- c(NA, diff(security_data$price) / head(security_data$price, -1) * 100)
  out_of_sample_data[out_of_sample_data$security == security, "momentum"] <- security_data$momentum
}

# Generate buy signals for out-of-sample data
buy_matrix_out_sample <- generate_buy_signals(out_of_sample_data)

# Calculate the total payoff for out-of-sample data
total_payoff_out_sample <- calculate_payoff(buy_matrix_out_sample, returns)

# Plot the total payoff over time for out-of-sample
plot(as.Date(rownames(buy_matrix_out_sample)), total_payoff_out_sample, type = "l", 
     main = "Total Payoff Over Time (Out-of-Sample)", xlab = "Date", ylab = "Cumulative Total Payoff", 
     col = "red", lwd = 2)

# Calculate returns for in-sample
initial_investment <- 1  # Amount invested in rand
total_value_in_sample <- initial_investment  # Start with the initial investment
n_dates_in_sample <- nrow(buy_matrix_in_sample)  # Number of days in the in-sample period
daily_returns_vector_in_sample <- numeric(n_dates_in_sample)  # Vector to store daily returns for in-sample

# Loop through each date to calculate daily returns based on buy signals for in-sample
for (i in 1:n_dates_in_sample) {
  date <- rownames(buy_matrix_in_sample)[i]  # Get the current date
  daily_buy_signals <- buy_matrix_in_sample[i, ]  # Get buy signals for the current date
  daily_returns <- returns[returns$date == date, ]
  
  # Calculate the return for the stocks bought
  for (security in colnames(buy_matrix_in_sample)[which(daily_buy_signals == 1)]) {
    security_return <- daily_returns[daily_returns$security == security, "return1"]
    if (length(security_return) > 0) {
      total_value_in_sample <- total_value_in_sample * (1 + security_return/10)
    }
  }
  
  # Store the cumulative value for that day
  daily_returns_vector_in_sample[i] <- total_value_in_sample
}

# Final total return from the initial investment for in-sample
total_return_in_sample <- total_value_in_sample - initial_investment

# Display results for in-sample
cat("Final value of the investment after the in-sample period: R", round(total_value_in_sample, 2), "\n")
cat("Total return from the investment (in-sample): R", round(total_return_in_sample, 2), "\n")

# Calculate returns for out-of-sample
total_value_out_sample <- initial_investment  # Start with the initial investment for out-of-sample
n_dates_out_sample <- nrow(buy_matrix_out_sample)  # Number of days in the out-of-sample period
daily_returns_vector_out_sample <- numeric(n_dates_out_sample)  # Vector to store daily returns for out-of-sample

# Loop through each date to calculate daily returns based on buy signals for out-of-sample
for (i in 1:n_dates_out_sample) {
  date <- rownames(buy_matrix_out_sample)[i]  # Get the current date
  daily_buy_signals <- buy_matrix_out_sample[i, ]  # Get buy signals for the current date
  daily_returns <- returns[returns$date == date, ]
  
  # Calculate the return for the stocks bought
  for (security in colnames(buy_matrix_out_sample)[which(daily_buy_signals == 1)]) {
    security_return <- daily_returns[daily_returns$security == security, "return1"]
    if (length(security_return) > 0) {
      total_value_out_sample <- total_value_out_sample * (1 + security_return)
    }
  }
  
  # Store the cumulative value for that day
  daily_returns_vector_out_sample[i] <- total_value_out_sample
}

# Final total return from the initial investment for out-of-sample
total_return_out_sample <- total_value_out_sample - initial_investment

# Display results for out-of-sample
cat("Final value of the investment after the out-of-sample period: R", round(total_value_out_sample, 2), "\n")
cat("Total return from the investment (out-of-sample): R", round(total_return_out_sample, 2), "\n")
