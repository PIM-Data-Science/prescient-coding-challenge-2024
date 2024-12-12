# Load necessary data
setwd("C:\\Users\\siyot\\OneDrive\\Desktop\\Khuliso\\prescient-coding-challenge-2024\\data")
data0 <- read.csv("data0.csv")
data1 <- read.csv("data1.csv")
returns <- read.csv("returns.csv")

# Calculate momentum for each security based on daily price changes
data1$momentum <- NA
unique_securities <- unique(data1$security)

for (security in unique_securities) {
  security_data <- data1[data1$security == security, ]
  security_data$momentum <- c(NA, diff(security_data$price) / head(security_data$price, -1) * 100)
  data1[data1$security == security, "momentum"] <- security_data$momentum
}

# Determine the last month in the dataset for splitting
last_date <- max(data1$date)
last_month_start <- as.Date(paste0(format(last_date, "%Y-%m"), "-01"))
last_month_end <- as.Date(paste0(format(last_date, "%Y-%m"), "-01")) + months(1) - days(1)

# Split the data into in-sample and out-of-sample periods
in_sample_data <- data1[data1$date < last_month_start, ]
out_of_sample_data <- data1[data1$date >= last_month_start, ]

# Calculate returns for the in-sample period (assuming R1 investment)
generate_buy_signals <- function(data) {
  buy_matrix <- matrix(0, nrow = length(unique(data$date)), ncol = length(unique(data$security)))
  rownames(buy_matrix) <- unique(data$date)
  colnames(buy_matrix) <- unique(data$security)
  
  for (date in unique(data$date)) {
    daily_data <- data[data$date == date, ]
    top_stocks <- head(daily_data[order(-daily_data$momentum), "security"], 10)
    buy_matrix[date, top_stocks] <- 1
  }
  
  return(buy_matrix)
}

# Generate buy signals for in-sample period
buy_matrix_in_sample <- generate_buy_signals(in_sample_data)

# Calculate in-sample return
initial_investment <- 1  # Amount invested in rand
total_value_in_sample <- initial_investment  # Start with the initial investment
n_dates_in_sample <- nrow(buy_matrix_in_sample)  # Number of days in the in-sample period
daily_returns_vector_in_sample <- numeric(n_dates_in_sample)  # Vector to store daily returns

for (i in 1:n_dates_in_sample) {
  date <- rownames(buy_matrix_in_sample)[i]
  daily_buy_signals <- buy_matrix_in_sample[i, ]
  daily_returns <- returns[returns$date == date, ]
  
  for (security in colnames(buy_matrix_in_sample)[which(daily_buy_signals == 1)]) {
    security_return <- daily_returns[daily_returns$security == security, "return1"]
    
    if (length(security_return) > 0) {
      total_value_in_sample <- total_value_in_sample * (1 + security_return)
    }
  }
  
  daily_returns_vector_in_sample[i] <- total_value_in_sample
}

# Final total return from the in-sample investment
total_return_in_sample <- total_value_in_sample - initial_investment
cat("Final value of the in-sample investment after the period: R", round(total_value_in_sample, 2), "\n")
cat("Total return from the in-sample investment: R", round(total_return_in_sample, 2), "\n")

# Generate buy signals for out-of-sample period
buy_matrix_out_of_sample <- generate_buy_signals(out_of_sample_data)

# Calculate out-of-sample return
total_value_out_of_sample <- initial_investment  # Start with the initial investment
n_dates_out_of_sample <- nrow(buy_matrix_out_of_sample)  # Number of days in the out-of-sample period
daily_returns_vector_out_of_sample <- numeric(n_dates_out_of_sample)  # Vector to store daily returns

for (i in 1:n_dates_out_of_sample) {
  date <- rownames(buy_matrix_out_of_sample)[i]
  daily_buy_signals <- buy_matrix_out_of_sample[i, ]
  daily_returns <- returns[returns$date == date, ]
  
  for (security in colnames(buy_matrix_out_of_sample)[which(daily_buy_signals == 1)]) {
    security_return <- daily_returns[daily_returns$security == security, "return1"]
    
    if (length(security_return) > 0) {
      total_value_out_of_sample <- total_value_out_of_sample * (1 + security_return)
    }
  }
  
  daily_returns_vector_out_of_sample[i] <- total_value_out_of_sample
}

# Final total return from the out-of-sample investment
total_return_out_of_sample <- total_value_out_of_sample - initial_investment
cat("Final value of the out-of-sample investment after the period: R", round(total_value_out_of_sample, 2), "\n")
cat("Total return from the out-of-sample investment: R", round(total_return_out_of_sample, 2), "\n")

# Optional: Plot total value over time for both in-sample and out-of-sample
dates_in_sample <- as.Date(rownames(buy_matrix_in_sample))
dates_out_of_sample <- as.Date(rownames(buy_matrix_out_of_sample))

plot(dates_in_sample, daily_returns_vector_in_sample, type = "l", col = "blue", lwd = 2,
     main = "Total Value of Investment Over Time (In-Sample and Out-of-Sample)", 
     xlab = "Date", ylab = "Total Value (R)", ylim = c(0, max(c(daily_returns_vector_in_sample, daily_returns_vector_out_of_sample), na.rm = TRUE)))
lines(dates_out_of_sample, daily_returns_vector_out_of_sample, col = "red", lwd = 2)
legend("topleft", legend = c("In-Sample", "Out-of-Sample"), col = c("blue", "red"), lwd = 2)
