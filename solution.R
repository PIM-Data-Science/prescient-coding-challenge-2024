# Load necessary libraries
# Note: Since you prefer not to use additional packages, I'll use base R functions where possible.

# Load data
data0 <- read.csv('data0.csv', stringsAsFactors = FALSE)
data1 <- read.csv('data1.csv', stringsAsFactors = FALSE)
returns <- read.csv('returns.csv', stringsAsFactors = FALSE)

# Merge data
data <- merge(data1, data0, by = 'security', all.x = TRUE)
data$security <- as.factor(data$security)
data$sector <- as.factor(data$sector)

# Create the new sector variable
data$sector_grouped <- ifelse(data$sector %in% c('Information Technology', 'Staples', 'Consumer Discretionary'),
                              as.character(data$sector), 'other')

# Convert 'sector_grouped' to a factor
data$sector_grouped <- as.factor(data$sector_grouped)


# Convert date columns
data$date <- as.Date(data$date)
returns$date <- as.Date(returns$date)

# Sort data
data <- data[order(data$date, data$security), ]

# Split data into 70:30 ratio
unique_dates <- sort(unique(data$date))
split_index <- floor(length(unique_dates) * 0.7)
train_dates <- unique_dates[1:split_index]
test_dates <- unique_dates[(split_index + 1):length(unique_dates)]

train_data <- subset(data, date %in% train_dates)
test_data <- subset(data, date %in% test_dates)

# Prepare training data
train_data$label <- as.factor(train_data$label)
train_data$security <- as.factor(train_data$security)
train_data$sector <- as.factor(train_data$sector)

# Fit the models
full_model <- glm(label ~ sector_grouped + price + return30 + ratio_pe + ratio_pcf + ratio_de + ratio_roe + ratio_roa,
                  data = train_data, family = binomial)
null_model <- glm(label ~ 1, data = train_data, family = binomial)

# Stepwise selection (both directions)
stepwise_model <- step(null_model, scope = list(lower = null_model, upper = full_model),
                       direction = "both", trace = TRUE)

# Predict probabilities on test_data
predicted_probabilities <- predict(stepwise_model, newdata = test_data, type = "response")

# Add predicted probabilities to test_data
test_data$predicted_probability <- predicted_probabilities

# Create 'prob_data' data frame with necessary columns
prob_data <- data.frame(
  date = test_data$date,
  company = test_data$security,
  probability = test_data$predicted_probability
)

# Ensure 'date' and 'company' are appropriate types
prob_data$date <- as.Date(prob_data$date)
prob_data$company <- as.character(prob_data$company)

# Get list of all unique dates and companies
all_dates <- sort(unique(prob_data$date))
all_companies <- sort(unique(prob_data$company))

# Create complete grid of all date-company combinations
complete_grid <- expand.grid(date = all_dates, company = all_companies)

# Merge 'prob_data' with 'complete_grid' to ensure all combinations are present
prob_data_complete <- merge(complete_grid, prob_data, by = c("date", "company"), all.x = TRUE)

# Reshape data into matrix with dates as rows and companies as columns using 'xtabs'
prob_matrix <- xtabs(probability ~ date + company, data = prob_data_complete)
prob_matrix <- as.matrix(prob_matrix)

# Replace NA values with -Inf to handle missing probabilities
prob_matrix[is.na(prob_matrix)] <- 0

# Initialize the binary matrix
prob_binary_matrix <- matrix(0, nrow = nrow(prob_matrix), ncol = ncol(prob_matrix))
rownames(prob_binary_matrix) <- rownames(prob_matrix)
colnames(prob_binary_matrix) <- colnames(prob_matrix)

# Loop over each date to select top 10 companies
for (i in 1:nrow(prob_matrix)) {
  # Extract probabilities for the current date
  probs <- prob_matrix[i, ]
  
  # Get indices of top 10 probabilities
  top10_indices <- order(probs, decreasing = TRUE)[1:10]
  
  # Assign 1 to the top 10 companies in the binary matrix
  prob_binary_matrix[i, top10_indices] <- 1
}

# Verify that each row has exactly 10 ones
row_sums <- rowSums(prob_binary_matrix)
if (all(row_sums == 10)) {
  cat("All dates have 10 companies selected.\n")
} else {
  cat("Some dates do not have exactly 10 companies selected.\n")
}

# Now, integrate the payoff calculation

# Read returns data
df_returns <- read.csv('returns.csv', stringsAsFactors = FALSE)
df_returns$date <- as.Date(df_returns$date)

# Filter df_returns to include only dates in the test set
start_test <- min(test_data$date)
end_test <- max(test_data$date)
df_returns <- subset(df_returns, date >= start_test & date <= end_test)

# Reshape df_returns to wide format with dates as rows and securities as columns
# Assuming df_returns has columns: 'date', 'security', 'return1'
# Use 'xtabs' to reshape
returns_matrix <- xtabs(return1 ~ date + security, data = df_returns)
returns_matrix <- as.matrix(returns_matrix)

# Replace NA values with 0 (assuming no gain or loss if return is missing)
returns_matrix[is.na(returns_matrix)] <- 0

# Ensure that the dates and companies in returns_matrix match those in prob_binary_matrix
common_dates <- intersect(rownames(prob_binary_matrix), rownames(returns_matrix))
common_companies <- intersect(colnames(prob_binary_matrix), colnames(returns_matrix))

# Subset both matrices to the common dates and companies
prob_binary_matrix_sub <- prob_binary_matrix[common_dates, common_companies]
returns_matrix_sub <- returns_matrix[common_dates, common_companies]

# Ensure matrices are aligned
prob_binary_matrix_sub <- prob_binary_matrix_sub[order(rownames(prob_binary_matrix_sub)), order(colnames(prob_binary_matrix_sub))]
returns_matrix_sub <- returns_matrix_sub[order(rownames(returns_matrix_sub)), order(colnames(returns_matrix_sub))]

# Define the number of buys per day
n_buys <- 10

# Define the initial capital
initial_capital <- 100

# Calculate the payoff

# Calculate equal weights for each buy
df_buys_weighted <- prob_binary_matrix_sub * (1 / n_buys)  # Equally weighted

# Calculate daily returns
daily_returns <- rowSums(df_buys_weighted * returns_matrix_sub)

# Calculate cumulative return starting from initial capital
cumulative_return <- initial_capital * cumprod(1 + daily_returns)

# Prepare data frame for analysis
df_payoff <- data.frame(
  date = as.Date(rownames(prob_binary_matrix_sub)),
  daily_return = daily_returns,
  cumulative_return = cumulative_return
)

# Calculate total payoff over the period
total_payoff <- cumulative_return[length(cumulative_return)] - initial_capital
payoff_percentage <- (total_payoff / initial_capital) * 100

print(paste("---> Payoff for these buys between", min(df_payoff$date), "and", max(df_payoff$date), "is", round(payoff_percentage, 2), "%"))

# Plot cumulative return over time
plot(df_payoff$date, df_payoff$cumulative_return, type = "l", xlab = "Date", ylab = "Cumulative Return", main = "Strategy Cumulative Return Over Time")

