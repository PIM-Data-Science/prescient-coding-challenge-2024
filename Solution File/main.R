setwd("C:/prescient-coding-challenge-2024/Solution File")
traindat = read.csv("traindat.csv")
traindat$industry = as.factor(traindat$industry)
traindat$X10dayrollingstddev = NULL
equities = unique(traindat$Company)

library(h2o)
h2o.init()

N = dim(traindat)[1]
set.seed(2024)
set = sample(1:N,floor(0.5*N), replace = FALSE)
data_train = as.h2o(traindat[set,])
data_val   = as.h2o(traindat[-set,])

m = ncol(traindat)

model = h2o.deeplearning(x = 4:m, 
                         y =3,
                         training_frame = data_train,
                         validation_frame = data_val,
                         standardize = TRUE,
                         hidden = c(5,5),
                         activation = 'Tanh',
                         distribution = 'gaussian',
                         loss = 'Quadratic',
                         l2 = 0.001,
                         rate = 0.01,
                         adaptive_rate = FALSE,
                         epochs = 1000,
                         reproducible = TRUE,
                         seed = 2024
                         )

# Load the new dataframe
testdat = read.csv("testdat.csv")
testdat$industry = as.factor(testdat$industry)

# Convert the new dataframe to an H2O object
testdat_h2o = as.h2o(testdat)

# Make predictions using the trained model
predictions = h2o.predict(model, testdat_h2o)

# Convert predictions back to R data frame if needed
predictions_df = as.data.frame(predictions)

# View predictions
print(head(predictions_df))

k = length(equities)

predicted_values <- predictions_df$predict

dailyreturns = matrix(data = predicted_values, ncol = k, byrow = TRUE)

# Initialize an empty matrix to store binary results
binary_returns_matrix <- matrix(0, nrow = nrow(dailyreturns), ncol = ncol(dailyreturns))

# Loop through each row to fill binary_returns_matrix
for (i in 1:nrow(dailyreturns)) {
  top_10_indices <- order(dailyreturns[i, ], decreasing = TRUE)[1:10]  # Get indices of the top 10 returns
  binary_returns_matrix[i, top_10_indices] <- 0.1  # Set the top 10 returns to 0.1
}

# Convert to data frame if needed
binary_returns_df <- as.data.frame(binary_returns_matrix)

# Confirm the dimensions of the new binary dataframe
print(dim(binary_returns_df))  # Should be (74542, k)

# View the first few rows to confirm correctness
print(head(binary_returns_df))

# Ensure each row sums to 1
row_sums <- rowSums(binary_returns_df)
print(head(row_sums))  # Should be all 1s

testrets = matrix(data = testdat$return1, 
                  ncol = k, byrow = TRUE)
q = nrow(binary_returns_df)
tradingretsmat = matrix(NA,nrow = q) 

for (i in 1:q) {
  tradingretsmat[i,] = as.matrix(binary_returns_df[i,])%*%as.matrix(testrets[i,])
}

returns = cumprod(tradingretsmat+1)
l = length(returns)

plot(y = returns,x = 1:l , type = "l", ylab = "Time")
print(binary_returns_df)
