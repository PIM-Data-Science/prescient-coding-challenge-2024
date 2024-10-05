# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%
data0 = 'prescient-coding-challenge-2024/data/data0.csv'
data1 = 'prescient-coding-challenge-2024/data/data1.csv'
returns = 'prescient-coding-challenge-2024/data/returns.csv'

import pathlib

# check if file exists
for file in [data0, data1, returns]:
    if not pathlib.Path(file).exists():
        print(f'File {file} does not exist')

# %%
# Merge the sector information from data0 into data1
data0 = pd.read_csv(data0)
data1 = pd.read_csv(data1)
returns = pd.read_csv(returns)

data_combined = pd.merge(data1, data0, on='security')

# Merge the return data from returns.csv
data_combined = pd.merge(data_combined, returns, on=['date', 'security'])

# Check the combined data
data_combined.head()

# %%
# Step 2: Data Cleaning and Preprocessing

# Handling missing values
# Fill missing values in numeric columns with median
numeric_cols = ['price', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa', 'return1']
data_combined[numeric_cols] = data_combined[numeric_cols].fillna(data_combined[numeric_cols].median())

# Drop rows where labels are missing, as they are crucial for our predictions
data_combined = data_combined.dropna(subset=['label'])

# Sort data by date to maintain time order for future models
data_combined = data_combined.sort_values(by=['security', 'date'])

# Normalize the financial ratios (scaling to 0-1 range)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Apply scaling to financial ratio columns
data_combined[['ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']] = scaler.fit_transform(
    data_combined[['ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']]
)

# Check the cleaned and scaled data
data_combined.head()


# %%
# Step 3: Feature Engineering

# Moving averages (let's calculate 7-day and 30-day moving averages for price)
data_combined['ma_7'] = data_combined.groupby('security')['price'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
data_combined['ma_30'] = data_combined.groupby('security')['price'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

# Price momentum (percentage change in price over 1, 7, and 30 days)
data_combined['pct_change_1d'] = data_combined.groupby('security')['price'].transform(lambda x: x.pct_change(periods=1))
data_combined['pct_change_7d'] = data_combined.groupby('security')['price'].transform(lambda x: x.pct_change(periods=7))
data_combined['pct_change_30d'] = data_combined.groupby('security')['price'].transform(lambda x: x.pct_change(periods=30))

# Handling missing values after rolling operations by filling them with 0
data_combined[['pct_change_1d', 'pct_change_7d', 'pct_change_30d']] = data_combined[['pct_change_1d', 'pct_change_7d', 'pct_change_30d']].fillna(0)

# Encoding sector using one-hot encoding
data_combined = pd.get_dummies(data_combined, columns=['sector'])

# Check the final dataset after feature engineering
data_combined.head()


# %%
# Step 4: Model Training

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define the features and target variable
features = [
    'price', 'ma_7', 'ma_30', 'pct_change_1d', 'pct_change_7d', 'pct_change_30d', 
    'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa'
] + [col for col in data_combined.columns if 'sector_' in col]  # Include sector features

X = data_combined[features]
y = data_combined['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# %%
# Step 4: Model Training

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define the features and target variable
features = [
    'price', 'ma_7', 'ma_30', 'pct_change_1d', 'pct_change_7d', 'pct_change_30d', 
    'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa'
] + [col for col in data_combined.columns if 'sector_' in col]  # Include sector features

X = data_combined[features]
y = data_combined['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# %%
# Step 5: Portfolio Construction

# Get the predicted probabilities for each stock on the test set
y_pred_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (positive performance)

# Add the predicted probabilities to the test data
test_data = data_combined.iloc[X_test.index]
test_data['predicted_prob'] = y_pred_probs

# Initialize a DataFrame to store the portfolio for each day
portfolio = pd.DataFrame()

# For each day in the test set, select the top 10 stocks based on predicted probabilities
for date in test_data['date'].unique():
    daily_data = test_data[test_data['date'] == date]
    top_10_stocks = daily_data.nlargest(10, 'predicted_prob')  # Select top 10 stocks
    
    # Store the selected stocks and their details
    portfolio = pd.concat([portfolio, top_10_stocks])

# Evaluate the portfolio performance using Total Return Index (TRI)
portfolio['daily_return'] = portfolio['return1']

# Calculate cumulative return
portfolio['cumulative_return'] = (1 + portfolio['daily_return']).cumprod()

# Calculate TRI (Total Return Index) for the portfolio
portfolio['TRI'] = portfolio.groupby('date')['cumulative_return'].transform('last')

# Display portfolio performance
portfolio[['date', 'security', 'daily_return', 'cumulative_return', 'TRI']].head()


# %%
# Step 2: Create a buy matrix with 1s and 0s, ensuring each row sums to 10

# Create a pivot table for the buy matrix
buy_matrix = pd.pivot_table(
    data=portfolio,
    values='predicted_prob',
    index='date',
    columns='security',
    fill_value=0
)

# Replace probabilities with 1s for the top 10 stocks and 0s for the rest
def top_10_buy(row):
    # Get the indices of the 10 largest values
    top_10_indices = row.nlargest(10).index
    # Create a new row where only the top 10 are marked as 1
    new_row = pd.Series(0, index=row.index)
    new_row[top_10_indices] = 1
    return new_row

# Apply the top_10_buy function to each row
buy_matrix = buy_matrix.apply(top_10_buy, axis=1)

# Ensure each row sums to exactly 10
assert all(buy_matrix.sum(axis=1) == 10), "Each row should sum to exactly 10 buys."

# Display the buy matrix (1s for buys, 0s for don't buy)
buy_matrix.head()


# %%
import matplotlib.pyplot as plt

# Step 3: Generate the payoff chart (cumulative return over time)

# Plot cumulative return for the entire portfolio over time
plt.figure(figsize=(10, 6))
plt.plot(portfolio['date'], portfolio['cumulative_return'], label='Cumulative Return', color='blue')
plt.title('Portfolio Cumulative Return Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Optionally, you can also plot the TRI if desired
plt.figure(figsize=(10, 6))
plt.plot(portfolio['date'], portfolio['TRI'], label='Total Return Index (TRI)', color='green')
plt.title('Portfolio Total Return Index Over Time')
plt.xlabel('Date')
plt.ylabel('TRI')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# 


