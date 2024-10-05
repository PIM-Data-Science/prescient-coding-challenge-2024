#  %% Header
# Prescient Coding Challenge 2024
# Authors: Timika Buthuram , Jesse Jabez Arendse
# Team Name: Cappuccinos

# %% Imports

import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

#  %% Functions
def predictionAvg(values):
    # Convert values to a numpy array if it's not already
    values = np.array(values)

    # Calcultae the average of values
    average = sum(values)/len(values)

    return average

def predictionDvt(values):
    # Get the derivative of the array
    derivative = np.diff(values)

    return derivative

def predictionLinearRegression(values):
    values = np.array(values)
    # Create an array of indices (e.g., 0, 1, 2, ...)
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next value
    next_index = np.array([[len(values)]])
    predicted_value = model.predict(next_index)
    return predicted_value

def predictionARIMA(values):
    # Fit the ARIMA model
    model = ARIMA(values, order=(1, 1, 1))  # Adjust the order as necessary
    model_fit = model.fit()

    # Forecast the next value
    forecast = model_fit.forecast(steps=1)
    return forecast

def predictionGPT(values):
    weights = None
    # Convert values to a numpy array if it's not already
    values = np.array(values)
    
    # If no weights are provided, use uniform weights
    if weights is None:
        weights = np.ones(len(values) - 1)
    
    # Ensure weights are a numpy array
    weights = np.array(weights)
    
    # Calculate the weighted differences (derivative approximation)
    weighted_diff = weights * np.diff(values)
    
    # Sum the weighted differences to compute the final prediction
    prediction_value = np.sum(weighted_diff)

    return prediction_value

def prediction(values):
    
    return predictionLinearRegression(values)

# %% Data importing
print('---> Python Script Start', t0 := datetime.now())

print('---> the parameters')

# training and test dates

start_train = datetime(2017, 1, 1      ,0,0,0)
end_train =   datetime(2023, 11, 30    ,0,0,0) # gap for embargo (no overlap between train and test)
start_test =  datetime(2024, 1, 1      ,0,0,0) # test set is this datasets 2024 data
end_test =    datetime(2024, 6, 30     ,0,0,0)

n_buys = 10
verbose = False

print('---> initial data set up')

# sector data
df_securities = pd.read_csv('data/data0.csv')['security']


# price and fin data
df_data = pd.read_csv('data/returns.csv')
df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())

df_x = df_data[['date', 'security', 'return1']].copy()
df_y = df_data[['date', 'security']].copy()


# Read the CSV file
df_returns = pd.read_csv('data/returns.csv')

# Convert the 'date' column to datetime format for grouping and plotting
df_returns['date'] = pd.to_datetime(df_returns['date'])

# Filter the data for the date range 2024-01-01 to 2024-06-28
start_test = '2024-01-01'
end_test = '2024-06-28'     # Not using test dates since, returns csv file ends on 2024-06-28
df_test = df_returns[(df_returns['date'] >= start_test) & (df_returns['date'] <= end_test)]

df_test_unique_dates = df_test['date'].unique()

number_of_dayys_to_predict = len(df_test_unique_dates)


# %% Security Predictions

prediction_matrix = []

date_format = "%Y-%m-%d"  # Year-Month-Day Hour:Minute:Second

for test_day in range(0 , len(df_test_unique_dates)): 
    converted_start = datetime.strptime(start_test, date_format).date()
    converted_end = datetime.strptime(end_test, date_format).date()
    
    start_window = converted_start + timedelta(days=test_day)  # Increment by one day
    end_window  = converted_end + timedelta(days=test_day)  # Increment by one day

    # Group the dataframe by the 'security' column
    grouped = df_data.groupby('security')
    array_of_security = []
    for security, group in grouped: 
        df_train = group[(group['date'] >= start_window) & (group['date'] < end_window)]
        array_of_security.append(df_train)

    security_predictions = []
    for security in array_of_security:
        predicted = prediction( security['return1'].values)
        security_predictions.append(predicted)
        
    top_ten = np.argsort(np.array(security_predictions))[-n_buys:]  # Get last 10 indices from sorted indices


    prediction_array_for_tomorrow = np.zeros(len(df_securities))
    for index in top_ten:
        prediction_array_for_tomorrow[index] = 1

    prediction_matrix.append(prediction_array_for_tomorrow)


# %% Get answers


# For each date, find the 10 largest return1 values and sum them
df_top10_sum = df_test.groupby('date').apply(lambda x: x.nlargest(10, 'return1')['return1'].sum()).reset_index(name='Top10_Sum')

# Plot the summed top 10 values for each date within the specified range
plt.figure(figsize=(10, 6))
plt.plot(df_top10_sum['date'], df_top10_sum['Top10_Sum'], marker='o', linestyle='-', color='b')
plt.title('Highest payoffs for 2024-01-01 to 2024-06-28')
plt.xlabel('Date')
plt.ylabel('Percentage returns')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig_aapl = px.line(df_top10_sum, x='date', y='Top10_sum')
fig_aapl.show()


# this is our final answer :)
df_buys = pd.DataFrame(np.array(prediction_matrix))



# %% 
