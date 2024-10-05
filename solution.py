# %%
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px
import plotly.graph_objects as go

print('---> Python Script Start', t0 := datetime.datetime.now())

# %%
print('---> the parameters')

# training and test dates
start_train = datetime.date(2017, 1, 1)
end_train = datetime.date(2023, 11, 30) # gap for embargo (no overlap between train and test)
start_test = datetime.date(2024, 1, 1) # test set is this dataset's 2024 data
end_test = datetime.date(2024, 6, 30)

n_buys = 10
verbose = False

# %%
print('---> initial data set up')

# Load sector data
df_sectors = pd.read_csv('data/data0.csv') 
# Load price and financial data
df_data = pd.read_csv('data/data1.csv')
df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())

# Select features (X) and target (y)
df_x = df_data[['date', 'security', 'price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']].copy()
df_y = df_data[['date', 'security', 'label']].copy()

# Define the list of features to be used for model training
list_vars1 = ['price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']

# %%
# Feature scaling (optional, but improves performance for some models)
#scaler = MinMaxScaler()

# Merge the features and labels for easier data splitting later
#df_x[list_vars1] = scaler.fit_transform(df_x[list_vars1])

# %%
# Split the data into training and testing sets
print('---> splitting train and test sets')

# Train on data between 2017 and 2023
train_mask = (df_x['date'] >= start_train) & (df_x['date'] <= end_train)
test_mask = (df_x['date'] >= start_test) & (df_x['date'] <= end_test)

# Get training and test data for both X (features) and y (labels)
X_train = df_x.loc[train_mask, list_vars1]
X_test = df_x.loc[test_mask, list_vars1]
y_train = df_y.loc[train_mask, 'label']
y_test = df_y.loc[test_mask, 'label']

# %%
print('---> training the Random Forest Classifier')

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest on the training data
rf_model.fit(X_train, y_train)
predicted_probabilities = rf_model.predict_proba(X_test)
loops = int(len(predicted_probabilities) / 100)

days = [f'Day {i+1}' for i in range(loops)]  # Adjust the range as needed for the number of days
stock_names = df_sectors['security'].tolist()  # Extract security names

# Initialize the DataFrame with zeros
results_df = pd.DataFrame(0, index=stock_names, columns=days)

for i in range(loops):
    probability = predicted_probabilities[i::loops]  # Use slicing directly on the NumPy array
    
    # Extract the second values (class probabilities for class 1)
    second_values = probability[:, 1]  # No need for .astype(float) since it's already float
    top_10_indices = np.argsort(second_values)[-10:][::-1]
    top_10_securities = []

    for idx in top_10_indices:
        top_10_securities.append(df_sectors.iloc[idx])
    top_10_securities_df = pd.DataFrame(top_10_securities)
    security_list = top_10_securities_df['security'].tolist() 
    # Step 3: Update the DataFrame
    day_label = f'Day {i + 1}'
    for security in stock_names:
        if security in security_list:
            results_df.loc[security, day_label] = 1  # Place a 1 if in top 10
        else:
            results_df.loc[security, day_label] = 0
    
# Step 4: Save the DataFrame to a CSV file
results_df.to_csv('data/returns.csv')

# Optionally print the DataFrame
print(results_df)