import numpy as np
import pandas as pd
import datetime
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

print('---> Python Script Start', t0 := datetime.datetime.now())

# Set parameters
start_train = datetime.date(2017, 1, 1)
end_train = datetime.date(2023, 11, 30)  # gap for embargo (no overlap between train and test)
start_test = datetime.date(2024, 1, 1)  # test set is this dataset's 2024 data
end_test = datetime.date(2024, 6, 30)

n_buys = 10
verbose = False

print('---> Initial data set up')

# Sector data
df_sectors = pd.read_csv('data/data0.csv')

# Price and financial data
df_data = pd.read_csv('data/data1.csv')
df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())

df_x = df_data[['date', 'security', 'price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']].copy()
df_y = df_data[['date', 'security', 'label']].copy()

list_vars1 = ['price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']

# Create signals DataFrame for the specified date range
df_signals = pd.DataFrame(data={'date': df_x.loc[(df_x['date'] >= start_test) & (df_x['date'] <= end_test), 'date'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='date', inplace=True)


# Define a smaller hyperparameter grid for Random Forest
param_distributions = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

# Iterating over each date in df_signals
for i in range(len(df_signals)):
    if verbose: print('---> doing', df_signals.loc[i, 'date'])

    # Training set
    df_trainx = df_x[df_x['date'] < df_signals.loc[i, 'date']].copy()
    df_trainx.drop(labels=df_trainx[df_trainx['date'] == df_trainx['date'].max()].index, inplace=True)

    df_trainy = df_y[df_y['date'] < df_signals.loc[i, 'date']].copy()
    df_trainy.drop(labels=df_trainy[df_trainy['date'] == df_trainy['date'].max()].index, inplace=True)

    # Test set
    df_testx = df_x[df_x['date'] >= df_signals.loc[i, 'date']].copy()
    df_testy = df_y[df_y['date'] >= df_signals.loc[i, 'date']].copy()

    # Scale and store scaling objects
    dict_scaler = {}
    for col in list_vars1:
        dict_scaler[col] = MinMaxScaler(feature_range=(-1, 1))
        df_trainx[col] = dict_scaler[col].fit_transform(np.array(df_trainx[col]).reshape((len(df_trainx[col]), 1)))[:, 0]
        df_testx[col] = dict_scaler[col].transform(np.array(df_testx[col]).reshape((len(df_testx[col]), 1)))[:, 0]

    # Fit a classifier using Randomized Search if i == 0
    if i == 0:
        rf = RandomForestClassifier(random_state=0)
        randomized_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                               n_iter=10, cv=2, scoring='accuracy', n_jobs=-1, verbose=2, random_state=0)
        randomized_search.fit(np.array(df_trainx[list_vars1]), df_trainy['label'].values)

        # Best parameters
        print("Best Parameters from Randomized Search:", randomized_search.best_params_)
        clf = randomized_search.best_estimator_  # Use the best estimator

    # Predict and calculate accuracy
    df_testy['signal'] = clf.predict_proba(np.array(df_testx[list_vars1]))[:, 1]
    df_testy['pred'] = clf.predict(np.array(df_testx[list_vars1]))
    df_testy['count'] = 1

    df_current = df_testy[df_testy['date'] == df_signals.loc[i, 'date']]

    acc_total = (df_testy['label'] == df_testy['pred']).sum() / len(df_testy)
    acc_current = (df_current['label'] == df_current['pred']).sum() / len(df_current)

    print('---> accuracy test set', round(acc_total, 2), ', accuracy current date', round(acc_current, 2))

    # Add accuracy and signal to dataframe
    df_signals.loc[i, 'acc_total'] = acc_total
    df_signals.loc[i, 'acc_current'] = acc_current

    df_signals.loc[i, df_current['security'].values] = df_current['signal'].values

# Create buy matrix for payoff plot
df_signals['10th'] = df_signals[df_sectors['security'].values].apply(lambda x: sorted(x)[len(df_sectors) - n_buys - 1], axis=1)
df_index = pd.DataFrame(np.array(df_signals[df_sectors['security'].values]) > np.array(df_signals['10th']).reshape((len(df_signals), 1)))

# Set 1 for top 10 strongest signals
df_buys = pd.DataFrame()
df_buys[df_sectors['security'].values] = np.zeros((len(df_signals), len(df_sectors)))
df_buys[df_index.values] = 1
df_buys.insert(0, 'date', df_signals['date'].copy())

# Check some signal plots
fig_aapl = px.line(df_signals, x='date', y='AAPL')
fig_aapl.show()

fig_pixel = px.imshow(np.array(df_buys[df_sectors['security'].values]))
fig_pixel.show()

# Create return matrix
df_returns = pd.read_csv('data/returns.csv')
df_returns['date'] = pd.to_datetime(df_returns['date']).apply(lambda d: d.date())
df_returns = df_returns[df_returns['date'] >= start_test]
df_returns = df_returns.pivot(index='date', columns='security', values='return1')

def plot_payoff(df_buys):
    df = df_buys.copy()

    # Ensure exactly 10 buys each day, only for numeric columns
    assert (df.drop(columns='date').sum(axis=1) == 10).all(), '---> must have exactly 10 buys each day'

    # Prepare the payoff DataFrame
    df_payoff = df[['date']].copy()
    del df['date']
    arr_buys = np.array(df) * (1 / n_buys)  # equally weighted

    # Return matrix
    arr_ret = np.array(df_returns) + 1

    df_payoff['payoff'] = (arr_buys * arr_ret @ np.ones(len(df_sectors)).reshape((len(df_sectors), 1)))[:, 0]
    df_payoff['tri'] = df_payoff['payoff'].cumprod()

    fig_payoff = px.line(df_payoff, x='date', y='tri')
    fig_payoff.show()

    print(f"---> payoff for these buys between period {df_payoff['date'].min()} and {df_payoff['date'].max()} is {(df_payoff['tri'].values[-1] - 1) * 100:.2f}%")

    return df_payoff

# Execute the payoff plotting function
df_payoff = plot_payoff(df_buys)

# End of script timing
print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)
