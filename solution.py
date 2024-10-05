# %%

import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print('---> Python Script Start', t0 := datetime.datetime.now())

# %%

print('---> the parameters')

# training and test dates
start_train = datetime.date(2017, 1, 1)
end_train = datetime.date(2023, 11, 30) # gap for embargo (no overlap between train and test)
start_test = datetime.date(2024, 1, 1) # test set is this datasets 2024 data
end_test = datetime.date(2024, 6, 30)

n_buys = 10
verbose = False

print('---> initial data set up')

# sector data
df_sectors = pd.read_csv('data/data0.csv')

# price and fin data
df_data = pd.read_csv('data/data1.csv')
df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())

df_x = df_data[['date', 'security', 'price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']].copy()
df_y = df_data[['date', 'security', 'label']].copy()

list_vars1 = ['price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']

# we will perform walk forward validation for testing the buys - https://www.linkedin.com/pulse/walk-forward-validation-yeshwanth-n
df_signals = pd.DataFrame(data={'date':df_x.loc[(df_x['date']>=start_test) & (df_x['date']<=end_test), 'date'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='date', inplace=True) # this code just gets the dates that we need to generate buy signals for

# %%

for i in range(len(df_signals)):

    if verbose: print('---> doing', df_signals.loc[i, 'date'])

    # this iteretaions training set
    df_trainx = df_x[df_x['date']<df_signals.loc[i, 'date']].copy()
    df_trainx.drop(labels=df_trainx[df_trainx['date']==df_trainx['date'].max()].index, inplace=True) # no overlap with test set

    df_trainy = df_y[df_y['date']<df_signals.loc[i, 'date']].copy()
    df_trainy.drop(labels=df_trainy[df_trainy['date']==df_trainy['date'].max()].index, inplace=True) # no overlap with test set

    # this iteretaions test set
    df_testx = df_x[df_x['date']>=df_signals.loc[i, 'date']].copy()
    df_testy = df_y[df_y['date']>=df_signals.loc[i, 'date']].copy()

    # scale, and store scaling objects for test set
    dict_scaler = {}
    for col in list_vars1:

        dict_scaler[col] = MinMaxScaler(feature_range=(-1,1))
        df_trainx[col] = dict_scaler[col].fit_transform(np.array(df_trainx[col]).reshape((len(df_trainx[col]),1)))[:, 0]
        df_testx[col] = dict_scaler[col].transform(np.array(df_testx[col]).reshape((len(df_testx[col]),1)))[:, 0]

    # fit a classifier
    if i == 0:
        # clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=10, min_samples_split=5000, min_samples_leaf=5000, min_weight_fraction_leaf=0.0, max_features='sqrt', random_state=0)
        model1 = XGBClassifier(
            n_estimators=100,    # Number of trees
            learning_rate=0.075,   # Step size shrinkage
            max_depth=5,         # Maximum depth of a tree
            min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child
            gamma=0.1,             # Minimum loss reduction required to make a further partition
            subsample=0.8,       # Subsample ratio of the training instances
            colsample_bytree=0.8,# Subsample ratio of columns when constructing each tree
            objective='binary:logistic',  # Objective function
            eval_metric='logloss',        # Evaluation metric
            random_state=0
        )
        model2 = RandomForestClassifier(
            n_estimators=10,
            criterion='gini', 
            max_depth=10,
            min_samples_split=1000,
            min_samples_leaf=1000,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt', 
            random_state=0
        )
        model3 = GradientBoostingClassifier(
            n_estimators=10,         # Start with 100
            learning_rate=0.8,        # Typical starting value
            max_depth=10,              # Keep it shallow to avoid overfitting
            min_samples_split=1000,     # Minimum samples for split
            min_samples_leaf=1000,       # Minimum samples for leaf
            subsample=0.8,            # Use 80% of data for training each tree
            max_features='sqrt',      # Use sqrt of features
            random_state=0
        )
        model4 = LogisticRegression(
            C=1.0,                     # Inverse of regularization strength
            penalty='l2',              # Regularization type (l1 or l2)
            solver='lbfgs',           # Algorithm for optimization
            max_iter=1000,             # Maximum iterations
            tol=1e-4,                 # Tolerance for stopping criteria
            class_weight='balanced',   # Handle imbalanced classes
            fit_intercept=True         # Include intercept term
        )
        model5 = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            loss_function='Logloss',
            eval_metric='AUC',
            random_state=0,
            verbose=0
        )
        voting_clf = VotingClassifier(estimators=[
            ('xgb', model1),
            ('rf', model2),
            ('gb', model3),
            ('logreg', model4),
            ('catboost', model5)],
            voting='soft',  # Use 'soft' for probability-based voting
            weights=[6,4,0,0,0]
        )
        voting_clf.fit(np.array(df_trainx[list_vars1]), df_trainy['label'].values)

    # predict and calc accuracy - 0.5 is the implicit cuttoff here
    df_testy['signal'] = voting_clf.predict_proba(np.array(df_testx[list_vars1]))[:, 1] # use probs to get strength of classification
    df_testy['pred'] = voting_clf.predict(np.array(df_testx[list_vars1]))
    df_testy['count'] = 1

    df_current = df_testy[df_testy['date']==df_signals.loc[i, 'date']]

    acc_total = (df_testy['label'] == df_testy['pred']).sum()/len(df_testy)
    acc_current = (df_current['label'] == df_current['pred']).sum()/len(df_current)
    
    print('---> accuracy test set', round(acc_total, 2), ', accuracy current date', round(acc_current, 2))

    # add accuracy and signal to dataframe
    df_signals.loc[i, 'acc_total'] = acc_total
    df_signals.loc[i, 'acc_current'] = acc_current

    df_signals.loc[i, df_current['security'].values] = df_current['signal'].values

# %%

# create buy matrix for payoff plot
df_signals['10th'] = df_signals[df_sectors['security'].values].apply(lambda x: sorted(x)[len(df_sectors)-n_buys-1], axis=1)

df_index = pd.DataFrame(np.array(df_signals[df_sectors['security'].values]) > np.array(df_signals['10th']).reshape((len(df_signals),1)))

# set 1 for top 10 strongest signals
df_buys = pd.DataFrame()
df_buys[df_sectors['security'].values] = np.zeros((len(df_signals), len(df_sectors)))
df_buys[df_index.values] = 1
df_buys.insert(0, 'date', df_signals['date'].copy())
df_buys

# check some signal plots
fig_aapl = px.line(df_signals, x='date', y='AAPL')
fig_aapl.show()

fig_pixel = px.imshow(np.array(df_buys[df_sectors['security'].values]))
fig_pixel.show()

# %%

# create return matrix
df_returns = pd.read_csv('data/returns.csv')
df_returns['date']= pd.to_datetime(df_returns['date']).apply(lambda d: d.date())
df_returns = df_returns[df_returns['date']>=start_test]
df_returns = df_returns.pivot(index='date', columns='security', values='return1')

def plot_payoff(df_buys):

    df = df_buys.copy()

    #assert (df.sum(axis=1)==10).sum() == len(df), '---> must have exactly 10 buys each day'

    # matrix of buys
    df_payoff = df[['date']].copy()
    del df['date']
    arr_buys = np.array(df)
    arr_buys = arr_buys*(1/n_buys) # equally weighted

    # return matrix
    arr_ret = np.array(df_returns)
    arr_ret = arr_ret + 1

    df_payoff['payoff'] = (arr_buys * arr_ret @ np.ones(len(df_sectors)).reshape((len(df_sectors), 1)))[:, 0]
    df_payoff['tri'] = df_payoff['payoff'].cumprod()

    fig_payoff = px.line(df_payoff, x='date', y='tri')
    fig_payoff.show()

    print(f"---> payoff for these buys between period {df_payoff['date'].min()} and {df_payoff['date'].max()} is {(df_payoff['tri'].values[-1]-1)*100 :.2f}%")

    return df_payoff

df_payoff = plot_payoff(df_buys)

# %%

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)

