import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

print('---> Python Script Start', t0 := datetime.datetime.now())

# Parameters
start_train = datetime.date(2017, 1, 1)
end_train = datetime.date(2023, 11, 30)
start_test = datetime.date(2024, 1, 1)
end_test = datetime.date(2024, 6, 30)
n_buys = 10
verbose = False

# Data loading and preprocessing
df_sectors = pd.read_csv('data/data0.csv')
df_data = pd.read_csv('data/data1.csv')
df_data['date'] = pd.to_datetime(df_data['date']).dt.date

df_x = df_data[['date', 'security', 'price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']].copy()
df_y = df_data[['date', 'security', 'label']].copy()

list_vars1 = ['price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']

df_signals = pd.DataFrame(data={'date':df_x.loc[(df_x['date']>=start_test) & (df_x['date']<=end_test), 'date'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='date', inplace=True)


# Define models
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)

classifiers = {
    # 'LogisticRegression': lr_model,
    'XGBoost': xgb_model,
    # 'CatBoost': cat_model
}

results = {}

# Prepare the entire training set
df_trainx = df_x[df_x['date'] < start_test].copy()
df_trainy = df_y[df_y['date'] < start_test].copy()

scaler = MinMaxScaler(feature_range=(-1,1))
df_trainx[list_vars1] = scaler.fit_transform(df_trainx[list_vars1])

# Fit all models once
for clf_name, clf in classifiers.items():
    print(f'---> Fitting {clf_name}')
    clf.fit(df_trainx[list_vars1], df_trainy['label'])

for clf_name, clf in classifiers.items():
    print(f'---> Processing {clf_name}')
    df_signals_clf = df_signals.copy()
    
    for i in range(len(df_signals)):
        if verbose: print('---> doing', df_signals_clf.loc[i, 'date'])

        df_testx = df_x[df_x['date']==df_signals_clf.loc[i, 'date']].copy()
        df_testy = df_y[df_y['date']==df_signals_clf.loc[i, 'date']].copy()

        df_testx[list_vars1] = scaler.transform(df_testx[list_vars1])
        
        y_pred_proba = clf.predict_proba(df_testx[list_vars1])
        if clf_name != 'LogisticRegression':
            y_pred_proba = y_pred_proba[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        df_testy['signal'] = y_pred_proba
        df_testy['pred'] = y_pred

        df_current = df_testy[df_testy['date']==df_signals_clf.loc[i, 'date']]

        acc_current = (df_current['label'] == df_current['pred']).mean()
        
        print(f'---> {clf_name} accuracy current date {acc_current:.2f}')

        df_signals_clf.loc[i, 'acc_current'] = acc_current
        df_signals_clf.loc[i, df_current['security'].values] = df_current['signal'].values

    df_signals_clf['10th'] = df_signals_clf[df_sectors['security'].values].apply(lambda x: sorted(x)[len(df_sectors)-n_buys-1], axis=1)
    df_index = pd.DataFrame(np.array(df_signals_clf[df_sectors['security'].values]) > np.array(df_signals_clf['10th']).reshape((len(df_signals_clf),1)))

    df_buys = pd.DataFrame(0, index=df_signals_clf.index, columns=df_sectors['security'].values)
    df_buys[df_index.values] = 1
    df_buys = pd.concat([df_signals_clf['date'], df_buys], axis=1)

    results[clf_name] = df_buys

# Manual Ensemble
print('---> Processing Ensemble')
df_signals_ensemble = df_signals.copy()

for i in range(len(df_signals)):
    ensemble_preds = np.mean([results[clf_name].iloc[i, 1:] for clf_name in classifiers.keys()], axis=0)
    df_signals_ensemble.loc[i, df_sectors['security'].values] = ensemble_preds

df_signals_ensemble['10th'] = df_signals_ensemble[df_sectors['security'].values].apply(lambda x: sorted(x)[len(df_sectors)-n_buys-1], axis=1)
df_index_ensemble = pd.DataFrame(np.array(df_signals_ensemble[df_sectors['security'].values]) > np.array(df_signals_ensemble['10th']).reshape((len(df_signals_ensemble),1)))

df_buys_ensemble = pd.DataFrame(0, index=df_signals_ensemble.index, columns=df_sectors['security'].values)
df_buys_ensemble[df_index_ensemble.values] = 1
df_buys_ensemble = pd.concat([df_signals_ensemble['date'], df_buys_ensemble], axis=1)

results['Ensemble'] = df_buys_ensemble

# Create return matrix
df_returns = pd.read_csv('data/returns.csv')
df_returns['date'] = pd.to_datetime(df_returns['date']).dt.date
df_returns = df_returns[df_returns['date']>=start_test]
df_returns = df_returns.pivot(index='date', columns='security', values='return1')

def plot_payoff(df_buys):
    df = df_buys.copy()
    df_payoff = df[['date']].copy()
    arr_buys = np.array(df.iloc[:, 1:])
    arr_buys = arr_buys*(1/n_buys) # equally weighted
    arr_ret = np.array(df_returns) + 1
    df_payoff['payoff'] = (arr_buys * arr_ret @ np.ones(len(df_sectors)).reshape((len(df_sectors), 1)))[:, 0]
    df_payoff['tri'] = df_payoff['payoff'].cumprod()
    return df_payoff

for clf_name, df_buys in results.items():
    # Payoff plot
    df_payoff = plot_payoff(df_buys)
    fig_payoff = px.line(df_payoff, x='date', y='tri')
    fig_payoff.show()

    print(f"---> Payoff for {clf_name} between {df_payoff['date'].min()} and {df_payoff['date'].max()} is {(df_payoff['tri'].values[-1]-1)*100:.2f}%")

# Comparison plot
fig = go.Figure()
for clf_name, df_buys in results.items():
    df_payoff = plot_payoff(df_buys)
    fig.add_trace(go.Scatter(x=df_payoff['date'], y=df_payoff['tri'], mode='lines', name=clf_name))

fig.update_layout(title='Classifier Comparison', xaxis_title='Date', yaxis_title='Total Return Index')
fig.show()

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)