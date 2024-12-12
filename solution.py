#!/usr/bin/python3
#!encoding=utf8

import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go

def momentum_model():
    # Define the training period, the previous year is the most important to train to determine the growth rate of the stock
    start_train = datetime.datetime(2022, 11, 29)
    end_train = datetime.datetime(2023, 11, 29)

    # Define the performance period for the additional column
    start_perf = datetime.datetime(2024, 1, 1)
    end_perf = datetime.datetime(2024, 6, 30)

    # Read data1.csv and extract relevant columns (date, stock, price)
    data1 = pd.read_csv('data/data1.csv', usecols=[0, 1, 2], parse_dates=[0], names=['Date', 'Stock', 'Price'], header=0)
    print("--> Imported data")
    # Copy the date range directly from data1.csv (A1761:A1884)
    date_range = data1.loc[1760:1883, 'Date'].reset_index(drop=True)
    # Filter data to keep only the training period data
    data_train = data1[(data1['Date'] >= start_train) & (data1['Date'] <= end_train)]
    # Calculate performance ratio for each stock during the training period
    performance_ratios = {}
    for stock in data_train['Stock'].unique():
        stock_data = data_train[data_train['Stock'] == stock]
        start_price = stock_data['Price'].iloc[0]  # First price in the training period
        end_price = stock_data['Price'].iloc[-1]   # Last price in the training period

        # Calculate performance ratio
        performance_ratio = end_price / start_price
        performance_ratios[stock] = performance_ratio

    # Convert the performance ratios into a DataFrame
    performance_df = pd.DataFrame(list(performance_ratios.items()), columns=['Stock', 'Performance_Ratio'])

    # Get the top 10 highest performing stocks
    top_performance_df = performance_df.nlargest(10, 'Performance_Ratio')

    # Convert performance ratios to percentage increase
    top_performance_df['Performance_Ratio'] = (top_performance_df['Performance_Ratio'] - 1) * 100

    # Calculate performance from January 1, 2024, to June 30, 2024
    for stock in top_performance_df['Stock']:
        stock_data_perf = data1[(data1['Stock'] == stock) & (data1['Date'] >= start_perf) & (data1['Date'] <= end_perf)]
        if not stock_data_perf.empty:
            start_price_perf = stock_data_perf['Price'].iloc[0]  # First price in the performance period
            end_price_perf = stock_data_perf['Price'].iloc[-1]   # Last price in the performance period

            # Calculate performance ratio for the performance period
            performance_ratio_perf = end_price_perf / start_price_perf
            top_performance_df.loc[top_performance_df['Stock'] == stock, 'Performance_Ratio_2024'] = performance_ratio_perf
        else:
            top_performance_df.loc[top_performance_df['Stock'] == stock, 'Performance_Ratio_2024'] = None  # No data

    # Convert the 2024 performance ratios to percentage increase
    top_performance_df['Performance_Ratio_2024'] = (top_performance_df['Performance_Ratio_2024'] - 1) * 100

    # Calculate the average performance ratio for the last column
    average_performance_ratio_2024 = top_performance_df['Performance_Ratio_2024'].mean()

    # Create a DataFrame to store buy signals
    buy_signals = pd.DataFrame(0, index=date_range, columns=data_train['Stock'].unique())  # Initialize with 0s

    # Fill the DataFrame with 1s for the top stocks, indicating a buy on each day
    buy_signals.loc[:, top_performance_df['Stock']] = 1

    # Change the cell A1 to "date" and save to output.csv
    buy_signals.to_csv('output.csv', header=True, index=True)

    # Load the output CSV and change the first cell to "date"
    output_df = pd.read_csv('output.csv')

    # Change the first header to "date" and drop the index column if it exists
    output_df.columns.values[0] = "date"
    return output_df

def plot_payoff(sectors_dataframe, buys_dataframe, returns_dataframe, allowed_buys):
    df = buys_dataframe.copy()
   # assert (df.sum(axis=2)==10).sum() == len(df), '---> must have exactly 10 buys each day'
    # matrix of buys
    df_payoff = df[['date']].copy()
    del df['date']
    arr_buys = np.array(df)
    arr_buys = arr_buys*(1/allowed_buys) # equally weighted
    # return matrix
    arr_ret = np.array(returns_dataframe)
    arr_ret = arr_ret + 1
    df_payoff['payoff'] = (arr_buys * arr_ret @ np.ones(len(sectors_dataframe)).reshape((len(sectors_dataframe), 1)))[:, 0]
    df_payoff['tri'] = df_payoff['payoff'].cumprod()
    fig_payoff = px.line(df_payoff, x='date', y='tri')
    fig_payoff.show()
    print(f"---> payoff for these buys between period {df_payoff['date'].min()} and {df_payoff['date'].max()} is {(df_payoff['tri'].values[-1]-1)*100 :.2f}%")
    return df_payoff


def main ():
    print('---> Python Script Start', t0 := datetime.datetime.now())
    # training and test dates
    start_train = datetime.date(2017, 1, 1)
    end_train = datetime.date(2023, 11, 30) # gap for embargo (no overlap between train and test)
    start_test = datetime.date(2024, 1, 1) # test set is this datasets 2024 data
    end_test = datetime.date(2024, 6, 30)

    n_buys = 10

    print('---> initial data set up')

    # sector data
    df_sectors = pd.read_csv('data/data0.csv')

    # price and fin data
   # df_data = pd.read_csv('data/data1.csv')
   # df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())
   # df_x = df_data[['date', 'security', 'price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']].copy()
   # df_y = df_data[['date', 'security', 'label']].copy()
    list_vars1 = ['price', 'return30', 'ratio_pe', 'ratio_pcf', 'ratio_de', 'ratio_roe', 'ratio_roa']
    #buy matrix test
    df_buy = momentum_model()

    # create return matrix
    df_returns = pd.read_csv('data/returns.csv')
    df_returns['date']= pd.to_datetime(df_returns['date']).apply(lambda d: d.date())
    df_returns = df_returns[df_returns['date']>=start_test]
    df_returns = df_returns.pivot(index='date', columns='security', values='return1')
    #determine payoff and generate graph
    df_payoff = plot_payoff(df_sectors, df_buy, df_returns, n_buys)
    print('---> Python Script End', t1 := datetime.datetime.now())
    print('---> Total time taken', t1 - t0)

if __name__ == '__main__':
    main()
