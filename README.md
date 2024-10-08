![img](img/header.png)

# Update 2024-10-08

We are busy grading the submissions. A few comments after having a look at some of the submissions.

1. The 1st grading guide is strictly adhered to.
2. Data leakage seems to be common in the submissions.

Here's a ChatGPT explanation of data leakage

Data leakage in machine learning refers to a situation where information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates. This can result in models that perform well during testing but fail in real-world applications because they rely on data that wouldn’t be available during actual predictions.

Here are a few common examples of data leakage:

1. **Target Leakage**. This occurs when the training dataset includes features that directly or indirectly contain information about the target variable. For instance, if you're predicting whether a patient has a disease and you include a feature that measures a related treatment received after the diagnosis, the model might learn from this post-diagnosis information, leading to misleading accuracy during validation.

2. **Train-Test Contamination**. If the test set is influenced by the training set, this is another form of leakage. For example, if you normalize your data using the mean and standard deviation of the entire dataset (including both training and test data), the model may inadvertently have access to test data characteristics during training. Instead, you should compute normalization parameters using only the training data.

3. **Temporal Leakage**. In time-series forecasting, using future data to predict past events can lead to leakage. For example, if you build a model to predict stock prices using data that includes future prices, the model will appear to perform well, but it wouldn’t be viable in a real-world scenario where future data isn’t available.

4. **Feature Engineering Leakage**. If you create features that use information from the target variable or future information, this can introduce leakage. For example, if you’re predicting customer churn and create a feature that sums up customer interactions from the future, it could skew the results.
Prevention Strategies

To prevent data leakage, you can take the following steps:

- **Careful Feature Selection**: Ensure that features are derived only from the training data and don’t contain information about the target variable.
- **Proper Data Splitting**: Always split your data into training and test sets before any preprocessing to avoid contamination.
- **Cross-Validation**: Use techniques like k-fold cross-validation, ensuring that the splits respect the temporal order in time-series data.
- **Monitoring**: Regularly evaluate the model performance using separate validation datasets that are representative of real-world conditions.

## Welcome!

Welcome to the Prescient Coding Challenge 2024!

## The Problem Description

You have been provided with price and financial data for 100 US stocks. Your task is to generate 1-day-ahead trading signals for each stock. Additionally, you need to select the top 10 stocks each day to form a portfolio. The performance of your selected portfolio will be evaluated based on the total return index over the evaluation (test) period.

This type of trading is known as [swing trades](https://www.investopedia.com/terms/s/swingtrading.asp). You are only allowed [long positions](https://www.investopedia.com/ask/answers/100314/whats-difference-between-long-and-short-position-market.asp), i.e. you matrix of buys will only contain 1's and 0's.

You are given files

1. `README.md` - this file
2. `data0.csv` - 1st data file
3. `data1.csv` - 2nd data file
4. `returns.csv` - returns file
5. `solution.py` - a skeleton structure with sample solution for the problem description

## The Data

The data provided is a mix of daily, monthly, and yearly data. Where possible the data has been issued daily otherwise forward filled to match the pricing data availability.

- The file `data0.csv` contains the security sector data
- The file `data1.csv` contains price, historical returns, financial ratios and the 1-day-ahead price change label for each security and trading day

A brief description of the columns in the data are:

- `date` - close of business day
- `security` - the instrument code, in this case its the stock ticker
- `sector` - the security's sector classification
- `price` - closing day price in USD
- `ratio_pe` - price to earnings ratio
- `ratio_pcf` - price to cash flow ratio
- `ratio_de` - debt to equity ratio
- `ratio_roe` - return on equity
- `ratio_roa` - return on assets
- `label` - a 0 or 1 label, with 0 indicating a loss taking bet and 1 a positive winning bet

[Ratios Reading](https://www.forbes.com/sites/investor-hub/article/10-key-financial-ratios-every-investor-should-know)

## The Output

We are interested in the total payoff for the buys in the testing period `2024-01-01` to `2024-06-30`. The high level steps are

1. Generate buy-signals
2. Create a buy-matrix of 1s (buys) and 0s (don't buy) with each row summing to 10 (10 buys)
3. Generate payoff chart

Your buy-matrix will create your payoff chart using the `plot_payoff` function.

## Hints

- You may use a subset of features.
- You may engineer features using the existing features.
- You may use pure rule based, quant, or ML methods
- You may create more than 1 model to generate buy-signals
- ChatGPT is allowed

## Getting The Project On Your Computer (GitHub)


1. Sign in or sign up to GitHub.
2. On the Coding Challenge repo page, fork the repo as shown below.

![alt text](img/image2.png)

3. Once the project shows as a repo on your GitHub profile, clone the repo.

![alt text](img/image3.png)

4. Since this is on your personal GitHub profile, you can work on your `main` branch.

## How To Submit Your Answer

1. Assuming you are working on your `main` branch
2. `git add .`
3. `git commit -m 'Your Team Name'`
4. `git push origin main`
5. Make sure that your changes are only in one of either a `solution.py` or a `solution.R`.
6. You should see your changes on your repo.
7. On the "Pull Requests" tab, select "New pull request"
8. The GitHub summary should mention only 1 file change
9. Select "Create pull request"
10. Add your team name and short description of how you solved the problem. Confirm the "Create pull request"

![alt text](img/image4.png)

11. You should now be able to see your team's pull request on our repository's list of pull requests.

![alt text](img/image5.png)


## Grading Guide

The table below is the 1st grading guide.

| Step | Criteria                                                                 | Action                                         |
|------|--------------------------------------------------------------------------|------------------------------------------------|
| 1    | Submitted on 5 October 2024 before 2pm?                                  | Yes - next step, no - disqualified             |
| 2    | No tampering with data sets and no additional data imports of any kind?                                              | Yes - next step, no - disqualified             |
| 3    | Script runs without intervention from us?                               | Yes - next step, no - disqualified             |
| 4    | Script runs within 10 minutes?                                            | Yes - next step, no - disqualified             |
| 5    | Does it produce the same solution on consecutive runs? Simulation and stochastic estimation needs to be highly stable. | Yes - next step, no - disqualified             |
| 6    | Does not contain look-ahead bias? Stock picking in this case will be considered look-ahead because you can see future prices. | Yes - next step, no - disqualified             |
| 7    | Successfully feeds into TRI function and produces desired chart?         | Yes - next step, no - disqualified             |

The 2nd grading is a combination of the classification score, TRI final level and the solution originality decided by the Prescient Investment Management Team.

# Download Links

1. [Git](https://git-scm.com/downloads)
2. [Python ](https://www.python.org/downloads/)
3. [VS Code](https://code.visualstudio.com/download)
4. [R Base](https://cran.r-project.org/)
5. [R Studio](https://posit.co/downloads/)
