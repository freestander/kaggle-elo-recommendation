# Kaggle Elo Merchant Category Recommendation
This is the repo for Kaggle's Elo Merchant Category Recommendation. The Kaggle competition page can be found in [the link here.](https://www.kaggle.com/c/elo-merchant-category-recommendation)

# Problem Description
The model will predict a loyalty score for each card_id based on the the customer features and merchant transaction history. 

# Data
- train.csv - the training data set
- test.csv - the test data set
- merchants.csv - additional information about all merchants / merchant_ids in the dataset.
  - `merchant_id`: Unique merchant identifier
  - `merchant_group_id`: Merchant group (anonymized )
  - `merchant_category_id`: Unique identifier for merchant category (anonymized )
  - `subsector_id`: Merchant category group (anonymized )
  - `numerical_1`: anonymized measure
  - `numerical_2`: anonymized measure
  - `category_1`: anonymized category
  - `most_recent_sales_range`: Range of revenue (monetary units) in last active month --> A > B > C > D > E
  - `most_recent_purchases_range`: Range of quantity of transactions in last active month --> A > B > C > D > E
  - `avg_sales_lag3`: Monthly average of revenue in last 3 months divided by revenue in last active month
  - `avg_purchases_lag3`: Monthly average of transactions in last 3 months divided by transactions in last active month
  - `active_months_lag3`: Quantity of active months within last 3 months
  - `avg_sales_lag6`: Monthly average of revenue in last 6 months divided by revenue in last active month
  - `avg_purchases_lag6`: Monthly average of transactions in last 6 months divided by transactions in last active month
  - `active_months_lag6`: Quantity of active months within last 6 months
  - `avg_sales_lag12`: Monthly average of revenue in last 12 months divided by revenue in last active month
  - `avg_purchases_lag12`: Monthly average of transactions in last 12 months divided by transactions in last active month
  - `active_months_lag12`: Quantity of active months within last 12 months
  - `category_4`: anonymized category
  - `city_id`: City identifier (anonymized )
  - `state_id`: State identifier (anonymized )
  - `category_2`: anonymized category
- historical_transactions.csv - up to 3 months' worth of historical transactions for each card_id
- new_merchant_transactions.csv - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.
- sample_submission.csv - a sample submission file in the correct format whichcontains all card_ids you are expected to predict for.

# Exploratory Data Analysis

# Data Processing

# Model Selection

# Summary
