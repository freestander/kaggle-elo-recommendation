# Kaggle Elo Merchant Category Recommendation
This is the repo for Kaggle's Elo Merchant Category Recommendation. The Kaggle competition page can be found in [the link here.](https://www.kaggle.com/c/elo-merchant-category-recommendation)

# Problem Description
The model will predict a loyalty score for each `card_id` based on the the customer features and merchant transaction history. 

# Data Description
- train.csv - The training data set
  - `first_active_month`: Card first active month
  - `card_id`: Card identifier
  - `feature_1`: Customer feature 1
  - `feature_2`: Customer feature 2
  - `feature_3`: Customer feature 3
  - `target`: Customer loyalty score
- test.csv - The test data set
  - `first_active_month`: Card first active month
  - `card_id`: Card identifier
  - `feature_1`: Customer feature 1
  - `feature_2`: Customer feature 2
  - `feature_3`: Customer feature 3
- merchants.csv - Additional information about all merchants / `merchant_id`s in the dataset.
  - `merchant_id`: Unique merchant identifier
  - `merchant_group_id`: Merchant group (anonymized)
  - `merchant_category_id`: Unique identifier for merchant category (anonymized)
  - `subsector_id`: Merchant category group (anonymized)
  - `numerical_1`: Anonymized measure
  - `numerical_2`: Anonymized measure
  - `category_1`: Anonymized category
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
  - `category_4`: Anonymized category
  - `city_id`: City identifier (anonymized)
  - `state_id`: State identifier (anonymized)
  - `category_2`: Anonymized category
- historical_transactions.csv - Up to 3 months' worth of historical transactions for each `card_id`
  - `authorized_flag`: 'Y' if approved, 'N' if denied
  - `card_id`: Card identifier
  - `city_id`: City identifier (anonymized)
  - `category_1`: Anonymized category
  - `installments`: Number of installments of purchase
  - `category_3`: Anonymized category
  - `merchant_category_id`: Unique identifier for merchant category (anonymized)
  - `merchant_id`: Unique merchant identifier
  - `month_lag`: Month lag to reference date
  - `purchase_amount`: Normalized purchase amount
  - `purchase_date`: Purchase date
  - `category_2`: Anonymized category
  - `state_id`: State identifier (anonymized)
  - `subsector_id`: Merchant category group (anonymized)
- new_merchant_transactions.csv - Two months' worth of data for each `card_id` containing ALL purchases that `card_id` made at `merchant_id`s that were not visited in the historical data.
  - `authorized_flag`: 'Y' if approved, 'N' if denied
  - `card_id`: Card identifier
  - `city_id`: City identifier (anonymized)
  - `category_1`: Anonymized category
  - `installments`: Number of installments of purchase
  - `category_3`: Anonymized category
  - `merchant_category_id`: Unique identifier for merchant category (anonymized)
  - `merchant_id`: Unique merchant identifier
  - `month_lag`: Month lag to reference date
  - `purchase_amount`: Normalized purchase amount
  - `purchase_date`: Purchase date
  - `category_2`: Anonymized category
  - `state_id`: State identifier (anonymized)
  - `subsector_id`: Merchant category group (anonymized)
- sample_submission.csv - A sample submission file in the correct format whichcontains all card_ids you are expected to predict for.
  - `card_id`: Card identifier
  - `target`: Customer loyalty score

# Exploratory Data Analysis
The `features` in train and test data set have similar distributions.
- `feature_1` has 5 unique values
- `feature_2` has 3 unique values
- `feature_3` has 2 unique values

The `first_active_month` in in train and test data set have similar distributions.
- More customer are active in recent years.
- Distribution is very skewed towards in years 2016 - 2018.

The `target` value in train data set is normally distrubited with some outliers.
- The target variable is normally distributed around zero.
- There are some very low loyalty scores below -30.

# Data Processing and Feature Engineering
- Transform the `target` in the train data set to make it more normally distributed.
- Remove outlier with respect to `target` in the train data set (optional)

# Model Evaluation
- LightGBM
- XGBoost

# Model Selection

# Summary
