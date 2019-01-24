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
- `feature_1` has 5 unique values.
- `feature_2` has 3 unique values.
- `feature_3` has 2 unique values.

![feature 1](./images/feature_1.png)
![feature 2](./images/feature_2.png)
![feature 3](./images/feature_3.png)

The `first_active_month` in in train and test data set have similar distributions.
- More customers are active in the recent years.
- Distribution is very skewed towards years 2016 - 2018.

![first active month in train](./images/first_active_month_hist_train.png)
![first active month in test](./images/first_active_month_hist_test.png)

The `target` value in train data set is normally distrubited with some outliers.
- The target variable is normally distributed around zero.
- There are some very low loyalty scores below -30.

![first active month in train](./images/target_hist.png)

# Data Processing and Feature Engineering
- Reduce memory footprint by optimizing data types.
- Process outliers with respect to `target` in train.csv (optional).
- Log-transform the `target` in train.csv to make it more normally distributed.
- Fill in logic for NA values of `first_active_month` in test.csv.
- Create new features from historical_transactions.csv
  - hist_trans_min
  - hist_trans_max
  - hist_trans_median
  - hist_trans_std
  - hist_trans_sum
  - hist_trans_count
  - hist_auth_min_n
  - hist_auth_min_y
  - hist_auth_max_n
  - hist_auth_max_y
  - hist_auth_median_n
  - hist_auth_median_y
  - hist_auth_std_n
  - hist_auth_std_y
  - hist_auth_sum_n
  - hist_auth_sum_y
  - hist_auth_count_y_pct
  - hist_install_count_y_pct
  - hist_cat_1_y_pct
  - hist_cat_2_1_pct
  - hist_cat_2_2_pct
  - hist_cat_2_3_pct
  - hist_cat_2_4_pct
  - hist_cat_2_5_pct
  - hist_cat_3_A_pct
  - hist_cat_3_B_pct
  - hist_cat_3_C_pct
  - hist_cat_4_n_pct
  - hist_cat_4_y_pct
  - hist_subsector_1_pct
  - hist_subsector_2_pct
  - hist_subsector_3_pct
  - hist_subsector_4_pct
  - hist_subsector_5_pct
  - hist_subsector_7_pct
  - hist_subsector_8_pct
  - hist_subsector_9_pct
  - hist_subsector_10_pct
  - hist_subsector_11_pct
  - hist_subsector_12_pct
  - hist_subsector_13_pct
  - hist_subsector_14_pct
  - hist_subsector_15_pct
  - hist_subsector_16_pct
  - hist_subsector_17_pct
  - hist_subsector_18_pct
  - hist_subsector_19_pct
  - hist_subsector_20_pct
  - hist_subsector_21_pct
  - hist_subsector_22_pct
  - hist_subsector_23_pct
  - hist_subsector_24_pct
  - hist_subsector_25_pct
  - hist_subsector_26_pct
  - hist_subsector_27_pct
  - hist_subsector_28_pct
  - hist_subsector_29_pct
  - hist_subsector_30_pct
  - hist_subsector_31_pct
  - hist_subsector_32_pct
  - hist_subsector_33_pct
  - hist_subsector_34_pct
  - hist_subsector_35_pct
  - hist_subsector_36_pct
  - hist_subsector_37_pct
  - hist_subsector_38_pct
  - hist_subsector_39_pct
  - hist_subsector_40_pct
  - hist_subsector_41_pct
  - hist_date_min_delta
  - hist_date_max_delta

- Create new features from new_merchant_transactions.csv
  - new_trans_min
  - new_trans_max
  - new_trans_median
  - new_trans_std
  - new_trans_sum
  - new_trans_count
  - new_install_count_y_pct
  - new_cat_1_y_pct
  - new_cat_2_1_pct
  - new_cat_2_2_pct
  - new_cat_2_3_pct
  - new_cat_2_4_pct
  - new_cat_2_5_pct
  - new_cat_3_A_pct
  - new_cat_3_B_pct
  - new_cat_3_C_pct
  - new_cat_4_n_pct
  - new_cat_4_y_pct
  - new_subsector_1_pct
  - new_subsector_2_pct
  - new_subsector_3_pct
  - new_subsector_4_pct
  - new_subsector_5_pct
  - new_subsector_7_pct
  - new_subsector_8_pct
  - new_subsector_9_pct
  - new_subsector_10_pct
  - new_subsector_11_pct
  - new_subsector_12_pct
  - new_subsector_13_pct
  - new_subsector_14_pct
  - new_subsector_15_pct
  - new_subsector_16_pct
  - new_subsector_17_pct
  - new_subsector_18_pct
  - new_subsector_19_pct
  - new_subsector_20_pct
  - new_subsector_21_pct
  - new_subsector_22_pct
  - new_subsector_23_pct
  - new_subsector_24_pct
  - new_subsector_25_pct
  - new_subsector_26_pct
  - new_subsector_27_pct
  - new_subsector_28_pct
  - new_subsector_29_pct
  - new_subsector_30_pct
  - new_subsector_31_pct
  - new_subsector_32_pct
  - new_subsector_33_pct
  - new_subsector_34_pct
  - new_subsector_35_pct
  - new_subsector_36_pct
  - new_subsector_37_pct
  - new_subsector_38_pct
  - new_subsector_39_pct
  - new_subsector_40_pct
  - new_subsector_41_pct
  - new_date_min_delta
  - new_date_max_delta

# Parameter Tuning and Model Evaluation
- LightGBM
  - Number of boosting rounds 63
  - Training RMSE: 3.5938221991741477
  - Testing RMSE: 3.7996810440253364
  - LB Score: 3.870

![first active month in train](./images/lgb_result.png)

- XGBoost
  - Number of boosting rounds 43
  - Training RMSE: 3.572817
  - Testing RMSE: 3.803299
  - LB Score: 3.882

![first active month in train](./images/xgb_result.png)

# Model Selection and Ensemble

# Summary

# Reference
- [Starter EDA + XGBoost of Elo Merchant Data](https://www.kaggle.com/robikscube/starter-eda-xgboost-of-elo-merchant-data)
- [Simple Exploration Notebook - Elo](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo)
- [Kaggle Beginner: LGBM Starter](https://www.kaggle.com/bgrad3031/kaggle-beginner-lgbm-starter)
- [Reducing Memory Footprint](https://www.kaggle.com/rahulahuja010/reducing-memory-footprint)
- [Reducing memory use of Transactions df by 12x](https://www.kaggle.com/poedator/reducing-memory-use-of-transactions-df-by-12x)
- [Boruta feature elimination](https://www.kaggle.com/tilii7/boruta-feature-elimination)
- [Feature Selection with Null Importances](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances)
- [Hyperparameter tuning](https://www.kaggle.com/fabiendaniel/hyperparameter-tuning)
- [A Tutorial of Model Monotonicity Constraint Using Xgboost](https://xiaoxiaowang87.github.io/monotonicity_constraint/)
- [A template for my python machine learning projects](https://github.com/mengdong/python-ml-structure)
- [Template codes and examples for Python machine learning concepts](https://github.com/yuxiangdai/machine-learning-templates)
- [Your First Machine Learning Project in Python Step-By-Step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
- [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
- [A Comprehensive Guide to Ensemble Learning (with Python codes)](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/)
- [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/pdf/1811.12808.pdf)
