# Linear Regression of month purchase_ammount as features

from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
import pandas as pd
from tqdm import tqdm
import numpy as np

# reduce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

print('Loading historical transactions')
hist = pd.read_parquet('../input/historical_transactions.parquet')
hist = reduce_mem_usage(hist)

print('Expanding out purchase_amount by month_lag')
historical_transactions_df = hist.sort_values(by=['card_id', 'purchase_date'])[['card_id', 'month_lag', 'purchase_amount']]
agg_func = {'purchase_amount': ['count','mean', 'sum', 'median', 'nunique'],}
historical_transactions_df_by_month_lag = hist.groupby(['card_id', 'month_lag']).agg(agg_func)
historical_transactions_df_by_month_lag.columns = ['_'.join(map(str,col)) for col in historical_transactions_df_by_month_lag.columns]

regr = LinearRegression()
col_names = ['card_id', 'coef_sum', 'intercept_sum','coef_count','intercept_count','coef_mean','intercept_mean']
df = pd.DataFrame(columns=col_names)

card_ids = list(hist['card_id'].unique())
out_dfs = [None] * len(card_ids)
a = 0

print('Running linear model on month_lag purchase_amount')
for card_id in tqdm(card_ids):
    # sum
    x = list(historical_transactions_df_by_month_lag.loc[card_id].index)
    y = list(historical_transactions_df_by_month_lag.loc[card_id]['purchase_amount_sum'])
    x = [[x] for x in x]
    y = [[x] for x in y]
    reg = regr.fit(x, y)
    coef_sum = (reg.coef_[0][0])
    intercept_sum = (reg.intercept_[0])
    
    # count
    x = list(historical_transactions_df_by_month_lag.loc[card_id].index)
    y = list(historical_transactions_df_by_month_lag.loc[card_id]['purchase_amount_count'])
    x = [[x] for x in x]
    y = [[x] for x in y]
    reg = regr.fit(x, y)
    coef_count = (reg.coef_[0][0])
    intercept_count = (reg.intercept_[0])

    # mean
    x = list(historical_transactions_df_by_month_lag.loc[card_id].index)
    y = list(historical_transactions_df_by_month_lag.loc[card_id]['purchase_amount_mean'])
    x = [[x] for x in x]
    y = [[x] for x in y]
    reg = regr.fit(x, y)
    coef_mean = (reg.coef_[0][0])
    intercept_mean = (reg.intercept_[0])

    out_dfs[a] = {'card_id': card_id,
                  'coef_sum': coef_sum, 'intercept_sum': intercept_sum,
                  'coef_count': coef_count, 'intercept_count': intercept_count,
                  'coef_mean': coef_mean, 'intercept_mean': intercept_mean}
    a += 1

features = []
for col in df.columns:
    if col != 'card_id':
        features.append(col)

df_regression = pd.concat([pd.DataFrame([out_dfs[i]], columns=col_names) for i in range(len(list(filter(None, out_dfs))))],ignore_index=True)

# Store Results
df_regression.to_csv('../working/hist_lags_regression.csv')
df_regression.to_parquet('../working/hist_lags_regression.parquet')

