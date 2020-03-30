import numpy as np
import pandas as pd
import tables
import lightgbm
import gc
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from libs.data_utils import CustomTimeSeriesSplitter, make_submission
from libs.trainer import train_lgb
from libs.wrmsse import WRMSSEEvaluator

INPUT_DIR = '../input/m5-forecasting-accuracy'

# READ data
data = pd.read_hdf(f'{INPUT_DIR}/data.h5')

# get CV
day_col = "d"
DAYS_PRED = 28
cv_params = {
    "n_splits": 3,
    "DAYS_PRED": DAYS_PRED,
    "train_days": 365*2+185,
    "test_days": DAYS_PRED,
    "day_col": day_col,
}
cv = CustomTimeSeriesSplitter(**cv_params)
features=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
       'wm_yr_wk', 'event_name_1', 'event_type_1',
       'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI',
       'sell_price', 'demand_shift_t28', 'demand_shift_t29',
       'demand_shift_t30', 'demand_shift_t35', 'demand_shift_t42',
       'demand_shift_t49', 'demand_shift_t56', 'demand_rolling_std_t7',
       'demand_rolling_std_t14', 'demand_rolling_std_t21',
       'demand_rolling_std_t28', 'demand_rolling_std_t60',
       'demand_rolling_std_t90', 'demand_rolling_std_t180',
       'demand_rolling_mean_t7', 'demand_rolling_mean_t14',
       'demand_rolling_mean_t21', 'demand_rolling_mean_t28',
       'demand_rolling_mean_t60', 'demand_rolling_mean_t90',
       'demand_rolling_mean_t180', 'demand_rolling_median_t7',
       'demand_rolling_median_t14', 'demand_rolling_median_t21',
       'demand_rolling_median_t28', 'demand_rolling_median_t60',
       'demand_rolling_median_t90', 'demand_rolling_median_t180',
       'demand_rolling_max_t7', 'demand_rolling_max_t14',
       'demand_rolling_max_t21', 'demand_rolling_max_t28',
       'demand_rolling_max_t60', 'demand_rolling_max_t90',
       'demand_rolling_max_t180', 'demand_rolling_min_t7',
       'demand_rolling_min_t14', 'demand_rolling_min_t21',
       'demand_rolling_min_t28', 'demand_rolling_min_t60',
       'demand_rolling_min_t90', 'demand_rolling_min_t180',
       'demand_rolling_skew_t7', 'demand_rolling_skew_t14',
       'demand_rolling_skew_t21', 'demand_rolling_skew_t28',
       'demand_rolling_skew_t60', 'demand_rolling_skew_t90',
       'demand_rolling_skew_t180', 'price_change_t1', 'price_change_t365',
       'price_rolling_mean_t7', 'price_rolling_mean_t30',
       'price_rolling_max_t7', 'price_rolling_max_t30', 'price_rolling_min_t7',
       'price_rolling_min_t30', 'price_rolling_std_t7',
       'price_rolling_std_t30', 'price_norm', 'price_nunique', 'year',
       'quarter', 'month', 'week', 'day', 'dayofweek', 'is_year_end',
       'is_year_start', 'is_quarter_end', 'is_quarter_start', 'is_month_end',
       'is_month_start', 'is_weekend']

# features = [
#     "item_id",
#     "dept_id",
#     "cat_id",
#     "store_id",
#     "state_id",
#     "event_name_1",
#     "event_type_1",
#     "snap_CA",
#     "snap_TX",
#     "snap_WI",
#     "sell_price",
#     # demand features.
#     "shift_t28",
#     "shift_t29",
#     "shift_t30",
#     "rolling_std_t7",
#     "rolling_std_t30",
#     "rolling_std_t60",
#     "rolling_std_t90",
#     "rolling_std_t180",
#     "rolling_mean_t7",
#     "rolling_mean_t30",
#     "rolling_mean_t60",
#     "rolling_mean_t90",
#     "rolling_mean_t180",
#     "rolling_skew_t30",
#     "rolling_kurt_t30",
#     # price features
#     "price_change_t1",
#     "price_change_t365",
#     "rolling_price_std_t7",
#     "rolling_price_std_t30",
#     # time features.
#     "year",
#     "month",
#     "week",
#     "day",
#     "dayofweek",
#     "is_weekend",
# ]

# make dataset
is_train = data["d"] < 1914

# Attach "d" to X_train for cross validation.
X_train = data[is_train][[day_col] + features].reset_index(drop=True)
y_train = data[is_train]["demand"].reset_index(drop=True)
X_test = data[~is_train][features].reset_index(drop=True)

# keep these two columns to use later.
id_date = data[~is_train][["id", "date"]].reset_index(drop=True)
tr_id_date = data[is_train][["id", "date"]].reset_index(drop=True)
del data
gc.collect()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# train config
bst_params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "objective": "regression",
    "n_jobs": -1,
    "seed": 42,
    "learning_rate": 0.05,
    "bagging_fraction": 0.75,
    "bagging_freq": 10,
    "colsample_bytree": 0.75,
}

fit_params = {
    "num_boost_round": 100_000,
    "early_stopping_rounds": 200,
    "verbose_eval": 100,
}

# train
models, losses = train_lgb(
    bst_params, fit_params, X_train, y_train, cv, tr_id_date, drop_when_train=[day_col]
)

del X_train, y_train
gc.collect()

# plot feature importance
imp_type = "gain"
importances = np.zeros(X_test.shape[1])
preds = np.zeros(X_test.shape[0])

for model in models:
    preds += model.predict(X_test)
    importances += model.feature_importance(imp_type)

preds = preds / cv.get_n_splits()
importances = importances / cv.get_n_splits()

submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
make_submission(id_date.assign(demand=preds), submission)

import pdb;pdb.set_trace()
