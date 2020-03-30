import numpy as np
import os
import warnings
import gc

import sys
sys.path.append('./')
from libs.data_utils import encode_categorical, extract_num, reshape_sales, merge_calendar, merge_prices
from libs.feature_utils import add_demand_features, add_price_features, add_time_features

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import tables
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Read Data
INPUT_DIR = "../input/m5-forecasting-accuracy"
print("Reading files...")
calendar = pd.read_csv(f"{INPUT_DIR}/calendar.csv")
prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv")
sales = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv",)
submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

NUM_ITEMS = sales.shape[0]
DAYS_PRED = submission.shape[1] - 1

# Encode Categorical
calendar = encode_categorical(calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"])
sales = encode_categorical(sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],)
prices = encode_categorical(prices, ["item_id", "store_id"])

# merge tables
data = reshape_sales(sales, submission, DAYS_PRED, d_thresh=1941 - 365 * 3)
del sales
gc.collect()

calendar["d"] = extract_num(calendar["d"])
data = merge_calendar(data, calendar)
del calendar
gc.collect()

data = merge_prices(data, prices)
del prices
gc.collect()


# feature engineering
data = add_demand_features(data, DAYS_PRED)
data = add_price_features(data)
dt_col = "date"
data = add_time_features(data, dt_col)
data = data.sort_values("date")

print("start date:", data[dt_col].min())
print("end date:", data[dt_col].max())
print("data shape:", data.shape)

# save data
data.to_hdf(f"{INPUT_DIR}/data.h5", key='df', mode='w')
