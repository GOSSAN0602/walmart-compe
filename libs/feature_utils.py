import numpy as np
import pandas as pd
from tqdm import tqdm

def add_demand_features(df, DAYS_PRED):
    for diff in tqdm([0, 1, 2, 7, 14, 21, 28]):
        shift = DAYS_PRED + diff
        df[f"demand_shift_t{shift}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )
    for window in tqdm([7, 14, 21, 28, 60, 90, 180]):
        df[f"demand_rolling_std_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )
    for window in tqdm([7, 14, 21, 28, 60, 90, 180]):
        df[f"demand_rolling_mean_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    for window in tqdm([7, 14, 21, 28, 60, 90, 180]):
        df[f"demand_rolling_median_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).median()
        )
    for window in tqdm([7, 14, 21, 28, 60, 90, 180]):
        df[f"demand_rolling_max_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).max()
        )
    for window in tqdm([7, 14, 21, 28, 60, 90, 180]):
        df[f"demand_rolling_min_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).min()
        )
    for window in tqdm([7, 14, 21, 28, 60, 90, 180]):
        df[f"demand_rolling_skew_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).skew()
        )
    print('demand finish')
    return df


def add_price_features(df):
    df["shift_price_t1"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1)
    )
    df["price_change_t1"] = (df["shift_price_t1"] - df["sell_price"]) / (
        df["shift_price_t1"]
    )
    df["rolling_price_max_t365"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(365).max()
    )
    df["price_change_t365"] = (df["rolling_price_max_t365"] - df["sell_price"]) / (
        df["rolling_price_max_t365"]
    )
    for window in tqdm([7, 30]):
        df[f"price_rolling_mean_t{window}"] = df.groupby(["store_id","item_id"])["sell_price"].transform(
            lambda x: x.rolling(window).mean()
        )
    for window in tqdm([7, 30]):
        df[f"price_rolling_max_t{window}"] = df.groupby(["store_id","item_id"])["sell_price"].transform(
            lambda x: x.rolling(window).max()
        )
    for window in tqdm([7, 30]):
        df[f"price_rolling_min_t{window}"] = df.groupby(["store_id","item_id"])["sell_price"].transform(
            lambda x: x.rolling(window).min()
        )
    for window in tqdm([7, 30]):
        df[f"price_rolling_std_t{window}"] = df.groupby(["store_id","item_id"])["sell_price"].transform(
            lambda x: x.rolling(window).std()
        )
    df['price_norm'] = df['sell_price'] / df['price_rolling_max_t7']
    df['price_nunique'] = df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')

    print('price finish')
    return df.drop(["rolling_price_max_t365", "shift_price_t1"], axis=1)


def add_time_features(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)

    print('days finish')
    return df
