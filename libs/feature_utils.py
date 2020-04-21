import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('./')
from libs.data_utils import reduce_mem_usage

def add_demand_features(df, DAYS_PRED):
    # lag feature
    for diff in tqdm([0, 7, 14, 21, 28]):
        shift = DAYS_PRED + diff
        df[f"demand_shift_t{shift}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )

    # deamand basic feature
    for window in tqdm([7, 14, 21, 28]):
        df[f"demand_rolling_std_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )
    for window in tqdm([7, 14, 21, 28]):
        df[f"demand_rolling_mean_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    for window in tqdm([7, 14, 21, 28]):
        df[f"demand_rolling_median_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).median()
        )
    for window in tqdm([7, 14, 21, 28]):
        df[f"demand_rolling_max_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).max()
        )
    for window in tqdm([7, 14, 21, 28]):
        df[f"demand_rolling_min_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).min()
        )

    # state_id * cat_id
    agg_df = df.groupby(["state_id", "cat_id", "d"])["demand"].mean().reset_index()
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"state_cate_demand_mean_window{window}"] = agg_df.groupby(["state_id", "cat_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"state_cat_demand_rolling_std_window{window}"] = agg_df.groupby(["state_id","cat_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"state_cat_demand_rolling_mean_window{window}"] = agg_df.groupby(["state_id","cat_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    """
    shiftの名前を組み合わせで変える！！！！！
    """
    df = df.merge(agg_df, on=["cat_id","state_id","d"], how="left")
    del agg_df
    gc.collect()
    ## state * cate * dayofweek
    agg_list = []
    for diff in tqdm([28, 35, 42, 49, 56]):
        agg_list.append(data.groupby(["state_id", "cat_id","d"])[f"demand_shift_t{diff}"].mean().reset_index())
    agg_df = agg_list[0]
    for i in range(len(agg_list )-1):
        agg_df = agg_df.merge(agg_list[i +1], on=["state_id","cat_id","d"], how="left")
    for i in range(4):
        agg_df[f"state_cat_t{28+i*7}/t{35+i*7}"] = agg_df[f"demand_shift_t{28+i*7}"]/agg_df[f"demand_shift_t{35+i*7}"]
    agg_df["state_cat_same_dayofweek_4times_mean"] = (agg_df["demand_shift_t28"]+agg_df["demand_shift_t28"]+agg_df["demand_shift_t35"]+agg_df["demand_shift_t42"]) / 4
    df = df.merge(agg_df, on=["cat_id","state_id","d"], how="left")
    del agg_df, agg_list
    gc.collect()

    # state_id * dept_id
    agg_df = df.groupby(["state_id", "dept_id", "d"])["demand"].mean().reset_index()
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"state_dept_demand_mean_window{window}"] = agg_df.groupby(["state_id", "dept_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"state_dept_demand_rolling_std_window{window}"] = agg_df.groupby(["state_id","dept_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"state_dept_demand_rolling_mean_window{window}"] = agg_df.groupby(["state_id","dept_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    df = df.merge(agg_df, on=["dept_id","state_id","d"], how="left")
    del agg_df
    gc.collect()
    ## state * dept * dayofweek
    agg_list = []
    for diff in tqdm([28, 35, 42, 49, 56]):
        agg_list.append(data.groupby(["state_id", "dept_id","d"])[f"demand_shift_t{diff}"].mean().reset_index())
    agg_df = agg_list[0]
    for i in range(len(agg_list )-1):
        agg_df = agg_df.merge(agg_list[i +1], on=["state_id","dept_id","d"], how="left")
    for i in range(4):
        agg_df[f"state_dept_t{28+i*7}/t{35+i*7}"] = agg_df[f"demand_shift_t{28+i*7}"]/agg_df[f"demand_shift_t{35+i*7}"]
    agg_df["state_dept_same_dayofweek_4times_mean"] = (agg_df["demand_shift_t28"]+agg_df["demand_shift_t28"]+agg_df["demand_shift_t35"]+agg_df["demand_shift_t42"]) / 4
    df = df.merge(agg_df, on=["dept_id","state_id","d"], how="left")
    del agg_df, agg_list
    gc.collect()

    # store_id * cat_id
    agg_df = df.groupby(["store_id", "cat_id", "d"])["demand"].mean().reset_index()
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"store_cat_demand_mean_window{window}"] = agg_df.groupby(["store_id", "cat_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"store_cat_demand_rolling_std_window{window}"] = agg_df.groupby(["store_id","cat_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"store_cat_demand_rolling_mean_window{window}"] = agg_df.groupby(["store_id","cat_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    df = df.merge(agg_df, on=["cat_id","store_id","d"], how="left")
    del agg_df
    gc.collect()
    ## store * cat * dayofweek
    agg_list = []
    for diff in tqdm([28, 35, 42, 49, 56]):
        agg_list.append(data.groupby(["store_id", "cat_id","d"])[f"demand_shift_t{diff}"].mean().reset_index())
    agg_df = agg_list[0]
    for i in range(len(agg_list )-1):
        agg_df = agg_df.merge(agg_list[i +1], on=["store_id","cat_id","d"], how="left")
    for i in range(4):
        agg_df[f"store_cat_t{28+i*7}/t{35+i*7}"] = agg_df[f"demand_shift_t{28+i*7}"]/agg_df[f"demand_shift_t{35+i*7}"]
    agg_df["store_cat_same_dayofweek_4times_mean"] = (agg_df["demand_shift_t28"]+agg_df["demand_shift_t28"]+agg_df["demand_shift_t35"]+agg_df["demand_shift_t42"]) / 4
    df = df.merge(agg_df, on=["cat_id","store_id","d"], how="left")
    del agg_df, agg_list
    gc.collect()

    # store_id * dept_id
    agg_df = df.groupby(["store_id", "dept_id", "d"])["demand"].mean().reset_index()
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"store_dept_demand_mean_window{window}"] = agg_df.groupby(["store_id", "dept_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"store_dept_demand_rolling_std_window{window}"] = agg_df.groupby(["store_id","dept_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )
    for window in tqdm([7, 14, 21, 28]):
        agg_df[f"store_dept_demand_rolling_mean_window{window}"] = agg_df.groupby(["store_id","dept_id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )
    df = df.merge(agg_df, on=["dept_id","store_id","d"], how="left")
    del agg_df
    gc.collect()
    ## store * dept * dayofweek
    agg_list = []
    for diff in tqdm([28, 35, 42, 49, 56]):
        agg_list.append(data.groupby(["store_id", "dept_id","d"])[f"demand_shift_t{diff}"].mean().reset_index())
    agg_df = agg_list[0]
    for i in range(len(agg_list )-1):
        agg_df = agg_df.merge(agg_list[i +1], on=["store_id","dept_id","d"], how="left")
    for i in range(4):
        agg_df[f"store_dept_t{28+i*7}/t{35+i*7}"] = agg_df[f"demand_shift_t{28+i*7}"]/agg_df[f"demand_shift_t{35+i*7}"]
    agg_df["store_dept_same_dayofweek_4times_mean"] = (agg_df["demand_shift_t28"]+agg_df["demand_shift_t28"]+agg_df["demand_shift_t35"]+agg_df["demand_shift_t42"]) / 4
    df = df.merge(agg_df, on=["dept_id","store_id","d"], how="left")
    del agg_df, agg_list
    gc.collect()
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
    for window in tqdm([7]):
        df[f"price_rolling_mean_t{window}"] = df.groupby(["store_id","item_id"])["sell_price"].transform(
            lambda x: x.rolling(window).mean()
        )
    for window in tqdm([7]):
        df[f"price_rolling_max_t{window}"] = df.groupby(["store_id","item_id"])["sell_price"].transform(
            lambda x: x.rolling(window).max()
        )
    for window in tqdm([7]):
        df[f"price_rolling_min_t{window}"] = df.groupby(["store_id","item_id"])["sell_price"].transform(
            lambda x: x.rolling(window).min()
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
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]
    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)

    # holiday flag
    pub_holi_list = ["MemorialDay","IndependenceDay","Eid al-Fitr","LaborDay","ColumbusDay","ColumbusDay","EidAlAdha","NewYear","MartinLutherKingDay","SuperBowl","PresidentsDay","StPatricksDay"]
    df["public_holiday"] = (df["event_name_1"].isin(pub_holi_list))*1
    # eventday flag
    eventday_list = ["Halloween","VeteransDay","Thanksgiving","Christmas","Father's day","Mother's day","OrthodoxChristmas","ValentinesDay","Purim End","Easter", "esach End", "Cinco De Mayo","OrthodoxEaster", "Chanukah End", "NBAFinalsStart", "NBAFinalsEnd", "Chanukah End"]
    df["eventday"] = (df["event_name_1"].isin(eventday_list))*1
    # NFL days
    df["NFL"] = 0
    df.loc[('2013-06-06' <= df["date"])&(df["date"] <= '2013-06-20'),"NFL"] = 1
    df.loc[('2016-06-05' <= df["date"])&(df["date"] <= '2014-06-15'),"NFL"] = 1
    df.loc[('2015-06-04' <= df["date"])&(df["date"] <= '2015-06-16'),"NFL"] = 1
    # Ramadan days
    df["Ramadan"] = 0
    df.loc[('2013-07-09' <= df["date"])&(df["date"] <= '2013-08-08'),"Ramadan"] = 1
    df.loc[('2014-06-29' <= df["date"])&(df["date"] <= '2014-07-29'),"Ramadan"] = 1
    df.loc[('2015-06-18' <= df["date"])&(df["date"] <= '2015-07-18'),"Ramadan"] = 1
    # drop cols
    df = df.drop(["event_name_1","event_name_2","event_type_1","event_type_2"],axis=1)
    print('days finish')
    return df
