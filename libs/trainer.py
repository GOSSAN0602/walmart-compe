import lightgbm as lgb
import pandas as pd
import numpy as np
import gc
import sys
sys.path.append('./')
from libs.wrmsse import WRMSSEEvaluator

def train_lgb(bst_params, fit_params, X, y, cv, tr_id_date, drop_when_train=None):
    # Read data
    INPUT_DIR = "../input/m5-forecasting-accuracy"
    print("Reading files...")
    calendar = pd.read_csv(f"{INPUT_DIR}/calendar.csv")
    prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv")
    train_df = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv",)

    models = []
    losses = pd.DataFrame(
                          columns=['rmse', 'wrmsse'],
                          index=['fold_'+str(x) for x in range(cv.get_n_splits())]+['average']
                          )

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n----- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) -----\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(X_trn.drop(drop_when_train, axis=1), label=y_trn)
        val_set = lgb.Dataset(X_val.drop(drop_when_train, axis=1), label=y_val)

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
        )
        models.append(model)

        # WRMSSE
        tr_d_col = ['d_'+str(x) for x in X_trn['d'].unique().tolist()]
        va_d_col = ['d_'+str(x) for x in X_val['d'].unique().tolist()]
        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        tr_fold_df = train_df.loc[:,id_columns + tr_d_col]
        va_fold_df = train_df.loc[:, ['id']+va_d_col]
        valid_preds = va_fold_df.copy()
        valid_preds_array = model.predict(X_val.drop(drop_when_train, axis=1))
        va_id_date = tr_id_date.iloc[idx_val, :]
        va_id_date = va_id_date.assign(demand=valid_preds_array)
        valid_preds = va_id_date[["id", "date", "demand"]]
        valid_preds = valid_preds.pivot(index="id", columns="date", values="demand").reset_index()
        valid_preds.columns = ["id"] + va_d_col
        valid_preds = pd.merge(va_fold_df.loc[:,'id'],valid_preds,how='left',on='id')
        valid_preds.drop(['id'],axis=1,inplace=True)
        va_fold_df.drop(['id'],axis=1,inplace=True)
        evaluator = WRMSSEEvaluator(tr_fold_df, va_fold_df, calendar, prices)
        losses.loc[f"fold_{idx_fold}",'wrmsse'] = evaluator.score(valid_preds)

        del idx_trn, idx_val, X_trn, X_val, y_trn, y_val, valid_preds, valid_preds_array, evaluator
        gc.collect()

    #losses.loc['average','rmse']=losses['rmse'].mean()
    losses.loc['average','wrmsse']=losses['wrmsse'].mean()

    return models, losses
