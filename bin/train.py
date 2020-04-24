import numpy as np
import pandas as pd
import tables
import lightgbm
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import shutil
import datetime
import os
sys.path.append('./')
from libs.data_utils import CustomTimeSeriesSplitter, make_submission, reduce_mem_usage
from libs.trainer import train_lgb
from libs.wrmsse import WRMSSEEvaluator
from libs.get_features import get_features

log_dir = f'./log/{datetime.datetime.now()}'
os.mkdir(log_dir)
shutil.copyfile("./libs/get_features.py", log_dir+"/get_features.py")
INPUT_DIR = '../input/m5-forecasting-accuracy'

# READ data
data = pd.read_hdf(f'{INPUT_DIR}/data.h5')
# get CV
day_col = "d"
DAYS_PRED = 28
cv_params = {
    "n_splits": 1,
    "DAYS_PRED": DAYS_PRED,
    "train_days": 365*2 + 185,
    "test_days": DAYS_PRED,
    "day_col": day_col,
}
cv = CustomTimeSeriesSplitter(**cv_params)

features=get_features()
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
importances = pd.DataFrame()
importances['feature'] = X_test.columns
importances['average'] = np.zeros(X_test.shape[1])
preds = np.zeros(X_test.shape[0])

for fold_n, model in enumerate(models):
    preds += model.predict(X_test)
    importances[f'fold_{fold_n + 1}'] = model.feature_importance(imp_type)

preds = preds / cv.get_n_splits()
importances['average'] = importances[[f'fold_{fold_n + 1}' for fold_n in range(cv.get_n_splits())]].mean(axis=1)
plt.figure(figsize=(32, 16))
sns.barplot(data=importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title(f"50 TOP feature importance / wrmsse:{losses.loc['average','wrmsse']}")
plt.savefig(f"{log_dir}/{losses.loc['average','wrmsse']}.png")

submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
make_submission(id_date.assign(demand=preds), submission, log_dir)
