import os
import sys
import xgboost as xgb
from loguru import logger

def fit_model(x, dmatrix):
    """
    This is a very hacky way of doing hyperparameter
    optimization with XGBoost Crossvalidation
    """
    for _, row in x.iterrows():
        params = {"objective": "binary:logistic",
                  "eval_metric": "auc",
                  "learning_rate": row[0],
                  "n_estimators": row[1],
                  "max_depth": 4,
                  "min_child_weight": 0,
                  "gamma": 0,
                  "subsample": 0.7,
                  "colsample_bytree": 0.7,
                  "scale_pos_weight": 1,
                  "seed": 27,
                  "reg_alpha": row[2]
                  }
        cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=3,
                            num_boost_round=50, early_stopping_rounds=150,
                            metrics="auc", as_pandas=True, seed=27,
                            stratified=True,
                            callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True), ])
    # return cv_results
    return cv_results[-1:].values[0]
