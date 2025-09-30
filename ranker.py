import xgboost as xgb
import numpy as np
import pandas as pd

def prepare_ranker_data(df):
    """
    Prepare dataset for learning-to-rank.
    df should have columns: [user_id, stock_id, features..., relevance]
    """
    X = df.drop(columns=["user_id", "stock_id", "relevance"]).values
    y = df["relevance"].values
    group = df.groupby("user_id").size().to_numpy()
    return X, y, group

def train_ranker(X, y, group, params=None, num_rounds=50):
    if params is None:
        params = {
            "objective": "rank:pairwise",
            "eta": 0.1,
            "gamma": 1.0,
            "min_child_weight": 0.1,
            "max_depth": 6
        }
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(group)
    model = xgb.train(params, dtrain, num_boost_round=num_rounds)
    return model

def predict_ranker(model, X_new):
    dtest = xgb.DMatrix(X_new)
    return model.predict(dtest)
