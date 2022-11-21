import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import evaluate

amex_data = pd.read_parquet("amex_data.parquet")

X = amex_data.drop("target", axis=1)
y = amex_data.target


# Creating Local Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0, stratify=y
)


# Creating Stratified K Fold CV and define scoring metrics
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scoring = ["accuracy", "f1_weighted", "roc_auc"]

# XGBoost
xgb_model = XGBClassifier(
    random_state=0,
    objective="binary:logistic",
    tree_method="gpu_hist",
    predictor="gpu_predictor",
)

# Defining the search space
params = {
    "n_estimators": np.array([100, 150], dtype="int64"),
    "max_depth": np.array([6, 7, 8], dtype="int64"),
    "learning_rate": np.arange(0.01, 0.05, 0.01, dtype="float64"),
    "booster": np.array(["gbtree", "dart"]),
    "gamma": np.array([1.5, 1], dtype="float64"),
    "subsample": np.arange(0.85, 0.9, 0.01, dtype="float64"),
    "colsample_bytree": [0.5],
    "min_child_weight": [8],
    "lambda": [70],
}

# Using Randomized Search to get best parameters
xgb_rscv = RandomizedSearchCV(
    xgb_model,
    params,
    cv=skf,
    scoring=scoring,
    random_state=0,
    refit="roc_auc",
    verbose=2,
)
xgb_best_model = xgb_rscv.fit(X_train, y_train)
print("XGB Best Params: ", xgb_best_model.best_params_)

# Evaluating perfromance on the Test set:
y_pred_test = xgb_best_model.predict(X_test)
y_pred_test_prob = xgb_best_model.predict_proba(X_test)
acc, roc_auc, f1 = evaluate(y_test, y_pred_test, y_pred_test_prob)


# LightGBM
lgbm_model = LGBMClassifier(
    random_state=0,
    objective="binary",
    metric="binary_logloss",
    device="gpu",
    gpu_platform_id=0,
    gpu_device_id=0,
)

# Defining the search space
params = {
    "n_estimators": np.array([100, 150], dtype="int64"),
    "boosting_type": np.array(["gbdt"]),
    "learning_rate": np.arange(0.01, 0.05, 0.01, dtype="float64"),
    "feature_fraction": np.array([0.3, 0.4], dtype="float64"),
    "bagging_fraction": [0.3, 0.5],
    "reg_lambda": [10, 12],
}

# Using Randomized Search to get best parameters
lgbm_rscv = RandomizedSearchCV(
    lgbm_model,
    params,
    cv=skf,
    scoring=scoring,
    random_state=0,
    refit="roc_auc",
    verbose=2,
)
lgbm_best_model = lgbm_rscv.fit(X_train, y_train)
print("LGBM Best Params: ", lgbm_best_model.best_params_)

# Evaluating perfromance on the Test set:
y_pred_test = lgbm_best_model.predict(X_test)
y_pred_test_prob = lgbm_best_model.predict_proba(X_test)
acc, roc_auc, f1 = evaluate(y_test, y_pred_test, y_pred_test_prob)
