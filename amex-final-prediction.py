import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import RocCurveDisplay
from utils import evaluate

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt


amex_data = pd.read_parquet("amex_data.parquet")

X = amex_data.drop("target", axis=1)
y = amex_data.target

# Creating Local Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0, stratify=y
)

# Using fine-tuned parameters for XGBoost
xgb_params = {
    "subsample": 0.85,
    "n_estimators": 150,
    "min_child_weight": 8,
    "max_depth": 7,
    "learning_rate": 0.04,
    "lambda": 70,
    "gamma": 1.5,
    "colsample_bytree": 0.5,
    "booster": "dart",
}

xgb_model = XGBClassifier(
    random_state=0,
    objective="binary:logistic",
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    **xgb_params
)

# Using fine-tuned parameters for LightGBM
lgbm_params = {
    "reg_lambda": 10,
    "n_estimators": 100,
    "learning_rate": 0.04,
    "feature_fraction": 0.3,
    "boosting_type": "gbdt",
    "bagging_fraction": 0.3,
}

lgbm_model = lgbm_model = LGBMClassifier(
    random_state=0,
    objective="binary",
    metric="binary_logloss",
    device="gpu",
    gpu_platform_id=0,
    gpu_device_id=0,
    **lgbm_params
)

ensemble_estimators = [("XGBoost", xgb_model), ("LightGBM", lgbm_model)]

# Ensemble of both the models
best_model = VotingClassifier(estimators=ensemble_estimators, voting="soft")
clf = best_model.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
y_pred_test_prob = clf.predict_proba(X_test)
acc, roc_auc, f1 = evaluate(y_test, y_pred_test, y_pred_test_prob)

# Plot AUC-ROC Curve
fig = RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
