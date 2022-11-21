import numpy as np
import cudf
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


def generate_aggregate_features(df, agg_col):
    """
    Generate new aggregated features based on existing features.

    Parameters
    ----------
    df : dataframe
        Dataframe with all the features that need to be aggregated.

    agg_col : string (column name)
        Column that is used for aggregating the features.

    Returns
    -------
    df : dataframe
        Dataframe with aggregated features.
    """

    all_cols = [c for c in list(df.columns) if c not in [agg_col, "S_2"]]
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    num_features = [col for col in all_cols if col not in cat_features]

    num_agg = df.groupby(agg_col)[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    num_agg.columns = ["_".join(x) for x in num_agg.columns]
    cat_agg = df.groupby(agg_col)[cat_features].agg(["count", "last", "nunique"])
    cat_agg.columns = ["_".join(x) for x in cat_agg.columns]

    df = cudf.concat([num_agg, cat_agg], axis=1)
    print("Shape after feature engineering: ", df.shape)

    return df


def evaluate(y_true, y_pred, y_pred_prob):
    """
    Computes the Accuracy, ROC_AUC Score and F1 Score for the predicted values.

    Parameters
    ----------
    y_true:
        Ground truth (correct) labels.

    y_pred:
        Predicted labels, as returned by a classifier.

    y_pred_prob:
        Predicted probabilites for each label, as returned by a classifier.


    Returns
    -------
    acc: float
        Accuracy Score as a percentage.

    roc_auc: float
        ROC AUC Score for positive label, as a percentage.

    f1_score: float
        F1 Score as a percentage.
    """

    acc = round(accuracy_score(y_true, y_pred), 2)
    roc_auc = round(roc_auc_score(y_true, y_pred_prob[:, 1]), 2)
    f1 = round(f1_score(y_true, y_pred), 2)

    print(f"Accuracy score: {acc * 100} %")
    print(f"ROC AUC Score: {roc_auc * 100} %")
    print(f"F1 Score: {f1 * 100} %")
    print("Classification Report: \n", classification_report(y_true, y_pred))

    return acc, roc_auc, f1


def evaluate_cv(scores):
    """
    Computes the average Accuracy, ROC_AUC Score and F1 Score (Weighted) for
    the predicted values over all the folds.

    Parameters
    ----------
    scores: dictionary
        Dictionary of scoring metrics over each fold.


    Returns
    -------
    acc: float
        Average Accuracy Score over all the folds, as a percentage.

    roc_auc: float
        Average ROC AUC Score over all the folds, as a percentage.

    f1_score: float
        Weighted F1 Score over all the folds, as a percentage.
    """

    acc = round(np.mean(scores["test_accuracy"]) * 100, 2)
    f1_weighted = round(np.mean(scores["test_f1_weighted"]) * 100, 2)
    roc_auc = round(np.mean(scores["test_roc_auc"]) * 100, 2)

    print(f"Test Accuracy: {acc}%")
    print(f"Test F1-Weighted: {f1_weighted}%")
    print(f"Test ROC-AUC: {roc_auc}%")

    return acc, f1_weighted, roc_auc
