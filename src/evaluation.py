import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import make_preprocessor

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def make_cv(seed=42):
    return StratifiedKFold(5, shuffle=True, random_state=seed)


def cv_report(name, clf, X, y, cv=None):
    """5-fold CV of preprocessor + classifier; returns one summary row."""
    cv = cv or make_cv()
    pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)])
    s = cross_validate(pipe, X, y, cv=cv, scoring=SCORING,
                       return_train_score=True, n_jobs=-1)
    return {"model": name,
            "train_AUC": s["train_roc_auc"].mean(),
            "CV_AUC": s["test_roc_auc"].mean(), "CV_AUC_std": s["test_roc_auc"].std(),
            "CV_PR_AUC": s["test_pr_auc"].mean(), "CV_PR_std": s["test_pr_auc"].std()}


def to_table(rows):
    return pd.DataFrame(rows).round(4)