import numpy as np
import pandas as pd
import shap
from scipy import sparse

from src.features import CAT


def _dense(a):
    return a.toarray() if sparse.issparse(a) else np.asarray(a)


def original_name(feat):
    """'cat__Contract_One year' -> 'Contract'; 'num__tenure' -> 'tenure'."""
    kind, rest = feat.split("__", 1)
    if kind == "cat":
        for c in CAT:
            if rest.startswith(c + "_"):
                return c
    return rest


def shap_by_feature(pipe, X):
    """SHAP values (log-odds of churn) summed back to the original columns.
    Returns (DataFrame indexed like X, base_value)."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    A = _dense(pre.transform(X))
    names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(A)
    groups = [original_name(n) for n in names]
    df = pd.DataFrame(sv, columns=names, index=X.index).T.groupby(groups).sum().T
    base = float(np.ravel(explainer.expected_value)[0])
    return df, base


def top_reasons(shap_row, x_row, n=3):
    """Top n features moving this customer's risk up (+) or down (-)."""
    s = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)[:n]
    out = []
    for f, v in s.items():
        val = x_row[f]
        val = val.item() if hasattr(val, "item") else val
        out.append((f, val, round(float(v), 3)))
    return out