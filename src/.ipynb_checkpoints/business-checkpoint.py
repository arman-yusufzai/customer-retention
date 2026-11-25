import numpy as np
import pandas as pd


def realized_net(y, monthly, contact, save_rate, offer_cost):
    """Dollar outcome of contacting the customers in `contact` (boolean array).
    A contacted customer who really churns is saved with prob `save_rate`
    (worth 12 months of revenue); every contacted customer costs `offer_cost`."""
    y, monthly, contact = np.asarray(y), np.asarray(monthly), np.asarray(contact)
    gain = y * save_rate * monthly * 12
    return float((gain - offer_cost)[contact].sum())


def expected_benefit(p, monthly, save_rate, offer_cost):
    """Per-customer expected net benefit of contacting, from predicted churn prob p."""
    return np.asarray(p) * save_rate * np.asarray(monthly) * 12 - offer_cost


def gains_table(y, p, bins=10):
    """Decile table: customers sorted by risk, with lift and cumulative capture."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    d["decile"] = np.ceil(np.arange(1, len(d) + 1) / len(d) * bins).astype(int)
    g = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"),
                                avg_pred=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / d["y"].mean()
    g["cum_capture"] = g["churners"].cumsum() / d["y"].sum()
    return g