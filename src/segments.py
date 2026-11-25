import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CONTRACT_LEN = {"Month-to-month": 0, "One year": 1, "Two year": 2}


def segment_matrix(X):
    """Behavioural / product-utilization features used for clustering."""
    S = pd.DataFrame(index=X.index)
    S["tenure"] = X["tenure"]
    S["MonthlyCharges"] = X["MonthlyCharges"]
    S["addon_count"] = X["addon_count"]
    S["streaming_flag"] = X["streaming_flag"]
    S["has_support_addons"] = X["has_support_addons"]
    S["contract_len"] = X["Contract"].map(CONTRACT_LEN)
    S["fiber"] = (X["InternetService"] == "Fiber optic").astype(int)
    S["no_internet"] = (X["InternetService"] == "No").astype(int)
    return S


def fit_segments(X_train, k=4, seed=42):
    S = segment_matrix(X_train)
    scaler = StandardScaler().fit(S)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(S))
    return scaler, km


def assign_segments(X, scaler, km):
    return km.predict(scaler.transform(segment_matrix(X)))


def name_segment(r):
    """Rule-based names, so they stay correct even if cluster numbers change."""
    if r["no_internet"] > 0.5:
        return "No-internet / phone only"
    if r["tenure"] > 45:
        return "Loyal power users"
    if r["fiber"] > 0.5:
        return "New fiber, month-to-month"
    return "Mid-value, month-to-month"


# Column order of segment_matrix(); needed to read the K-means centroids.
SEGMENT_COLS = ["tenure", "MonthlyCharges", "addon_count", "streaming_flag",
                "has_support_addons", "contract_len", "fiber", "no_internet"]


def segment_names(scaler, km):
    """{cluster_id: name}. A K-means centroid is the mean of its members, so
    un-scaling the centroids gives the same profile the notebooks print."""
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),
                           columns=SEGMENT_COLS)
    return {int(i): name_segment(r) for i, r in centers.iterrows()}