import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
NUM = ["tenure", "MonthlyCharges", "addon_count"]
BIN = ["SeniorCitizen", "has_support_addons", "streaming_flag", "is_new_customer"]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
       "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
       "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
       "PaperlessBilling", "PaymentMethod"]


def load_and_clean(path):
    df = pd.read_csv(path)
    # 11 blank TotalCharges are all tenure-0 customers (not billed yet) -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["Churn"])


def add_features(df):
    """Engagement / product-utilization proxies built from the real columns."""
    df = df.copy()
    df["addon_count"] = sum((df[c] == "Yes").astype(int) for c in ADDONS)
    df["has_support_addons"] = ((df["TechSupport"] == "Yes") |
                                (df["OnlineSecurity"] == "Yes")).astype(int)
    df["streaming_flag"] = ((df["StreamingTV"] == "Yes") |
                            (df["StreamingMovies"] == "Yes")).astype(int)
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    return df


def get_splits(path, test_size=0.2, seed=42):
    """Clean, engineer features, drop ID and TotalCharges, stratified split."""
    df = add_features(load_and_clean(path)).drop(columns=["customerID", "TotalCharges"])
    X, y = df.drop(columns="churn"), df["churn"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("bin", "passthrough", BIN),
    ])