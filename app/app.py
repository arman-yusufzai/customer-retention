"""Customer retention app: churn probability, segment, expected benefit,
SHAP reasons and a recommended action.

Run from the project root:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.business import expected_benefit
from src.explain import shap_by_feature, top_reasons
from src.features import add_features
from src.segments import assign_segments, name_segment, segment_matrix

try:
    from src.segments import segment_names
except ImportError:                      # older segments.py without the helper
    segment_names = None

st.set_page_config(page_title="Customer Retention", page_icon="📞", layout="wide")

# ----------------------------------------------------------------- constants
YES_NO = ["No", "Yes"]
ADDON_OPTS = ["No", "Yes", "No internet service"]
OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": YES_NO,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ADDON_OPTS,
    "OnlineBackup": ADDON_OPTS,
    "DeviceProtection": ADDON_OPTS,
    "TechSupport": ADDON_OPTS,
    "StreamingTV": ADDON_OPTS,
    "StreamingMovies": ADDON_OPTS,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]

# default customer: 6 months, fiber, month-to-month, $75, electronic check
DEFAULTS = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 6, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0,
}
FIELDS = list(DEFAULTS)

# SHAP feature -> retention action (first match among the customer's risk-raising drivers)
ACTIONS = {
    "Contract": "Offer a discounted one- or two-year plan",
    "tenure": "Schedule an onboarding / check-in call",
    "is_new_customer": "Schedule an onboarding / check-in call",
    "OnlineSecurity": "Offer a free trial of a security / tech-support bundle",
    "TechSupport": "Offer a free trial of a security / tech-support bundle",
    "has_support_addons": "Offer a free trial of a security / tech-support bundle",
    "OnlineBackup": "Offer a free trial of a security / tech-support bundle",
    "DeviceProtection": "Offer a free trial of a security / tech-support bundle",
    "addon_count": "Offer a free trial of a security / tech-support bundle",
    "PaymentMethod": "Offer an incentive to switch to automatic payment",
    "PaperlessBilling": "Offer an incentive to switch to automatic payment",
    "MonthlyCharges": "Review the plan and price with the customer",
    "InternetService": "Review the plan and price with the customer",
}


# ------------------------------------------------------------------- loaders
@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "models" / "churn_model.joblib")
    config = json.load(open(ROOT / "models" / "model_config.json"))
    scaler, km = joblib.load(ROOT / "models" / "segments.joblib")
    if segment_names is not None:
        names = segment_names(scaler, km)
    else:                                  # same result, computed here
        cols = list(segment_matrix(pd.DataFrame({
            "tenure": [0], "MonthlyCharges": [0], "addon_count": [0],
            "streaming_flag": [0], "has_support_addons": [0],
            "Contract": ["Month-to-month"], "InternetService": ["DSL"]})).columns)
        centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=cols)
        names = {int(i): name_segment(r) for i, r in centers.iterrows()}
    return model, config, scaler, km, names


@st.cache_data
def load_samples():
    path = ROOT / "app" / "sample_customers.csv"
    return pd.read_csv(path) if path.exists() else None


try:
    model, config, scaler, km, seg_names = load_assets()
except FileNotFoundError as e:
    st.error(f"Missing model file: {e.filename}\n\nRun notebooks 06 and 08 first "
             "(they create models/segments.joblib, models/churn_model.joblib and "
             "models/model_config.json).")
    st.stop()

samples = load_samples()

# ------------------------------------------------------------- session state
for f in FIELDS:
    st.session_state.setdefault(f"f_{f}", DEFAULTS[f])
st.session_state.setdefault("pick", "Custom")


def _to_widget(field, value):
    if field == "SeniorCitizen":
        return "Yes" if int(value) == 1 else "No"
    if field == "tenure":
        return int(value)
    if field == "MonthlyCharges":
        return float(value)
    return value


def on_pick():
    """Load the chosen real customer into the form (or restore the default)."""
    choice = st.session_state["pick"]
    if choice == "Custom":
        return
    row = samples.iloc[int(choice.split("#")[1]) - 1]
    for f in FIELDS:
        st.session_state[f"f_{f}"] = _to_widget(f, row[f])


def on_edit():
    """Any manual edit makes it a custom customer, so 'reveal' can't mislead."""
    st.session_state["pick"] = "Custom"


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Customer")
    pick_opts = ["Custom"]
    if samples is not None:
        pick_opts += [f"Real customer #{i + 1}" for i in range(len(samples))]
    st.selectbox("Start from", pick_opts, key="pick", on_change=on_pick,
                 help="Real held-out customers the model never trained on.")

    st.header("Business assumptions")
    st.caption("Illustrative numbers, not real company data.")
    save_rate = st.slider("Retention success rate", 0.05, 0.60,
                          float(config.get("save_rate", 0.25)), 0.05,
                          help="Share of contacted would-be churners the offer saves.")
    offer_cost = st.slider("Cost per contact ($)", 5, 200,
                           int(config.get("offer_cost", 50)), 5)
    st.caption("A saved customer is worth 12 months of their monthly charges.")

st.title("📞 Customer retention assistant")
st.caption("Predicts churn risk, explains it, and says whether a retention offer is worth it.")

# --------------------------------------------------------------------- form
col_in, col_out = st.columns([1, 1.25], gap="large")

with col_in:
    st.subheader("Customer details")
    ss = st.session_state
    a, b = st.columns(2)
    with a:
        st.selectbox("Gender", OPTIONS["gender"], key="f_gender", on_change=on_edit)
        st.selectbox("Partner", OPTIONS["Partner"], key="f_Partner", on_change=on_edit)
        st.selectbox("Senior citizen", OPTIONS["SeniorCitizen"], key="f_SeniorCitizen", on_change=on_edit)
    with b:
        st.selectbox("Dependents", OPTIONS["Dependents"], key="f_Dependents", on_change=on_edit)
        st.selectbox("Phone service", OPTIONS["PhoneService"], key="f_PhoneService", on_change=on_edit)
        st.selectbox("Multiple lines", OPTIONS["MultipleLines"], key="f_MultipleLines", on_change=on_edit)

    st.slider("Tenure (months)", 0, 72, key="f_tenure", on_change=on_edit)
    st.number_input("Monthly charges ($)", min_value=15.0, max_value=130.0, step=0.5,
                    key="f_MonthlyCharges", on_change=on_edit)

    st.selectbox("Contract", OPTIONS["Contract"], key="f_Contract", on_change=on_edit)
    st.selectbox("Internet service", OPTIONS["InternetService"], key="f_InternetService", on_change=on_edit)

    c, d = st.columns(2)
    with c:
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection"]:
            st.selectbox(col, OPTIONS[col], key=f"f_{col}", on_change=on_edit)
    with d:
        for col in ["TechSupport", "StreamingTV", "StreamingMovies"]:
            st.selectbox(col, OPTIONS[col], key=f"f_{col}", on_change=on_edit)

    st.selectbox("Paperless billing", OPTIONS["PaperlessBilling"], key="f_PaperlessBilling", on_change=on_edit)
    st.selectbox("Payment method", OPTIONS["PaymentMethod"], key="f_PaymentMethod", on_change=on_edit)

# --------------------------------------------------- build the model input row
vals = {f: st.session_state[f"f_{f}"] for f in FIELDS}
notes = []

# fix contradictory inputs
if vals["PhoneService"] == "No" and vals["MultipleLines"] != "No phone service":
    vals["MultipleLines"] = "No phone service"
    notes.append("No phone service → multiple lines set to 'No phone service'.")
if vals["PhoneService"] == "Yes" and vals["MultipleLines"] == "No phone service":
    vals["MultipleLines"] = "No"
    notes.append("Phone service is on → multiple lines set to 'No'.")
if vals["InternetService"] == "No":
    if any(vals[c] != "No internet service" for c in ADDON_COLS):
        notes.append("No internet → all add-ons set to 'No internet service'.")
    for c in ADDON_COLS:
        vals[c] = "No internet service"
else:
    if any(vals[c] == "No internet service" for c in ADDON_COLS):
        notes.append("Customer has internet → 'No internet service' add-ons set to 'No'.")
    for c in ADDON_COLS:
        if vals[c] == "No internet service":
            vals[c] = "No"

row = dict(vals)
row["SeniorCitizen"] = 1 if vals["SeniorCitizen"] == "Yes" else 0
X = add_features(pd.DataFrame([row]))[config["features"]]

# ------------------------------------------------------------------ predict
p = float(model.predict_proba(X)[0, 1])
seg_id = int(assign_segments(X, scaler, km)[0])
seg = seg_names.get(seg_id, f"Segment {seg_id}")
benefit = float(expected_benefit(p, X["MonthlyCharges"].values, save_rate, offer_cost)[0])

sv, _ = shap_by_feature(model, X)
reasons = top_reasons(sv.iloc[0], X.iloc[0], n=8)

if p >= 0.50:
    risk, icon = "High", "🔴"
elif p >= 0.25:
    risk, icon = "Medium", "🟠"
else:
    risk, icon = "Low", "🟢"

# the action comes from the customer's own biggest risk-raising drivers
action = None
for feat, _, shap_val in sorted(reasons, key=lambda r: -r[2]):
    if shap_val > 0 and feat in ACTIONS:
        action = ACTIONS[feat]
        break
if action is None:
    action = "Send a general check-in / loyalty message"

# ------------------------------------------------------------------ results
with col_out:
    st.subheader("Prediction")
    for n in notes:
        st.info(n)

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{p:.1%}")
    m2.metric("Risk level", f"{icon} {risk}")
    m3.metric("Expected benefit", f"${benefit:,.0f}")
    st.write(f"**Segment:** {seg}")
    st.caption("Risk bands (High ≥ 50%, Medium 25–50%) are a simple display choice. "
               "The contact decision uses the expected benefit.")

    st.markdown("#### Recommendation")
    if benefit > 0:
        st.success(f"**Contact this customer.** {action}.\n\n"
                   f"Expected net benefit: **${benefit:,.0f}** "
                   f"({p:.1%} × {save_rate:.0%} save rate × 12 × ${X['MonthlyCharges'].iloc[0]:,.2f}"
                   f" − ${offer_cost} cost).")
    else:
        st.warning(f"**Do not contact.** Expected benefit is ${benefit:,.0f}: the offer "
                   f"costs more than the revenue it is expected to save at these assumptions.")

    st.markdown("#### Why? Top drivers of this prediction")
    labels = [f"{f} = {v:.0f}" if isinstance(v, float) and float(v).is_integer()
              else f"{f} = {v:.2f}" if isinstance(v, float) else f"{f} = {v}"
              for f, v, _ in reasons]
    shap_vals = [s for _, _, s in reasons]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(labels[::-1], shap_vals[::-1],
            color=["#d62728" if s > 0 else "#1f77b4" for s in shap_vals[::-1]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect on churn risk (log-odds):  red raises risk, blue lowers it")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("SHAP shows what the model uses, not what causes churn.")

    # only meaningful for an untouched real customer
    pick = st.session_state["pick"]
    if pick != "Custom" and samples is not None:
        if st.checkbox("Reveal what actually happened", key="reveal"):
            churned = int(samples.iloc[int(pick.split("#")[1]) - 1]["churned"])
            if churned:
                st.error("This customer **churned**.")
            else:
                st.success("This customer **stayed**.")