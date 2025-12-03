import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

shap.initjs()

CSV_PATH = r"D:\graduation project\cardio_train.csv"

FEATURE_NAMES = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

LABEL_NAME = 'cardio'
MODEL_PATH = "cardio_model.pkl"

def load_data():
    df = pd.read_csv(CSV_PATH, sep=';')
    return df

def ensure_model():
    try:
        model = joblib.load(MODEL_PATH)
        explainer = joblib.load("explainer.pkl")
        return model, explainer
    except:
        df = load_data()
        X = df[FEATURE_NAMES].values
        y = df[LABEL_NAME].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(explainer, "explainer.pkl")

        return model, explainer

model, explainer = ensure_model()

st.title("Cardiovascular Risk Prediction App")

st.subheader("Patient Input Form")
user_input = {}
for feature in FEATURE_NAMES:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict"):
    input_array = np.array([list(user_input.values())])

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    st.write(f"### Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
    st.write(f"### Probability: {probability:.4f}")

    shap_values = explainer.shap_values(input_array)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    st.subheader("SHAP Feature Contribution")

    fig = plt.figure(figsize=(8,5))
    shap.summary_plot(sv, input_array, feature_names=FEATURE_NAMES, plot_type="bar", show=False)
    st.pyplot(plt.gcf())

st.subheader("Batch Processing (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV with same columns", type=['csv'])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file, sep=';')
    X_batch = batch_df[FEATURE_NAMES].values

    preds = model.predict(X_batch)
    probs = model.predict_proba(X_batch)[:, 1]

    batch_df['prediction'] = preds
    batch_df['risk_score'] = probs

    st.write(batch_df)
