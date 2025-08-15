import joblib
import pandas as pd

# Load saved pipeline
pipeline = joblib.load("heart_disease_pipeline.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]
columns = pipeline["columns"]

# ===== OPTION 1: Single patient prediction =====
single_patient = {
    "age": 55,
    "sex": "Female",
    "dataset": "Cleveland",
    "cp": "asymptomatic",
    "trestbps": 140.0,
    "chol": 250.0,
    "fbs": False,
    "restecg": "normal",
    "thalch": 160.0,
    "exang": True,
    "oldpeak": 1.2,
    "slope": "flat",
    "ca": 0.0,
    "thal": "normal"
}

# Encode and align with training columns
df_single = pd.DataFrame([single_patient])
df_single = pd.get_dummies(df_single).reindex(columns=columns, fill_value=0)

# Scale features
single_scaled = scaler.transform(df_single)

# Predict and get probability
pred = model.predict(single_scaled)[0]
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(single_scaled)[0][pred] * 100
    print(f"\nSingle Patient Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'} ({prob:.2f}% confidence)")
else:
    print(f"\nSingle Patient Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'} (model does not support predict_proba)")

# ===== OPTION 2: Bulk prediction from CSV =====
try:
    df_bulk = pd.read_csv("patients.csv")  # raw patient data
    df_bulk_encoded = pd.get_dummies(df_bulk).reindex(columns=columns, fill_value=0)
    bulk_scaled = scaler.transform(df_bulk_encoded)

    bulk_preds = model.predict(bulk_scaled)
    if hasattr(model, "predict_proba"):
        bulk_probs = model.predict_proba(bulk_scaled)
    else:
        bulk_probs = None

    # Add results to dataframe
    results = []
    for i, p in enumerate(bulk_preds):
        if bulk_probs is not None:
            confidence = bulk_probs[i][p] * 100
            results.append(f"{'Heart Disease' if p == 1 else 'No Heart Disease'} ({confidence:.2f}%)")
        else:
            results.append("Heart Disease" if p == 1 else "No Heart Disease")

    df_bulk["Prediction"] = results

    print("\nBulk Predictions:\n", df_bulk[["Prediction"]])
    df_bulk.to_csv("patients_with_predictions.csv", index=False)
    print("\n💾 Saved predictions with confidence to 'patients_with_predictions.csv'")

except FileNotFoundError:
    print("\nNo CSV file found for bulk predictions. Skipping bulk mode.")
