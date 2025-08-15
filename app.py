from flask import Flask, render_template, request, send_file, flash
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # replace with a secure key
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pipeline
pipeline = joblib.load("heart_disease_pipeline.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]
columns = pipeline["columns"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    download_link = None

    if request.method == "POST":
        # Bulk CSV upload
        if 'csvfile' in request.files and request.files['csvfile'].filename != '':
            file = request.files['csvfile']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                df_bulk = pd.read_csv(filepath)
                df_bulk_orig = df_bulk.copy()
                df_bulk_encoded = pd.get_dummies(df_bulk).reindex(columns=columns, fill_value=0)
                bulk_scaled = scaler.transform(df_bulk_encoded)
                bulk_preds = model.predict(bulk_scaled)
                if hasattr(model, "predict_proba"):
                    bulk_probs = model.predict_proba(bulk_scaled)
                else:
                    bulk_probs = None

                results = []
                for i, p in enumerate(bulk_preds):
                    if bulk_probs is not None:
                        conf = bulk_probs[i][p] * 100
                        results.append(f"{'Heart Disease' if p == 1 else 'No Heart Disease'} ({conf:.2f}%)")
                    else:
                        results.append('Heart Disease' if p == 1 else 'No Heart Disease')

                df_bulk_orig['Prediction'] = results
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bulk_results.csv')
                df_bulk_orig.to_csv(result_path, index=False)
                download_link = 'bulk_results.csv'
                prediction = f"{len(results)} patient(s) processed. Download results below."
            except Exception as e:
                prediction = f"Error processing CSV file: {e}"

        else:
            # Single patient prediction logic here (optional)
            # You can keep your existing single patient code here

            pass

    return render_template("index.html", prediction=prediction, confidence=confidence, download_link=download_link)

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
