from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("heart_disease_model.pkl")

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal"]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    form = {f: "" for f in FEATURES}

    if request.method == "POST":
        try:
            # Read inputs
            for f in FEATURES:
                form[f] = request.form.get(f, "")

            data = dict(form)

            # Convert types
            data["age"] = float(data["age"])
            data["trestbps"] = float(data["trestbps"])
            data["chol"] = float(data["chol"])
            data["thalach"] = float(data["thalach"])
            data["oldpeak"] = float(data["oldpeak"])

            for c in ["sex","cp","fbs","restecg","exang","slope","ca","thal"]:
                data[c] = int(data[c])

            input_df = pd.DataFrame([data], columns=FEATURES)

            pred = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(input_df)[0][1])

            if pred == 1:
                result = "HIGH RISK: Heart Disease (YES)"
            else:
                result = "LOW RISK: Heart Disease (NO)"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, probability=probability, form=form)

if __name__ == "__main__":
    app.run(debug=True)
