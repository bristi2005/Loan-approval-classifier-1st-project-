from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/predict")
def predict():
    return render_template("model.html")

@app.route("/output", methods=["GET", "POST"])
def output():
    if request.method == "POST":
        try:
            # Get form inputs safely (matching names in HTML)
            Married = request.form.get("Married")
            Dependents = request.form.get("Dependents")
            Education = request.form.get("Education")
            Self_Employed = request.form.get("Self_Employed")
            ApplicantIncome = float(request.form.get("ApplicantIncome") or 0)
            CoapplicantIncome = float(request.form.get("CoapplicantIncome") or 0)
            LoanAmount = float(request.form.get("LoanAmount") or 0) / 1000
            Loan_Amount_Term = float(request.form.get("Loan_Amount_Term") or 360)
            Credit_History = float(request.form.get("Credit_History") or 1)
            Property_Area = request.form.get("Property_Area")

            # Create DataFrame
            user_input_df = pd.DataFrame({
                "Married": [Married],
                "Dependents": [Dependents],
                "Education": [Education],
                "Self_Employed": [Self_Employed],
                "ApplicantIncome": [ApplicantIncome],
                "CoapplicantIncome": [CoapplicantIncome],
                "LoanAmount": [LoanAmount],
                "Loan_Amount_Term": [Loan_Amount_Term],
                "Credit_History": [Credit_History],
                "Property_Area": [Property_Area]
            })

            # Load trained model
            with open("model.pkl", "rb") as f:
                pipe = pickle.load(f)

            # Make prediction
            prediction = pipe.predict(user_input_df)[0]

            return render_template("result.html", value=prediction)

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
