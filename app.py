from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Salary Data.csv").dropna()

job_titles = sorted(data["Job Title"].unique())
education_levels = sorted(data["Education Level"].unique())
genders = sorted(data["Gender"].unique())

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("salary_model.pkl", "rb"))

MODEL_ACCURACY = 92.4
POLY_DEGREE = 2


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        job_titles=job_titles,
        education_levels=education_levels,
        genders=genders,
        accuracy=MODEL_ACCURACY,
        degree=POLY_DEGREE,
        form_data={}
    )


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    form_data = {
        "age": float(request.form["age"]),
        "experience": float(request.form["experience"]),
        "education": request.form["education"],
        "job": request.form["job"],
        "gender": request.form.get("gender", "Male")
    }

    input_df = pd.DataFrame([{
        "Age": form_data["age"],
        "Years of Experience": form_data["experience"],
        "Education Level": form_data["education"],
        "Job Title": form_data["job"],
        "Gender": form_data["gender"]
    }])

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Salary: ₹{prediction:,.2f}",
        job_titles=job_titles,
        education_levels=education_levels,
        genders=genders,
        accuracy=MODEL_ACCURACY,
        degree=POLY_DEGREE,
        form_data=form_data
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
# run: (gunicorn app:app --bind 0.0.0.0:5000)

