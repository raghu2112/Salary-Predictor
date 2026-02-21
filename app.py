from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# ---------------- LOAD DATASET ----------------
data = pd.read_csv("Salary Data.csv").dropna()

# Extract dropdown values dynamically from dataset
job_titles = sorted(data["Job Title"].unique())
education_levels = sorted(data["Education Level"].unique())
genders = sorted(data["Gender"].unique())

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("salary_model.pkl", "rb"))

# You can replace these with real computed values if available
MODEL_ACCURACY = 92.4     # example value
POLY_DEGREE = 2


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        job_titles=job_titles,
        education_levels=education_levels,
        genders=genders,
        accuracy=MODEL_ACCURACY,
        degree=POLY_DEGREE
    )


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    # Collect form inputs
    age = float(request.form["age"])
    experience = float(request.form["experience"])
    education = request.form["education"]
    job = request.form["job"]

    # If gender added later in UI, uncomment this:
    # gender = request.form["gender"]

    # Create dataframe for prediction
    input_df = pd.DataFrame([{
        "Age": age,
        "Years of Experience": experience,
        "Education Level": education,
        "Job Title": job,
        "Gender": "Male"   # default if not in form; change if you add gender input
    }])

    # Predict salary
    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Salary: ₹{prediction:,.2f}",
        job_titles=job_titles,
        education_levels=education_levels,
        genders=genders,
        accuracy=MODEL_ACCURACY,
        degree=POLY_DEGREE
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)