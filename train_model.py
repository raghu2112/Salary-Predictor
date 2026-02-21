import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("Salary Data.csv")

# Drop missing rows (important)
data = data.dropna()

# Features and target
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Categorical & numerical columns
categorical_cols = ["Gender", "Education Level", "Job Title"]
numeric_cols = ["Age", "Years of Experience"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# Pipeline: preprocessing → polynomial → regression
model = Pipeline([
    ("preprocess", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X, y)

# Save model
with open("salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")