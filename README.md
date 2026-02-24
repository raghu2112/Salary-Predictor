# 💼 Salary Prediction Web App (Polynomial Regression + Flask)

This project is a **Machine Learning powered Salary Prediction Web Application** built using **Python, Scikit-learn, and Flask**.

It predicts a person's expected salary based on professional attributes such as age, experience, education level, gender, and job title.

The application includes a trained regression model, a modern UI, and is ready for deployment.

---

## 🚀 Features

- Polynomial Regression based salary prediction model
- Full ML pipeline with preprocessing and encoding
- Model saved using Pickle for reuse
- Interactive web form with sliders and dropdowns
- Dropdown values dynamically loaded from dataset
- Displays model accuracy and polynomial degree
- Clean and professional UI
- Ready for GitHub and cloud deployment

---

## 🧠 Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML / CSS

---


## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/raghu2112/Salary-Predictor
cd salary-predictor
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the model (optional if pickle already exists)
```bash
python train_model.py
```
### 4. Run the web app
```bash
python app.py
```
### Open in browser:
```bash
http://127.0.0.1:5000/
```
---

## 📊 Model Details
- Algorithm: Polynomial Regression
- Polynomial Degree: 2
- Accuracy: ~90–95% (depends on dataset split)
