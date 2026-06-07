import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# DATASET
df = pd.read_csv("Titanic-Dataset.csv")

df["Age"] = SimpleImputer(strategy="median").fit_transform(df[["Age"]])

if "Embarked" in df.columns:
    df["Embarked"] = SimpleImputer(strategy="most_frequent").fit_transform(
        df[["Embarked"]]
    ).ravel()

sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

df["Sex"] = sex_encoder.fit_transform(df["Sex"])
df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

# Features 
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# Train/Test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# PREDICTION FUNCTION
def predict_survival():
    try:
        name = name_entry.get().strip()

        if not name:
            messagebox.showerror("Error", "Please enter passenger name.")
            return

        age = float(age_entry.get())
        pclass = int(class_var.get())
        gender = gender_var.get()

        sex = 1 if gender == "male" else 0

        # Default values
        sibsp = 0
        parch = 0
        fare = 32.0
        embarked = 2

        passenger = [[
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked
        ]]

        prediction = model.predict(passenger)[0]
        probability = model.predict_proba(passenger)[0]

        if prediction == 1:
            result_label.config(
                fg="#00ff88",
                text=f"""
Passenger: {name}

Prediction: SURVIVED

Chance of Survival:
{probability[1]*100:.2f}%
"""
            )
        else:
            result_label.config(
                fg="#ff6b6b",
                text=f"""
Passenger: {name}

Prediction: DID NOT SURVIVE

Risk:
{probability[0]*100:.2f}%
"""
            )

    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Please enter a valid age."
        )

# UI
root = tk.Tk()
root.title("Titanic Survival Predictor")
root.geometry("700x600")
root.configure(bg="#1e1e2f")
root.resizable(False, False)

title = tk.Label(
    root,
    text="🚢 Titanic Survival Prediction",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#1e1e2f"
)
title.pack(pady=20)

subtitle = tk.Label(
    root,
    text="Machine Learning Based Prediction System",
    font=("Segoe UI", 11),
    fg="lightgray",
    bg="#1e1e2f"
)
subtitle.pack()

card = tk.Frame(
    root,
    bg="#2b2b40",
    padx=30,
    pady=25
)
card.pack(pady=25)

# Name
tk.Label(
    card,
    text="Passenger Name",
    font=("Segoe UI", 11),
    fg="white",
    bg="#2b2b40"
).grid(row=0, column=0, pady=10, padx=10)

name_entry = tk.Entry(
    card,
    width=25,
    font=("Segoe UI", 11)
)
name_entry.grid(row=0, column=1)

# Class
tk.Label(
    card,
    text="Passenger Class",
    font=("Segoe UI", 11),
    fg="white",
    bg="#2b2b40"
).grid(row=1, column=0, pady=10)

class_var = tk.StringVar(value="3")

class_menu = tk.OptionMenu(
    card,
    class_var,
    "1",
    "2",
    "3"
)
class_menu.config(width=15)
class_menu.grid(row=1, column=1)

# Gender
tk.Label(
    card,
    text="Gender",
    font=("Segoe UI", 11),
    fg="white",
    bg="#2b2b40"
).grid(row=2, column=0, pady=10)

gender_var = tk.StringVar(value="male")

gender_menu = tk.OptionMenu(
    card,
    gender_var,
    "male",
    "female"
)
gender_menu.config(width=15)
gender_menu.grid(row=2, column=1)

# Age
tk.Label(
    card,
    text="Age",
    font=("Segoe UI", 11),
    fg="white",
    bg="#2b2b40"
).grid(row=3, column=0, pady=10)

age_entry = tk.Entry(
    card,
    width=25,
    font=("Segoe UI", 11)
)
age_entry.grid(row=3, column=1)

# Predict Button
predict_btn = tk.Button(
    root,
    text="Predict Survival",
    command=predict_survival,
    bg="#4CAF50",
    fg="white",
    font=("Segoe UI", 12, "bold"),
    padx=20,
    pady=10,
    borderwidth=0
)
predict_btn.pack(pady=15)

# Accuracy
accuracy_label = tk.Label(
    root,
    text=f"Model Accuracy: {accuracy*100:.2f}%",
    font=("Segoe UI", 10),
    fg="#90EE90",
    bg="#1e1e2f"
)
accuracy_label.pack()

# Result Box
result_label = tk.Label(
    root,
    text="Enter details and click Predict Survival",
    font=("Segoe UI", 12),
    fg="white",
    bg="#2b2b40",
    width=40,
    height=8,
    relief="ridge"
)
result_label.pack(pady=20)

root.mainloop()