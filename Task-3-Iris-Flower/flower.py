import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset

df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop("Id", axis=1)

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

# Prediction

def predict_species():
    try:
        sepal_length = float(sepal_length_entry.get())
        sepal_width = float(sepal_width_entry.get())
        petal_length = float(petal_length_entry.get())
        petal_width = float(petal_width_entry.get())

        sample = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        species = model.predict(sample)[0]
        probability = max(model.predict_proba(sample)[0]) * 100

        flower_emoji = {
            "Iris-setosa": "🌸",
            "Iris-versicolor": "🌺",
            "Iris-virginica": "🌷",
            "setosa": "🌸",
            "versicolor": "🌺",
            "virginica": "🌷"
        }

        result_label.config(
            text=f"""
{flower_emoji.get(species, "🌼")}

Predicted species

{species}

Confidence:
{probability:.2f}%
""",
            fg="#00FF99"
        )

    except:
        messagebox.showerror(
            "Error",
            "Please enter valid values."
        )

# GUI

root = tk.Tk()
root.title("Iris Flower Classification")
root.geometry("750x700")
root.configure(bg="#0F172A")
root.resizable(False, False)

title = tk.Label(
    root,
    text="🌸 Iris Flower Classification",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#0F172A"
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="Machine Learning using Random Forest Classifier",
    font=("Segoe UI", 10),
    fg="#CBD5E1",
    bg="#0F172A"
)
subtitle.pack()

card = tk.Frame(
    root,
    bg="#1E293B",
    padx=30,
    pady=25
)
card.pack(pady=25)

tk.Label(
    card,
    text="Sepal Length",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=0, column=0, pady=10, sticky="w")

sepal_length_entry = tk.Entry(
    card,
    width=25,
    font=("Segoe UI", 11)
)
sepal_length_entry.grid(row=0, column=1)

tk.Label(
    card,
    text="Sepal Width",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=1, column=0, pady=10, sticky="w")

sepal_width_entry = tk.Entry(
    card,
    width=25,
    font=("Segoe UI", 11)
)
sepal_width_entry.grid(row=1, column=1)

tk.Label(
    card,
    text="Petal Length",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=2, column=0, pady=10, sticky="w")

petal_length_entry = tk.Entry(
    card,
    width=25,
    font=("Segoe UI", 11)
)
petal_length_entry.grid(row=2, column=1)

tk.Label(
    card,
    text="Petal Width",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=3, column=0, pady=10, sticky="w")

petal_width_entry = tk.Entry(
    card,
    width=25,
    font=("Segoe UI", 11)
)
petal_width_entry.grid(row=3, column=1)

predict_button = tk.Button(
    root,
    text="Predict Species",
    command=predict_species,
    bg="#10B981",
    fg="white",
    font=("Segoe UI", 12, "bold"),
    padx=20,
    pady=10,
    borderwidth=0
)
predict_button.pack(pady=15)

accuracy_label = tk.Label(
    root,
    text=f"Model Accuracy: {accuracy * 100:.2f}%",
    bg="#0F172A",
    fg="#93C5FD",
    font=("Segoe UI", 10)
)
accuracy_label.pack()

result_label = tk.Label(
    root,
    text="Enter flower measurements and click Predict Species",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 13),
    width=45,
    height=14,
    relief="ridge",
    justify="center"
)
result_label.pack(pady=25)

root.mainloop()