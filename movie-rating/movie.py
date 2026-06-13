import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Dataset

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

df = df[
    [
        "Genre",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3",
        "Rating"
    ]
]

df = df.dropna(subset=["Rating"])

X = df.drop("Rating", axis=1)
y = df["Rating"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            ),
            ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
        )
    ]
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

score = r2_score(y_test, model.predict(X_test))

# Prediction

def predict_rating():
    try:
        movie = movie_entry.get().strip()
        genre = genre_entry.get().strip()
        director = director_entry.get().strip()
        actor1 = actor1_entry.get().strip()
        actor2 = actor2_entry.get().strip()
        actor3 = actor3_entry.get().strip()

        if not all([movie, genre, director, actor1, actor2, actor3]):
            messagebox.showerror(
                "Error",
                "Please fill all fields."
            )
            return

        sample = pd.DataFrame(
            {
                "Genre": [genre],
                "Director": [director],
                "Actor 1": [actor1],
                "Actor 2": [actor2],
                "Actor 3": [actor3]
            }
        )

        rating = model.predict(sample)[0]

        stars = "⭐" * max(1, round(rating))

        result_label.config(
            text=f"""
    {movie}

Predicted Rating

{stars}

{rating:.1f} / 10
""",
            fg="#00FF99"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI

root = tk.Tk()
root.title("Movie Rating Predictor")
root.geometry("750x700")
root.configure(bg="#0F172A")
root.resizable(False, False)

title = tk.Label(
    root,
    text="🎬 Movie Rating Predictor",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#0F172A"
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="Data Science Project using Random Forest Regression",
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
    text="Movie Name",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=0, column=0, sticky="w", pady=8)

movie_entry = tk.Entry(
    card,
    width=30,
    font=("Segoe UI", 11)
)
movie_entry.grid(row=0, column=1)

tk.Label(
    card,
    text="Genre",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=1, column=0, sticky="w", pady=8)

genre_entry = tk.Entry(
    card,
    width=30,
    font=("Segoe UI", 11)
)
genre_entry.grid(row=1, column=1)

tk.Label(
    card,
    text="Director",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=2, column=0, sticky="w", pady=8)

director_entry = tk.Entry(
    card,
    width=30,
    font=("Segoe UI", 11)
)
director_entry.grid(row=2, column=1)

tk.Label(
    card,
    text="Actor 1",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=3, column=0, sticky="w", pady=8)

actor1_entry = tk.Entry(
    card,
    width=30,
    font=("Segoe UI", 11)
)
actor1_entry.grid(row=3, column=1)

tk.Label(
    card,
    text="Actor 2",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=4, column=0, sticky="w", pady=8)

actor2_entry = tk.Entry(
    card,
    width=30,
    font=("Segoe UI", 11)
)
actor2_entry.grid(row=4, column=1)

tk.Label(
    card,
    text="Actor 3",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 11)
).grid(row=5, column=0, sticky="w", pady=8)

actor3_entry = tk.Entry(
    card,
    width=30,
    font=("Segoe UI", 11)
)
actor3_entry.grid(row=5, column=1)

predict_button = tk.Button(
    root,
    text="Predict Rating",
    command=predict_rating,
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
    text=f"Model R² Score: {score:.2f}",
    bg="#0F172A",
    fg="#93C5FD",
    font=("Segoe UI", 10)
)
accuracy_label.pack()

result_label = tk.Label(
    root,
    text="Enter movie details and click Predict Rating",
    bg="#1E293B",
    fg="white",
    font=("Segoe UI", 13),
    width=40,
    height=10,
    relief="ridge"
)
result_label.pack(pady=25)

root.mainloop()