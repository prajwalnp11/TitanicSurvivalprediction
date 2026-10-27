
import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")

# Load the trained model pipeline
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def prepare_passenger(data):
    """Create features in the same format used during training."""
    family_size = data["SibSp"] + data["Parch"] + 1

    return pd.DataFrame([{
        "Pclass": data["Pclass"],
        "Sex": data["Sex"],
        "Age": data["Age"],
        "SibSp": data["SibSp"],
        "Parch": data["Parch"],
        "Fare": data["Fare"],
        "Embarked": data["Embarked"],
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1)
    }])


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = request.form.to_dict()

            passenger = {
                "Pclass": int(request.form["Pclass"]),
                "Sex": request.form["Sex"],
                "Age": float(request.form["Age"]),
                "SibSp": int(request.form["SibSp"]),
                "Parch": int(request.form["Parch"]),
                "Fare": float(request.form["Fare"]),
                "Embarked": request.form["Embarked"]
            }

            if passenger["Pclass"] not in (1, 2, 3):
                raise ValueError("Choose a valid passenger class.")

            if passenger["Sex"] not in ("male", "female"):
                raise ValueError("Choose a valid sex.")

            if passenger["Embarked"] not in ("S", "C", "Q"):
                raise ValueError("Choose a valid port.")

            if not 0 <= passenger["Age"] <= 100:
                raise ValueError("Age must be between 0 and 100.")

            if passenger["SibSp"] < 0 or passenger["Parch"] < 0:
                raise ValueError("Family counts cannot be negative.")

            if passenger["Fare"] < 0:
                raise ValueError("Fare cannot be negative.")

            features = prepare_passenger(passenger)

            prediction = int(model.predict(features)[0])
            probability = model.predict_proba(features)[0]
            classes = list(model.classes_)
            survival_probability = float(
                probability[classes.index(1)]
            )

            result = {
                "label": (
                    "Likely to survive"
                    if prediction == 1
                    else "Unlikely to survive"
                ),
                "survived": prediction,
                "probability": round(
                    survival_probability * 100, 1
                )
            }

        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc) or "Please check your passenger details."

        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check that the saved model "
                "matches the training pipeline."
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
