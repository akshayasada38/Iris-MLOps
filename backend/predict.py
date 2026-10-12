import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.joblib")

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run python run_pipeline.py first."
        )

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    prediction = model.predict(sample)[0]
    return prediction

if __name__ == "__main__":
    result = predict_species(5.1, 3.5, 1.4, 0.2)
    print("Input flower measurements: 5.1, 3.5, 1.4, 0.2")
    print("Predicted Iris species:", result)
