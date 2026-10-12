import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.joblib")

app = FastAPI(title="Iris Neural Network Prediction")

class FlowerInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Species Prediction</title>
        <style>
            body { font-family: Arial; max-width: 500px; margin: 40px auto; padding: 15px; }
            input, button { width: 100%; padding: 10px; margin: 7px 0; box-sizing: border-box; }
            button { background: #2563eb; color: white; border: none; cursor: pointer; }
            #result { margin-top: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Iris Species Prediction</h1>
        <p>Enter flower measurements in centimeters.</p>
        <form id="form">
            <input name="sepal_length" type="number" step="any" value="5.1" placeholder="Sepal length" required>
            <input name="sepal_width" type="number" step="any" value="3.5" placeholder="Sepal width" required>
            <input name="petal_length" type="number" step="any" value="1.4" placeholder="Petal length" required>
            <input name="petal_width" type="number" step="any" value="0.2" placeholder="Petal width" required>
            <button type="submit">Predict Species</button>
        </form>
        <div id="result"></div>
        <script>
            document.getElementById("form").addEventListener("submit", async function(e) {
                e.preventDefault();
                const data = Object.fromEntries(new FormData(this));
                for (const key in data) data[key] = Number(data[key]);
                const result = document.getElementById("result");
                result.textContent = "Predicting...";
                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify(data)
                    });
                    const output = await response.json();
                    result.textContent = response.ok
                        ? "Predicted species: " + output.predicted_species
                        : "Error: " + output.detail;
                } catch (error) {
                    result.textContent = "Unable to connect to the prediction API.";
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(flower: FlowerInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]], columns=[
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ])

    prediction = model.predict(sample)[0]
    return {"predicted_species": str(prediction)}
