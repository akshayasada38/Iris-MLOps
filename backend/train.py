import json
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "IRIS.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

column_mapping = {
    "sepallengthcm": "sepal_length",
    "sepalwidthcm": "sepal_width",
    "petallengthcm": "petal_length",
    "petalwidthcm": "petal_width",
}
df.rename(columns=column_mapping, inplace=True)

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

if "species" not in df.columns or not all(c in df.columns for c in features):
    raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

X = df[features]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("neural_network", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=False,
        learning_rate_init=0.01,
        random_state=42,
    )),
])

mlflow.set_tracking_uri(
    "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db").replace("\\", "/")
)
mlflow.set_experiment("Iris-Neural-Network")

with mlflow.start_run():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    joblib.dump(model, os.path.join(MODEL_DIR, "iris_model.joblib"))

    with open(os.path.join(ARTIFACT_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)
    with open(os.path.join(ARTIFACT_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(matrix.tolist(), f, indent=4)

    mlflow.log_params({
        "model": "MLPClassifier",
        "hidden_layers": "16,8",
        "activation": "relu",
        "max_iter": 2000,
        "early_stopping": False,
        "test_size": 0.20,
    })

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_macro_f1", report["macro avg"]["f1-score"])

    mlflow.sklearn.log_model(
        model,
        name="iris_neural_network",
        skops_trusted_types=[
            "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
        ],
    )

    print("\nModel saved: models/iris_model.joblib")
    print("Evaluation reports saved in artifacts/")
    print("MLflow tracking database: mlflow.db")
