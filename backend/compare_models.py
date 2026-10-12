import os
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "IRIS.csv")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db").replace("\\", "/")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

df.rename(columns={
    "sepallengthcm": "sepal_length",
    "sepalwidthcm": "sepal_width",
    "petallengthcm": "petal_length",
    "petalwidthcm": "petal_width",
}, inplace=True)

features = [
    "sepal_length", "sepal_width",
    "petal_length", "petal_width"
]

X = df[features]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

mlflow.set_tracking_uri("sqlite:///" + DB_PATH)
mlflow.set_experiment("Iris_Model_Comparison")

for seed in [42, 21]:
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=seed
        ),
        "SVM": make_pipeline(
            StandardScaler(), SVC(random_state=seed)
        ),
        "QDA": QuadraticDiscriminantAnalysis(),
        "LDA": LinearDiscriminantAnalysis(),
        "Neural Network": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                max_iter=2000,
                random_state=seed,
            ),
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            macro_f1 = f1_score(
                y_test, predictions, average="macro", zero_division=0
            )

            mlflow.log_param("model", name)
            mlflow.log_param("random_seed", seed)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_macro_f1", macro_f1)

            print(
                f"Run: {name:16s} | Seed: {seed} | "
                f"Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}"
            )

print("\nModel comparison complete. Open the MLflow UI.")
