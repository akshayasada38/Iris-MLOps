import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "backend", "train.py")

print("========================================")
print(" Iris Neural Network - MLOps Pipeline")
print("========================================")

print("\nStep 1: Starting model training...")

subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd=BASE_DIR,
    check=True
)

print("\nStep 2: Model training completed.")
print("Step 3: Model and evaluation artifacts generated.")
print("\nMLOps pipeline completed successfully!")
