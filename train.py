import os
import sys
import traceback
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from skl2onnx import to_onnx
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset.csv")

    df = pd.read_csv(data_path)

    X = df[["size", "rooms", "year_built", "location", "condition"]]
    y = df["price"].astype(np.float32)

    numeric_features = ["size", "rooms", "year_built"]
    categorical_features = ["location", "condition"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_val, y_val)
    print(f"Validation R^2 score: {score:.4f}")

    # ✅ Explicit ONNX input schema (CRITICAL FIX)
    initial_types = [
        ("size", FloatTensorType([None, 1])),
        ("rooms", Int64TensorType([None, 1])),
        ("year_built", Int64TensorType([None, 1])),
        ("location", StringTensorType([None, 1])),
        ("condition", StringTensorType([None, 1])),
    ]

    onnx_model = to_onnx(pipeline, initial_types=initial_types)

    output_path = os.path.join(base_dir, "model.onnx")

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("✅ model.onnx exported successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n❌ TRAINING FAILED:\n")
        traceback.print_exc()
        sys.exit(1)