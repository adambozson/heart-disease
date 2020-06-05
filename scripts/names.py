from typing import Sequence

FEATURES = {
    3: "age",
    4: "sex",
    9: "cp",
    10: "trestbps",
    12: "chol",
    16: "fbs",
    19: "restecg",
    32: "thalach",
    38: "exang",
    40: "oldpeak",
    41: "slope",
    44: "ca",
    51: "thal",
}

BINARY_FEATURES = [4, 16, 38]
MULTICLASS_FEATURES = [9, 19, 41, 51]

CATEGORICAL_FEATURES = sorted(BINARY_FEATURES + MULTICLASS_FEATURES)
CONTINUOUS_FEATURES = [k for k in FEATURES if k not in CATEGORICAL_FEATURES]

TARGET = ["num"]


def get(keys: Sequence[int]) -> Sequence[str]:
    return list(map(FEATURES.__getitem__, keys))
