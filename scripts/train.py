import os
import pickle
import time
from argparse import ArgumentParser, Namespace
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import names
from data_exploration import DATA_PATH, get_data, impute_missing_values


def one_hot_expand(df: pd.DataFrame) -> Tuple[np.ndarray, Sequence[str]]:
    enc = OneHotEncoder(sparse=False)
    multiclass_names = names.get(names.MULTICLASS_FEATURES)
    df_multiclass = df[multiclass_names]
    X_onehot = enc.fit_transform(df_multiclass)

    onehot_names = []
    for i, name in enumerate(multiclass_names):
        for cat in enc.categories_[i]:
            onehot_names.append(f"{name}_{int(cat)}")

    other_names = names.get(
        [k for k in names.FEATURES if k not in names.MULTICLASS_FEATURES]
    )
    X_other = df[other_names].values

    X = np.hstack([X_other, X_onehot])
    feature_names = other_names + onehot_names
    assert len(feature_names) == X.shape[1]

    return X, feature_names


def split_and_normalise(X: np.ndarray, y: np.ndarray) -> Tuple[(np.ndarray,) * 4]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=64
    )
    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, C=0.1) -> LogisticRegression:
    model = LogisticRegression(C=C, class_weight="balanced")
    print(model)
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Model trained in {(time.time() - start_time):.3f}s")
    return model


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default=DATA_PATH)
    parser.add_argument("-o", "--output-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = get_data(args.input)

    df = impute_missing_values(df)
    X, feature_names = one_hot_expand(df)

    X_train, X_test, y_train, y_test = split_and_normalise(X, df.diagnosed.values)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    model = train_model(X_train, y_train)

    # Save the model to disk
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    output_data_path = os.path.join(args.output_dir, "data.npz")
    np.savez(
        output_data_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    print(f"Data saved to {output_data_path}")

    output_path = os.path.join(args.output_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model, feature_names), f)
    print(f"Model saved to {output_path}")
