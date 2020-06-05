import os
import pickle
from argparse import ArgumentParser, Namespace
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve


def proba_hist(model: BaseEstimator, X_train: np.ndarray) -> plt.Figure:
    probs = model.predict_proba(X_train)[:, 1]  # Prob(+ve diagnosis)
    fig, ax = plt.subplots()
    ax.hist(probs)
    ax.set_xlabel("Predicted probability of heart disease")
    ax.set_ylabel("Count")
    return fig


def eval_metrics(
    model: BaseEstimator, *data: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = data

    acc_train = model.score(X_train, y_train)
    decision_func_train = model.decision_function(X_train)
    auc_train = roc_auc_score(y_train, decision_func_train)
    fpr_train, tpr_train, _ = roc_curve(y_train, decision_func_train)
    print(f"Training set: AUC={auc_train}, acc={acc_train}")

    acc_test = model.score(X_test, y_test)
    decision_func_test = model.decision_function(X_test)
    auc_test = roc_auc_score(y_test, decision_func_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, decision_func_test)
    print(f"Testing set:  AUC={auc_test}, acc={acc_test}")

    return decision_func_train, decision_func_test


def roc_plot(
    decisions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    labels: Optional[str] = None,
    save_txt: Optional[str] = None,
):
    assert len(decisions) == len(targets)
    if save_txt != None:
        assert labels != None

    fig, ax = plt.subplots()
    for i, (dec, tar) in enumerate(zip(decisions, targets)):
        fpr, tpr, _ = roc_curve(tar, dec)
        l = None
        if labels != None:
            l = labels[i]
            if save_txt != None:
                arr = np.vstack([fpr, tpr]).T
                fname = os.path.join(save_txt, f"{l}_roc.txt")
                np.savetxt(fname, arr)
        ax.plot(fpr, tpr, label=l)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    return fig


def extract_model_params(
    model: BaseEstimator, names: Sequence[str], save: Optional[str] = None
) -> None:
    assert len(names) == model.coef_.shape[1]
    coeffs = {names[i]: model.coef_[0, i] for i in range(len(names))}
    coeffs["(constant)"] = model.intercept_[0]
    df = pd.DataFrame()
    df["coef"] = pd.Series(coeffs)
    df["abs_coef"] = abs(df.coef)
    df = df.sort_values("abs_coef", ascending=False).coef
    print(df)
    if save != None:
        df.to_csv(save, index_label="feature")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", default="outputs/data.npz")
    parser.add_argument("-m", "---model", default="outputs/model.pkl")
    parser.add_argument("-o", "--output-dir", default="outputs")
    return parser.parse_args()


def load_data(fname: str) -> Tuple[(np.ndarray,) * 4]:
    npz = np.load(fname)
    return npz["X_train"], npz["X_test"], npz["y_train"], npz["y_test"]


def load_model(fname: str) -> Tuple[BaseEstimator, Sequence[str]]:
    with open(fname, "rb") as f:
        model, feature_names = pickle.load(f)
    return model, feature_names


if __name__ == "__main__":
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data(args.data)
    model, feature_names = load_model(args.model)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print("Making prediction distribution plot...")
    fig = proba_hist(model, X_train)
    fname = os.path.join(args.output_dir, "proba_hist.png")
    fig.savefig(fname)
    print(f"Saved to {fname}")

    dec_train, dec_test = eval_metrics(model, X_train, X_test, y_train, y_test)

    print("Making ROC plot...")
    fig = roc_plot(
        [dec_train, dec_test],
        [y_train, y_test],
        ["train", "test"],
        save_txt=args.output_dir,
    )
    fname = os.path.join(args.output_dir, "roc.png")
    fig.savefig(fname)
    print(f"Saved to {fname}")

    print("\nRanked feature coefficients:")
    fname = os.path.join(args.output_dir, "coeffs.csv")
    extract_model_params(model, feature_names, save=fname)
    print(f"Saved to {fname}")
