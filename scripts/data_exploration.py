import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import chi2, f_classif

import names

DATA_PATH = "processed.cleveland.data"


def get_data(path: str = DATA_PATH, print_info: bool = True) -> pd.DataFrame:
    columns = list(names.FEATURES.values()) + names.TARGET
    df = pd.read_csv(DATA_PATH, names=columns, na_values="?")

    # Make the target column
    df["diagnosed"] = df.num > 0

    if print_info:
        print(f"Loaded {len(df)} entries from {path}\n")
        df.info()
        print("\n", df.head(10), "\n")

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna({"ca": df.ca.mean(), "thal": df.thal.mode()[0]})


def continuous_features_plot(df: pd.DataFrame) -> plt.Figure:
    g = sns.pairplot(df, vars=names.get(names.CONTINUOUS_FEATURES), hue="diagnosed")
    return g.fig


def categorical_features_plot(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharey=True)
    cat_vars = names.get(names.CATEGORICAL_FEATURES)
    for i, ax in enumerate(axes.flatten()):
        if i == 7:
            ax.axis("off")  # Don't display an empty axis
            break
        sns.countplot(x=cat_vars[i], data=df, hue="diagnosed", ax=ax)
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    return fig


def univariate_table(df: pd.DataFrame, save=None, print_table=True) -> pd.DataFrame:
    cont_vars = names.get(names.CONTINUOUS_FEATURES)
    f_score, p_val = f_classif(df[cont_vars], df.diagnosed)
    dff = pd.DataFrame(
        np.asarray([f_score, p_val]).T, index=cont_vars, columns=["f_score", "p_val"]
    )

    cat_vars = names.get(names.CATEGORICAL_FEATURES)
    chi_sq, p_val2 = chi2(df[cat_vars], df.diagnosed)
    dff = dff.append(
        pd.DataFrame(
            np.asarray([chi_sq, p_val2]).T, index=cat_vars, columns=["chi_sq", "p_val"]
        )
    )
    dff = dff[["chi_sq", "f_score", "p_val"]].sort_values("p_val")

    if print_table:
        print(dff)

    if save != None:
        dff.to_csv(save, na_rep="", index_label="feature")

    return dff


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default=DATA_PATH)
    parser.add_argument("-o", "--output-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = get_data(args.input)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print("Making continuous features plot...")
    fig = continuous_features_plot(df)
    fname = os.path.join(args.output_dir, "continuous_features.png")
    fig.savefig(fname)
    print(f"Saved to {fname}")

    print("Making categorical features plot...")
    fig = categorical_features_plot(df)
    fname = os.path.join(args.output_dir, "categorical_features.png")
    fig.savefig(fname)
    print(f"Saved to {fname}\n")

    df = impute_missing_values(df)

    print("Univariate analysis")
    csv_fname = os.path.join(args.output_dir, "univariate_table.csv")
    dff = univariate_table(df, save=csv_fname)
    print(f"Saved to {csv_fname}")
