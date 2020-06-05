# main.py
#
# This script amalgamates the `if __name__ == "__main__"`
# actions of the other parts of the analysis chain.
#

import os
import pickle

import numpy as np

from data_exploration import (categorical_features_plot,
                              continuous_features_plot, get_data,
                              impute_missing_values, univariate_table)
from download import download_data
from evaluation import eval_metrics, extract_model_params, proba_hist, roc_plot
from train import one_hot_expand, split_and_normalise, train_model

OUTPUT_DIR = "outputs"

############
# Download #
############
print("Downloading data...")
path = download_data(".")
print(f"\nSaved to {path}")

###########################
# Preliminary exploration #
###########################
df = get_data(path)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Making continuous features plot...")
fig = continuous_features_plot(df)
fname = os.path.join(OUTPUT_DIR, "continuous_features.png")
fig.savefig(fname)
print(f"Saved to {fname}")

print("Making categorical features plot...")
fig = categorical_features_plot(df)
fname = os.path.join(OUTPUT_DIR, "categorical_features.png")
fig.savefig(fname)
print(f"Saved to {fname}\n")

df = impute_missing_values(df)

print("Univariate analysis:")
csv_fname = os.path.join(OUTPUT_DIR, "univariate_table.csv")
dff = univariate_table(df, save=csv_fname)
print(f"Saved to {csv_fname}\n")

##################
# Model training #
##################
X, feature_names = one_hot_expand(df)

X_train, X_test, y_train, y_test = split_and_normalise(X, df.diagnosed.values)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")

model = train_model(X_train, y_train)

output_data_path = os.path.join(OUTPUT_DIR, "data.npz")
np.savez(
    output_data_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
print(f"Data saved to {output_data_path}")

output_path = os.path.join(OUTPUT_DIR, "model.pkl")
with open(output_path, "wb") as f:
    pickle.dump((model, feature_names), f)
print(f"Model saved to {output_path}\n")

##############
# Evaluation #
##############
print("Making prediction distribution plot...")
fig = proba_hist(model, X_train)
fname = os.path.join(OUTPUT_DIR, "proba_hist.png")
fig.savefig(fname)
print(f"Saved to {fname}\n")

dec_train, dec_test = eval_metrics(model, X_train, X_test, y_train, y_test)

print("\nMaking ROC plot...")
fig = roc_plot(
    [dec_train, dec_test], [y_train, y_test], ["train", "test"], save_txt=OUTPUT_DIR
)
fname = os.path.join(OUTPUT_DIR, "roc.png")
fig.savefig(fname)
print(f"Saved to {fname}")

print("\nRanked feature coefficients:")
fname = os.path.join(OUTPUT_DIR, "coeffs.csv")
extract_model_params(model, feature_names, save=fname)
print(f"Saved to {fname}")
