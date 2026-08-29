"""
scripts
=======

The main SOC-S2 thesis pipeline: data cleaning, the spatial train/test
split, feature-set construction, hyperparameter search, final test-set
evaluation, SHAP variable importance, and robustness checks

Uses the working feature file produced by the `features_extraction`
repo, and imports model implementations and metrics from the `predictor`
package 

Pipeline order (see run_pipeline.py, at this repo's root, for a runnable
end-to-end script):

    main_data_cleaning()    load + clean the working file, log-transform OC
    main_spatial_split()    grid-cell grouping, train/test split, outlier removal, imputation
    build_dataset_map()     Baseline/A/B/C/All feature arrays
    run_all()               (optional) PCA variance diagnostics per covariate subset
    run_all_and_save()      (optional) re-run RF/GB/Lasso hyperparameter search
    load_hyperparameters()  load the final hyperparameters (best_hyperparameters.json)
    main_evaluate_test()    deploy final models, evaluate on the held-out test set
    main_shap_analysis()    out-of-fold SHAP variable importance
    (robustness_checks functions) ablation studies (feature dropping, added indices, no-N)
"""


import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import loguniform, randint, uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from predictor import (
    compute_metrics,
    cv_shap_importance_lasso,
    cv_shap_importance_tree,
    deploy_gb,
    deploy_lasso,
    deploy_rf,
    search_gb_rs,
    search_lasso_hp,
    search_plsr_hp,
    search_rf_rs,
)

# ============================================================================
# config  -  Edit the values below to match your own machine
# ----------------------------------------------------------------------------

# Output of the `features_extraction` repo's last stage.
WORKING_FILE_CSV_PATH = "Working_files_full_covariates.csv"  # <-- replace with your own path

# Where to write results this repo produces (test-set metrics tables, SHAP
# importance tables, robustness-check tables). Defaults to a local `outputs/`
# folder so nothing is silently written outside the repo.
OUTPUT_DIR = "outputs"

# Hyperparameter config (see the hyperparameters section below). Defaults to
# the file bundled with this package (best_hyperparameters.json), set this
# to a different path to use a different tuning run's results.
HYPERPARAMETERS_JSON_PATH = None

# Random seed used throughout (spatial split, all model fits). This mirrors
# a scientific choice made consistently across the original notebooks (random_state=42 everywhere)
RANDOM_STATE = 42


# ============================================================================
# Load the working covariate file and prepare it for modelling
# ----------------------------------------------------------------------------

# Lab covariates known to sometimes carry a "<LOD" (below level of
# detection) string value instead of a number.
LAB_COLS = ["P", "N", "K", "pH_CaCl2", "EC"]

# Columns kept for modelling: identifiers, S2 spectral bands, non-lab environmental covariates, and lab covariates.
MODEL_COLS = [
    "POINTID", "lon", "lat", "OC",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "elevation", "slope", "aspect", "temp_mean", "precip_mean", 
    "peat", "clay_map_pct", "AWC_pct",
    "P", "N", "K", "pH_CaCl2", "EC",
]

def load_working_file(path=None):
    """
    Load the covariate CSV produced by the `features_extraction` repo
    """
    path = path or WORKING_FILE_CSV_PATH
    df_all = pd.read_csv(path)
    print(f"Loaded {df_all.shape[0]} rows, {df_all.shape[1]} columns from {path}")
    print(df_all.isnull().sum())
    return df_all


def report_lod_flags(df_all, lab_cols=LAB_COLS):
    """
    Print which lab columns contain a '<LOD' (below level of detection) flag.
    """
    flagged = {}
    for col in lab_cols:
        # Count entries containing '<' before any conversion, this isolates <LOD flags
        lod_mask = df_all[col].astype(str).str.contains("<", na=False)
        n_lod = lod_mask.sum()
        
        if n_lod > 0:
            print(f"{col}: {n_lod} values flagged as <LOD")
            flagged[col] = n_lod
    return flagged
    #should flag P (Phosphorous) of having 4 <LOD values


def fix_phosphorus_lod(df_all):
    """
    Replace P's '<LOD' string with LOD/2 (5 mg/kg, for an LOD of 10 mg/kg).

    This is the only LOD the original pipeline found. If your dataset flags LOD values in another column, add an equivalent fix for it before proceeding.
    """
    df_all = df_all.copy()
    df_all["P"] = df_all["P"].replace("<LOD", np.nan)
    df_all["P"] = pd.to_numeric(df_all["P"], errors="coerce")
    df_all.loc[df_all["P"].isna(), "P"] = 5.0
    return df_all


def select_model_columns(df_all, cols=MODEL_COLS):
    """
    Add the log-OC feature
    """
    df_full = df_all[cols].copy()
    df_full["OC (log)"] = np.log(df_full["OC"])
    return df_full


def print_skewness_summary(df_full):
    oc = df_full["OC"].values
    oc_log = df_full["OC (log)"].values
    print(f"Raw OC   — n={len(oc)}, skewness={df_full['OC'].skew():.2f}")
    print(f"Log(OC)  — n={len(oc_log)}, skewness={df_full['OC (log)'].skew():.2f}")
    return {"oc_skew": df_full["OC"].skew(), "log_oc_skew": df_full["OC (log)"].skew()}


def plot_oc_histograms(df_full):
    """
    Histogram + boxplot of raw and log-transformed OC.
    """
    oc = df_full["OC"].values
    oc_log = df_full["OC (log)"].values

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.hist(oc, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(oc.mean(), color="red", linestyle="--", label=f"Mean {oc.mean():.2f}")
    ax.axvline(np.median(oc), color="green", linestyle="--", label=f"Median {np.median(oc):.2f}")
    ax.set_xlabel("OC (g/kg)"); ax.set_ylabel("Count")
    ax.set_title(f"Raw OC — skewness = {df_full['OC'].skew():.2f}"); ax.legend()
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.hist(oc_log, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(oc_log.mean(), color="red", linestyle="--", label=f"Mean {oc_log.mean():.2f}")
    ax.axvline(np.median(oc_log), color="green", linestyle="--", label=f"Median {np.median(oc_log):.2f}")
    ax.set_xlabel("log OC"); ax.set_ylabel("Count")
    ax.set_title(f"Log OC — skewness = {df_full['OC (log)'].skew():.2f}"); ax.legend()
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.boxplot(oc_log, vert=True, patch_artist=True, widths=0.4,
               boxprops=dict(facecolor="steelblue", color="black"),
               medianprops=dict(color="green", linewidth=2),
               whiskerprops=dict(color="black"), capprops=dict(color="black"),
               flierprops=dict(marker="o", markerfacecolor="red", markersize=5, alpha=0.6))
    ax.set_ylabel("log OC")
    ax.set_title(f"Log OC — boxplot (n={len(oc_log)})")
    ax.set_xticks([])
    plt.tight_layout(); plt.show()


def plot_covariate_correlations(df_full, target="OC (log)"):
    """
    A correlation heatmap between features and target
    """
    bands = [target] + [c for c in MODEL_COLS]
    df_plot = df_full[bands].copy()

    corr = df_plot.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation matrix — {target} and features (n={len(df_plot)})", fontsize=12)
    plt.tight_layout(); plt.show()


def main_data_cleaning():
    df_all = load_working_file()
    report_lod_flags(df_all)
    df_all = fix_phosphorus_lod(df_all)
    df_full = select_model_columns(df_all)
    print_skewness_summary(df_full)
    return df_all, df_full


# ============================================================================
# Spatial grouping, train/test split, outlier removal, and imputation.
# ----------------------------------------------------------------------------

CELL_SIZE_M = 150_000

# Cells dropped for having too few points to be usable as a spatial group. This is SPECIFIC to the 310-point Germany dataset used in the thesis. It was picked by inspecting `gdf['cell_id'].value_counts()` after `assign_grid_cells`, not derived from a general rule. If you re-run this on a different point set, re-derive it the same way (or use `drop_sparse_cells` below with an explicit minimum-count threshold instead of this hardcoded list).

SPARSE_CELLS = ["150kmN20E30", "150kmN23E29", "150kmN18E27"] # this could differ in your run

IMPUTE_COLS = ["elevation", "slope", "aspect", "temp_mean", "precip_mean"] # these were the only feaetures with missing values in this pipeline


def assign_grid_cells(df_full, cell_size=CELL_SIZE_M):
    """
    Add `cell_id` (INSPIRE 150km grid cell) to a lon/lat dataframe.
    """
    gdf = gpd.GeoDataFrame(df_full, geometry=gpd.points_from_xy(df_full["lon"], df_full["lat"]), crs="EPSG:4326")
    # reproject to INSPIRE LAEA (equal-area, metres) which is based on the ETRS89 Lambert Azimuthal Equal Area with EPSG Code: EPSG:3035
    # essentially converting from coordinates into distance metrics with meters as units
    gdf = gdf.to_crs("EPSG:3035")

    gdf["cell_e"] = (gdf.geometry.x // cell_size).astype(int)
    gdf["cell_n"] = (gdf.geometry.y // cell_size).astype(int)
    gdf["cell_id"] = "150kmN" + gdf["cell_n"].astype(str) + "E" + gdf["cell_e"].astype(str)

    print(gdf["cell_id"].value_counts())
    print(f"Occupied cells: {gdf['cell_id'].nunique()}") 
    #For this study, this is 22 unique cells 
    return gdf


def drop_sparse_cells(gdf, cells_to_drop=SPARSE_CELLS):
    gdf = gdf.drop(gdf[gdf["cell_id"].isin(cells_to_drop)].index)
    print(f"Total sample size after dropping sparse cells: {len(gdf)}")
    #for this study, this was 304 samples from 310
    print(gdf["cell_id"].value_counts())
    return gdf


def split_train_test(gdf, n_splits=5, target_test_frac=0.20, random_state=42):
    """
    Stratified-by-log-OC-quintile, grouped-by-cell_id 80/20 split.

    `StratifiedGroupKFold` doesn't let you directly target a test fraction, so this searches its `n_splits` folds for whichever one's test fraction is closest to `target_test_frac`, and uses that fold.
    """
    gdf = gdf.copy()
    gdf["OC_bin"] = pd.qcut(gdf["OC (log)"], q=5, labels=False)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    #Testing what split with respect to both OC bin and group_id result in best 80/20 split
    best_fold, best_diff = None, 1.0
    for i, (train_index, test_index) in enumerate(sgkf.split(gdf, gdf["OC_bin"], groups=gdf["cell_id"])):
        test_frac = len(test_index) / len(gdf)
        diff = abs(test_frac - target_test_frac)
        print(f"Fold {i}: train={len(train_idx)} ({1 - test_frac:.1%}), test={len(test_idx)} ({test_frac:.1%})")
        if diff < best_diff:
            best_fold, best_diff = i, diff

    # then re-run split() and take that specific fold index
    for i, (train_index, test_index) in enumerate(sgkf.split(gdf, gdf["OC_bin"], groups=gdf["cell_id"])):
        if i == best_fold:
            train_val = gdf.iloc[train_index].copy()
            test = gdf.iloc[test_index].copy()
            break

    print(f"\nTraining + validation: {len(train_val)} points")
    print(f"Test set: {len(test)} points")
    print(f"Train log(OC) median={train_val['OC (log)'].median():.3f}, "
          f"skew={train_val['OC (log)'].skew():.3f}")
    print(f"Test log(OC) median={test['OC (log)'].median():.3f}, "
          f"skew={test['OC (log)'].skew():.3f}")

    return train_val, test


def remove_outliers(train_val, test, target_col="OC (log)"):
    """
    1.5xIQR outlier removal, with bounds computed on train_val ONLY.

    Applying train_val-only bounds to the test set avoids data leakage
    """
    q25, q75 = np.percentile(train_val[target_col], [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (1.5 * iqr)
    print(f"1.5xIQR bounds in {target_col}, from train_val only: [{lower_bound:.3f}, {upper_bound:.3f}]")

    train_val_mask = (train_val[target_col] < lower_bound) | (train_val[target_col] > upper_bound)
    test_mask = (test[target_col] < lower_bound) | (test[target_col] > upper_bound)
    print(f"Outliers flagged in train_val: {train_val_mask.sum()}")
    #for this study, 11 samples were flagged in train_val
    print(f"Outliers flagged in test: {test_mask.sum()}")
    #for this study, 3 samples were flagged in train_val

    train_val = train_val[~train_val_mask].copy()
    test = test[~test_mask].copy()
    print(f"train_val after outlier removal: {len(train_val)} samples") #leaving 233 samples
    print(f"test after outlier removal: {len(test)} samples") # leaving 57 samples
    return train_val, test


def impute_missing(train_val, test, impute_cols=IMPUTE_COLS):
    """
    Median-impute `impute_cols`, fitting the imputer on train_val ONLY.
    """
    print("Missing before imputation:")
    print(train_val[impute_cols].isnull().sum())
    print(test[impute_cols].isnull().sum())

    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    train_val = train_val.copy()
    test = test.copy()
    train_val[impute_cols] = imputer.fit_transform(train_val[impute_cols])
    test[impute_cols] = imputer.transform(test[impute_cols])

    #diagnostic test to see if it is imputed correctly
    print("Missing after imputation (should be all zero):")
    print(train_val[impute_cols].isnull().sum())
    print(test[impute_cols].isnull().sum())
    return train_val, test


def main_spatial_split(df_full):
    gdf = assign_grid_cells(df_full)
    gdf = drop_sparse_cells(gdf)
    train_val, test = split_train_test(gdf)
    train_val, test = remove_outliers(train_val, test)
    train_val, test = impute_missing(train_val, test)
    return train_val, test


# ============================================================================
# Feature set definitions and array-building helpers.
# ----------------------------------------------------------------------------

# --- Core feature sets ------------------------------------------------------
features_baseline = ["P", "N", "K", "pH_CaCl2", "EC"]
features_A = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
features_B = ["elevation", "slope", "aspect", "temp_mean", "precip_mean", "peat", "clay_map_pct", "AWC_pct"]
features_C = features_A + features_B
features_all = features_baseline + features_A + features_B

# Robustness check on features_all without N
features_baseline_noN = [c for c in features_baseline if c != "N"]
features_all_noN = features_baseline_noN + features_A + features_B


def build_arrays(df_split, feature_cols):
    """
    Extract (X, y_log, y_oc, groups) arrays for one feature set from a split.
    """
    cols_needed = feature_cols + ["OC (log)", "OC", "cell_id"]
    sub = df_split[cols_needed].dropna().copy()
    return (
        sub[feature_cols].values,
        sub["OC (log)"].values,
        sub["OC"].values,
        sub["cell_id"].values,
    )


def build_dataset_map(train_val):
    """
    Build the {feature_set_name: (X, y_log, y_oc, groups)} map used by every search/deploy call
    """
    dataset_map = {
        "Baseline": build_arrays(train_val, features_baseline),
        "A": build_arrays(train_val, features_A),
        "B": build_arrays(train_val, features_B),
        "C": build_arrays(train_val, features_C),
        "All": build_arrays(train_val, features_all),
    }
    for name, (X, y_log, y_oc, groups) in dataset_map.items():
        print(f" {name:8s}: X={X.shape}, y={y_log.shape}")
    return dataset_map

# ============================================================================
# pca_diagnostics.py  --  PCA variance diagnostics for each covariate subset.
# ----------------------------------------------------------------------------

SUBSETS = {
    "Feature Set A: Spectral Bands": features_A,
    "Baseline Feature Set: Lab-work covariates": features_baseline,
    "Feature Set B: Environmental Non Lab-work covariates": features_B,
    "Feature Set C: Spectral + Non Lab-work covariates": features_C,
    "Feature Set All": features_all,
}


def pca_variance_report(train_val, cols, title, variance_threshold=0.90):
    """
    run PCA, report/plot components needed for variance threshold
    """
    X = train_val[cols].values
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=None)
    pca.fit_transform(X_std)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_needed = int(np.searchsorted(cumulative, variance_threshold)) + 1

    print(f" PCA on {len(cols)} features - {title}")
    for i, (ev, cv) in enumerate(zip(explained, cumulative)):
        print(f" PC{i + 1}: {ev * 100:5.1f}% cumulative {cv * 100:5.1f}%")
    print(f"Components needed for {variance_threshold:.0%} variance: {n_needed}\n")

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    #Scree plot
    ax.bar(range(1, len(explained) + 1), explained * 100, color="steelblue", label="Variance explained per component")
    ax.plot(range(1, len(explained) + 1), cumulative * 100, "o-", color="darkorange", label="Cumulative")
    ax.axhline(variance_threshold * 100, color="red", linestyle="--", label=f"{variance_threshold:.0%} threshold")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"PCA {title} (train_val)")
    ax.set_xticks(range(1, len(explained) + 1))
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {"n_components_90pct": n_needed, "explained_variance_ratio": explained, "cumulative": cumulative}


def run_all(train_val, subsets=SUBSETS, variance_threshold=0.90):
    return {title: pca_variance_report(train_val, cols, title, variance_threshold) for title, cols in subsets.items()}


# ============================================================================
# Finding the final model, tune their hyperparameters used to deploy
# ----------------------------------------------------------------------------

DEFAULT_PATH = Path(__file__).parent / "best_hyperparameters.json"


def load_hyperparameters(path=None):
    """
    Load the hyperparameter config as a dict: {model: {feature_set: params}}.

    `model` is one of 'lasso', 'random_forest', 'gradient_boosting'.
    """
    path = path or HYPERPARAMETERS_JSON_PATH or DEFAULT_PATH
    with open(path) as f:
        data = json.load(f)
    data.pop("_comment", None)
    return data

def save_hyperparameters(data, path=None):
    """
    Write a hyperparameter config dict back out to JSON (e.g. after a new search_runner.py run)
    """
    path = path or HYPERPARAMETERS_JSON_PATH or DEFAULT_PATH
    data = {"_comment": "Final hyperparameters used to deploy each model per feature set.", **data}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote hyperparameters to {path}")

============================================================================
# search_runner.py  --  Run the hyperparameter searches (from the `predictor` package) for each
# ============================================================================
# initial range for the hyperparameter 
# for Random Forest:
test_grid = {
    'n_estimators':[50,100,300,500,1000],
    'max_features':[2,3,5], # extend to 10,18,23 going when changing feature set to fit with number of features
    'max_depth':[3,5,10,20],
    'min_samples_split': [2,5,10,20],
    'min_samples_leaf': [1,5,10,20]
}

#for Gradient Boosting 
test_grid = {
    'n_estimators':[50,100,300,1000],
    'learning_rate':[0.001, 0.01, 0.05,0.07, 0.1],
    'max_depth':[1,3,5,10,20],
    'min_samples_split': [2,5,10,20],
    'min_samples_leaf': [1,5,10,20],
    'subsample': [0.2,0.5, 0.7, 0.9],
    'max_features': [2, 3, 5] # extend to 10,18,23 going when changing feature set to fit with number of features
}


# Random Forest: final narrowed search spaces, one per feature set 
RF_SEARCH_SPACES = {
    "Baseline": dict(
        n_estimators=randint(100, 701), max_features=randint(3, 6), max_depth=randint(5, 21),
        min_samples_split=randint(2, 11), min_samples_leaf=randint(1, 6),
    ),
    "A": dict(
        n_estimators=randint(50, 1001), max_features=randint(1, 11), max_depth=randint(1, 16),
        min_samples_split=randint(2, 16), min_samples_leaf=randint(5, 21),
    ),
    "B": dict(
        n_estimators=randint(300, 501), max_features=randint(1, 6), max_depth=randint(1, 26),
        min_samples_split=randint(5, 31), min_samples_leaf=randint(1, 11),
    ),
    "C": dict(
        n_estimators=randint(500, 1201), max_features=randint(1, 19), max_depth=randint(10, 21),
        min_samples_split=randint(2, 21), min_samples_leaf=randint(1, 16),
    ),
    "All": dict(
        n_estimators=randint(10, 501), max_features=randint(10, 24), max_depth=randint(10, 26),
        min_samples_split=randint(2, 11), min_samples_leaf=randint(1, 21),
    ),
}
RF_N_ITER = 1500

# Gradient Boosting: final narrowed search spaces 
GB_SEARCH_SPACES = {
    "Baseline": dict(
        n_estimators=randint(500, 1501), learning_rate=loguniform(0.05, 0.1), max_depth=randint(1, 26),
        min_samples_split=randint(2, 11), min_samples_leaf=randint(1, 8), subsample=uniform(0.5, 0.4),
        max_features=randint(1, 6),
    ),
    "A": dict(
        n_estimators=randint(10, 301), learning_rate=loguniform(0.001, 0.05), max_depth=randint(1, 16),
        min_samples_split=randint(2, 36), min_samples_leaf=randint(1, 16), subsample=uniform(0.1, 0.6),
        max_features=randint(1, 11),
    ),
    "B": dict(
        n_estimators=randint(10, 501), learning_rate=loguniform(0.001, 0.02), max_depth=randint(1, 6),
        min_samples_split=randint(10, 26), min_samples_leaf=randint(1, 16), subsample=uniform(0.5, 0.4),
        max_features=randint(1, 9),
    ),
    "C": dict(
        n_estimators=randint(10, 501), learning_rate=loguniform(0.01, 0.05), max_depth=randint(1, 21),
        min_samples_split=randint(2, 16), min_samples_leaf=randint(1, 16), subsample=uniform(0.2, 0.7),
        max_features=randint(1, 11),
    ),
    "All": dict(
        n_estimators=randint(300, 1301), learning_rate=loguniform(0.01, 0.1), max_depth=randint(5, 21),
        min_samples_split=randint(2, 16), min_samples_leaf=randint(1, 11), subsample=uniform(0.5, 0.4),
        max_features=randint(1, 24),
    ),
}
GB_N_ITER = 200

# --- PLSR: n_components range, Baseline/A/B only (see module docstring) ----
PLSR_N_COMPONENT_RANGES = {
    "Baseline": range(1, 6),
    "A": range(1, 11),
    "B": range(1, 18),
}


def run_rf_search(dataset_map, feature_sets=None):
    """Run `search_rf_rs` for each feature set. RF is fit on the OC scale (not log)."""
    feature_sets = feature_sets or list(RF_SEARCH_SPACES.keys())
    best_params = {}
    for name in feature_sets:
        X, y_log, y_oc, groups = dataset_map[name]
        _, params = search_rf_rs(X, y_oc, groups, RF_N_ITER, RF_SEARCH_SPACES[name], feature_set=name)
        best_params[name] = params
    return best_params


def run_gb_search(dataset_map, feature_sets=None):
    """Run `search_gb_rs` for each feature set. GB is fit on the OC scale (not log)."""
    feature_sets = feature_sets or list(GB_SEARCH_SPACES.keys())
    best_params = {}
    for name in feature_sets:
        X, y_log, y_oc, groups = dataset_map[name]
        _, params = search_gb_rs(X, y_oc, groups, GB_N_ITER, GB_SEARCH_SPACES[name], feature_set=name)
        best_params[name] = params
    return best_params


def run_lasso_search(dataset_map, feature_sets=None):
    """Run `search_lasso_hp` for each feature set. Lasso is fit on log(OC)."""
    feature_sets = feature_sets or list(dataset_map.keys())
    best_alphas = {}
    for name in feature_sets:
        X, y_log, y_oc, groups = dataset_map[name]
        _, alpha = search_lasso_hp(X, y_log, groups, feature_set=name)
        best_alphas[name] = {"alpha": alpha}
    return best_alphas


def run_plsr_search(dataset_map, feature_sets=None):
    """Run `search_plsr_hp` for each feature set. PLSR is fit on log(OC).

    Only Baseline/A/B have a defined n_components range (see module
    docstring) -- pass `feature_sets` explicitly if you've added ranges
    for C/All yourself.
    """
    feature_sets = feature_sets or list(PLSR_N_COMPONENT_RANGES.keys())
    best_k = {}
    for name in feature_sets:
        X, y_log, y_oc, groups = dataset_map[name]
        _, k, _ = search_plsr_hp(X, y_log, groups, PLSR_N_COMPONENT_RANGES[name], feature_set=name)
        best_k[name] = k
    return best_k


def run_all_and_save(dataset_map, hyperparameters_path=None):
    """Run RF, GB, and Lasso searches for every feature set and save to best_hyperparameters.json.

    (PLSR's best_k isn't part of best_hyperparameters.json since
    `predictor.deploy_plsr` takes it as a positional int, not a params
    dict -- call `run_plsr_search` separately if you need it.)
    """
    data = {
        "lasso": run_lasso_search(dataset_map),
        "random_forest": run_rf_search(dataset_map),
        "gradient_boosting": run_gb_search(dataset_map),
    }
    save_hyperparameters(data, hyperparameters_path)
    return data


# ============================================================================
# Deploy the final Lasso, Random Forest, and Gradient Boosting models
# ----------------------------------------------------------------------------

FEATURE_SET_COLUMNS = {
    "Baseline": features_baseline,
    "A": features_A,
    "B": features_B,
    "C": features_C,
    "All": features_all,
}


def build_test_arrays(test, feature_set_columns=FEATURE_SET_COLUMNS):
    """Build (X, y_log, y_oc) test arrays for every feature set."""
    return {
        name: build_arrays(test, cols)[:3]  # (X, y_log, y_oc); drop groups, unused on test
        for name, cols in feature_set_columns.items()
    }


def evaluate_lasso(dataset_map, test_arrays, params):
    results, models = {}, {}
    for name, (X_train_val, y_train_val_log, y_oc_train_val, _groups) in dataset_map.items():
        alpha = params["lasso"][name]["alpha"]
        X_test, y_log_test, y_oc_test = test_arrays[name]

        model, scaler = deploy_lasso(X_train_val, y_train_val_log, alpha)  # refit on FULL train_val
        X_test_scaled = scaler.transform(X_test)                 # reuse train_val-fitted scaler
        pred_oc = np.exp(model.predict(X_test_scaled).ravel())  # back-transform from log(OC)

        m = compute_metrics(y_oc_test, pred_oc)
        results[name] = m
        models[name] = (model, scaler)
        print(f"[Lasso — {name:8s}] Test set: R^2={m['r2']:.3f}  RMSE={m['rmse']:.2f} g/kg  "
              f"RPD={m['rpd']:.2f}  RPIQ={m['rpiq']:.2f}")
    return pd.DataFrame(results).T, models


def evaluate_rf(dataset_map, test_arrays, params):
    results, models = {}, {}
    for name, (X_train_val, y_train_val_log, y_oc_train_val, _groups) in dataset_map.items():
        rf_params = params["random_forest"][name]
        X_test, y_log_test, y_oc_test = test_arrays[name]

        model, scaler = deploy_rf(X_train_val, y_oc_train_val, rf_params)  # RF fit directly on OC scale
        X_test_scaled = scaler.transform(X_test)
        pred_oc = model.predict(X_test_scaled)                 # no back-transform needed

        m = compute_metrics(y_oc_test, pred_oc)
        results[name] = m
        models[name] = (model, scaler)
        print(f"[RF — {name:8s}] Test set: R^2={m['r2']:.3f}  RMSE={m['rmse']:.2f} g/kg  "
              f"RPD={m['rpd']:.2f}  RPIQ={m['rpiq']:.2f}")
    return pd.DataFrame(results).T, models


def evaluate_gb(dataset_map, test_arrays, params):
    results, models = {}, {}
    for name, (X_train_val, y_train_val_log, y_oc_train_val, _groups) in dataset_map.items():
        gb_params = params["gradient_boosting"][name]
        X_test, y_log_test, y_oc_test = test_arrays[name]

        model, scaler = deploy_gb(X_train_val, y_oc_train_val, gb_params)  # GB fit directly on OC scale
        X_test_scaled = scaler.transform(X_test)
        pred_oc = model.predict(X_test_scaled)

        m = compute_metrics(y_oc_test, pred_oc)
        results[name] = m
        models[name] = (model, scaler)
        print(f"[GB — {name:8s}] Test set: R^2={m['r2']:.3f}  RMSE={m['rmse']:.2f} g/kg  "
              f"RPD={m['rpd']:.2f}  RPIQ={m['rpiq']:.2f}")
    return pd.DataFrame(results).T, models


def main_evaluate_test(dataset_map, test):
    params = load_hyperparameters()
    test_arrays = build_test_arrays(test)

    lasso_df, lasso_models = evaluate_lasso(dataset_map, test_arrays, params)
    rf_df, rf_models = evaluate_rf(dataset_map, test_arrays, params)
    gb_df, gb_models = evaluate_gb(dataset_map, test_arrays, params)

    return {
        "lasso": (lasso_df, lasso_models),
        "random_forest": (rf_df, rf_models),
        "gradient_boosting": (gb_df, gb_models),
    }

# ============================================================================
# SHAP variable importance for Lasso, Random Forest, and
# ----------------------------------------------------------------------------

def run_lasso_shap(dataset_map, params, plot=True):
    importance, explanations = {}, {}
    for name, feat_names in FEATURE_SET_COLUMNS.items():
        X, y_log, y_oc, groups = dataset_map[name]
        alpha_params = params["lasso"][name]

        imp_df, shap_exp = cv_shap_importance_lasso(X, y_log, groups, alpha_params, feat_names) #from shap_importance.py
        importance[name] = imp_df
        explanations[name] = shap_exp

        print(f"\n[Lasso — {name}] SHAP variable importance (mean |SHAP|, log(OC) scale):")
        print(imp_df.to_string(index=False))

        if plot:
            shap.plots.beeswarm(shap_exp, show=False)
            plt.title(f"Lasso — {name} — SHAP beeswarm (log-SOC scale)")
            plt.tight_layout()
            plt.show()
    return importance, explanations


def run_rf_shap(dataset_map, params, plot=True):
    importance, explanations = {}, {}
    for name, feat_names in FEATURE_SET_COLUMNS.items():
        X, y_log, y_oc, groups = dataset_map[name]
        rf_params = params["random_forest"][name]

        imp_df, shap_exp = cv_shap_importance_tree(X, y_oc, groups, RandomForestRegressor, rf_params, feat_names) #from shap_importance.py
        importance[name] = imp_df
        explanations[name] = shap_exp

        print(f"\n[RF — {name}] SHAP variable importance (SOC g/kg scale):")
        print(imp_df.to_string(index=False))

        if plot:
            shap.plots.beeswarm(shap_exp, show=False)
            plt.title(f"RF — {name} — SHAP beeswarm (train_val)")
            plt.tight_layout()
            plt.show()
    return importance, explanations


def run_gb_shap(dataset_map, params, plot=True):
    importance, explanations = {}, {}
    for name, feat_names in FEATURE_SET_COLUMNS.items():
        X, y_log, y_oc, groups = dataset_map[name]
        gb_params = params["gradient_boosting"][name]

        imp_df, shap_exp = cv_shap_importance_tree(X, y_oc, groups, GradientBoostingRegressor, gb_params, feat_names) #from shap_importance.py
        importance[name] = imp_df
        explanations[name] = shap_exp

        print(f"\n[GB — {name}] SHAP variable importance (SOC g/kg scale):")
        print(imp_df.to_string(index=False))

        if plot:
            shap.plots.beeswarm(shap_exp, show=False)
            plt.title(f"GB — {name} — SHAP beeswarm (train_val)")
            plt.tight_layout()
            plt.show()
    return importance, explanations


def main_shap_analysis(dataset_map, plot=True):
    params = load_hyperparameters()
    return {
        "lasso": run_lasso_shap(dataset_map, params, plot),
        "random_forest": run_rf_shap(dataset_map, params, plot),
        "gradient_boosting": run_gb_shap(dataset_map, params, plot),
    }

# ============================================================================
# Robustness checks against the main pipeline's results.
# ----------------------------------------------------------------------------

# Reported test-set results for the original ("with N") Feature Set All model

def check_all_without_nitrogen(train_val, test):
    """
    Feature Set All with N dropped: search RF/GB/Lasso, deploy, evaluate on test, compare to with-N results.
    """
    X_all_noN, y_log_all_noN, y_oc_all_noN, groups_all_noN = build_arrays(train_val, features_all_noN)

    _, best_params_all_noN = search_rf_rs(
        X_all_noN, y_oc_all_noN, groups_all_noN, 1500,
        dict(
            n_estimators=randint(10, 501), 
            max_features=randint(10, 23), 
            max_depth=randint(10, 26),
            min_samples_split=randint(2, 11),
            min_samples_leaf=randint(1, 21)),
        feature_set="All without N",
    )
    _, best_params_all_noN_gb = search_gb_rs(
        X_all_noN, y_oc_all_noN, groups_all_noN, 200,
        dict(
            n_estimators=randint(300, 1301), 
            learning_rate=loguniform(0.01, 0.1), 
            max_depth=randint(5, 21),
            min_samples_split=randint(2, 16), 
            min_samples_leaf=randint(1, 11), 
            subsample=uniform(0.5, 0.4),
            max_features=randint(1, 23)),
        feature_set="All without N",
    )
    _, best_alpha_all_noN = search_lasso_hp(X_all_noN, y_log_all_noN, groups_all_noN, feature_set="All without N")

    X_all_noN_test, y_log_all_noN_test, y_oc_all_noN_test, _ = build_arrays(test, features_all_noN)

    noN_configs = {
        ("L1", "All w/o N"): (X_all_noN, y_log_all_noN, best_alpha_all_noN, deploy_lasso, True),
        ("RF", "All w/o N"): (X_all_noN, y_oc_all_noN, best_params_all_noN, deploy_rf, False),
        ("GB", "All w/o N"): (X_all_noN, y_oc_all_noN, best_params_all_noN_gb, deploy_gb, False),
    }

    noN_results = {}
    for (model_label, fs_label), (X_train_val, y_train_val, params, deploy_fn, is_log) in noN_configs.items():
        model, scaler = deploy_fn(X_tv, y_tv, params)
        X_te_scaled = scaler.transform(X_all_noN_test)
        pred = model.predict(X_te_scaled)
        pred_oc = np.exp(pred) if is_log else pred
        m = compute_metrics(y_oc_all_noN_test, pred_oc)
        noN_results[(model_label, fs_label)] = m
        print(f"[{model_label} — {fs_label:15s}] Test set: R^2={m['r2']:.3f}  RMSE={m['rmse']:.2f} g/kg  "
              f"RPD={m['rpd']:.2f}  RPIQ={m['rpiq']:.2f}")

    noN_df = pd.DataFrame(noN_results).T
    noN_df.index.names = ["Model", "Feature Set"]

    with_n_df = pd.DataFrame(WITH_N_TEST_RESULTS).T
    with_n_df.index.names = ["Model", "Feature Set"]

    comparison = pd.concat([with_n_df, noN_df]).round(3).sort_index(level=0)
    return comparison
