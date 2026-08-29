"""
metrics_plotting
=========

Model implementations and evaluation metrics for the SOC-S2 thesis pipeline.

This file defines Functions for computing metrics, and plotting used in hyperparameter tuning 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# metrics - Regression evaluation metrics for SOC prediction
# ----------------------------------------------------------------------------

def compute_metrics(true_oc, pred_oc):
    """
    Compute R^2, RMSE, RPD, and RPIQ for predictions on the original OC scale.
    Returns dictionary with keys 'r2', 'rmse', 'rpd', 'rpiq'.
    """
    rmse = np.sqrt(mean_squared_error(true_oc, pred_oc))
    r2 = r2_score(true_oc, pred_oc)
    rpd = np.std(true_oc, ddof=1) / rmse
    # ddof=1 corrects for bias (toward lower values) in the variance estimate
    # introduced when the sample mean is used in place of the true population mean.
    iqr = np.percentile(true_oc, 75) - np.percentile(true_oc, 25)
    rpiq = iqr / rmse

    return {"r2": r2, "rmse": rmse, "rpd": rpd, "rpiq": rpiq}

# ============================================================================
# graphs - Diagnostic plots for hyperparameter search
# ----------------------------------------------------------------------------

def plot_gs_trends(gs_results, param_names, feature_set_name, algo_name):
    gs_df = pd.DataFrame(gs_results)
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, param_names): #loops each hyperparameters through their values 
        col = f"param_{param}"
        grouped = gs_df.groupby(col)["mean_test_score"].mean().reset_index() #finds the mean of each parameter performance 
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
        grouped = grouped.sort_values(col)
        ax.plot(grouped[col], -grouped["mean_test_score"], marker="o", color="steelblue")
        ax.set_xlabel(param)
        ax.set_ylabel("CV RMSE (g/kg)")
        ax.set_title(param)
    fig.suptitle(f"Stage 1 trend plots — {algo_name} — Feature Set {feature_set_name}")
    plt.tight_layout()
    plt.show()


def plot_parallel_coords(rs_results, param_names, feature_set_name, algo_name):
    rs_df = pd.DataFrame(rs_results)
    param_cols = [f"param_{p}" for p in param_names]
    for c in param_cols:
        rs_df[c] = pd.to_numeric(rs_df[c], errors="coerce")

    # scikit-learn stores RMSE as negative (neg_root_mean_squared_error); flip sign for display.
    rs_df["cv_rmse_log"] = -rs_df["mean_test_score"]

    fig, ax = plt.subplots(figsize=(max(8, 4 * len(param_names)), 4))
    norm = plt.Normalize(rs_df["cv_rmse_log"].min(), rs_df["cv_rmse_log"].max())
    cmap = cm.RdYlGn_r #colormap

    for _, row in rs_df.iterrows():
        vals = [row[c] for c in param_cols]
        norms_v = []
        for c, v in zip(param_cols, vals):
            mn, mx = rs_df[c].min(), rs_df[c].max()
            norms_v.append((v - mn) / (mx - mn + 1e-9))
        ax.plot(range(len(param_names)), norms_v, color=cmap(norm(row["cv_rmse_log"])), alpha=0.35, lw=0.8)

    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=15)
    ax.set_ylabel("Normalised parameter value")
    ax.set_title(f"Parallel coordinates — {algo_name} — Feature Set {feature_set_name}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="CV RMSE (log OC)")
    plt.tight_layout()
    plt.show()


def plot_hp_importance(rs_results, param_names, feature_set_name, algo_name):
    rs_df = pd.DataFrame(rs_results)
    param_cols = [f"param_{p}" for p in param_names]
    for c in param_cols:
        rs_df[c] = pd.to_numeric(rs_df[c], errors="coerce")
    rs_df["cv_rmse_log"] = -rs_df["mean_test_score"]

    importances = []
    for c, p in zip(param_cols, param_names):
        valid = rs_df[[c, "cv_rmse_log"]].dropna()
        corr = valid.corr().iloc[0, 1]
        importances.append((p, abs(corr)))
    importances.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*importances)

    fig, ax = plt.subplots(figsize=(6, max(3, len(param_names))))
    ax.barh(names, vals, color="steelblue")
    ax.set_xlabel("|Pearson r| with CV RMSE (log OC)")
    ax.set_title(f"Hyperparameter importance — {algo_name} — Feature Set {feature_set_name}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
