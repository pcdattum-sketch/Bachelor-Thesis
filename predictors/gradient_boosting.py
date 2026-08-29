"""
gradient_boosting.py
========

Defines the function for hyperparameter search and deployment for Gradient Boosting 
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

def search_gb_grid(X_train_val, y_train_val, groups, test_grid, feature_set):
    """Stage 1: coarse GridSearchCV over outer spatial GroupKFold CV.

    Diagnostic only, does not refit a final model. Use this to inspect trend plots before choosing a narrowed range for `search_rf_rs`'s RandomizedSearchCV.
    """
    outer_cv = GroupKFold(n_splits=5)
    print(f"\n[GBoosting — {feature_set}] Stage 1: Coarse GridSearch...")

    scaler_gs = StandardScaler()
    X_gs = scaler_gs.fit_transform(X_train_val)

    gs = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid=test_grid,
        cv=list(outer_cv.split(X_gs, y_train_val, groups)),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=False,  # diagnostic only, no model is kept
    )

    gs.fit(X_gs, y_train_val)
    plot_gs_trends(gs.cv_results_, list(test_grid.keys()), feature_set, "GBoosting")

    cv_results_df = pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False)
    print(f"  Stage 1 best combination: {cv_results_df.iloc[0]['params']}")

    return cv_results_df


def search_gb_rs(X_train_val, y_train_val, groups, N_iter, test_param, feature_set):
    """
    Stage 2: nested spatial GroupKFold CV with RandomizedSearchCV.

    Run `search_gb_grid` first and inspect its trend plots to choose a narrowed `test_param` range before calling this.

    The outer nested loop (5 outer folds x inner RandomizedSearchCV) is used ONLY to estimate generalisation performance (an honest, unbiased R^2/RMSE). It is NOT used to choose the hyperparameters that get deployed.

    A separate RandomizedSearchCV pass is run over the FULL train_val set specifically to select the final hyperparameters used downstream in `deploy_gb`. 
    """
    outer_cv = GroupKFold(n_splits=5)
    inner_cv = GroupKFold(n_splits=3)

    fold_results = []
    for fold_i, (train_index, val_index) in enumerate(outer_cv.split(X_train_val, y_train_val, groups)):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        groups_train = groups[train_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        rs = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_distributions=test_param,
            n_iter=N_iter,
            cv=list(inner_cv.split(X_train_scaled, y_train, groups_train)),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42,
            refit=True,
        )
        rs.fit(X_train_scaled, y_train)
        print(f"[sanity check] fold {fold_i} candidates evaluated: {len(rs.cv_results_['params'])}")
        pred_oc = rs.best_estimator_.predict(X_val_scaled)
        true_oc = y_val
        m = compute_metrics(true_oc, pred_oc)
        m.update({"fold": fold_i, "best_params": rs.best_params_})
        fold_results.append(m)

    cv_df = pd.DataFrame(fold_results)
    print(f"\n[GBoosting — {feature_set}] Outer CV results:")
    print(cv_df[["fold", "r2", "rmse", "rpd", "rpiq"]].to_string(index=False))
    print(f"  Mean R^2   = {cv_df['r2'].mean():.3f} +/- {cv_df['r2'].std():.3f}")
    print(f"  Mean RMSE = {cv_df['rmse'].mean():.2f} +/- {cv_df['rmse'].std():.2f} g/kg")
    print(f"  Mean RPD  = {cv_df['rpd'].mean():.2f} +/- {cv_df['rpd'].std():.2f}")
    print(f"  Mean RPIQ = {cv_df['rpiq'].mean():.2f} +/- {cv_df['rpiq'].std():.2f}")

    print("\n  Per-fold best params (stability check across outer folds):")
    for row in cv_df.itertuples():
        print(f"    Fold {row.fold}: {row.best_params}")

    # full train_val search, for deployment only
    scaler_final = StandardScaler()
    X_full_scaled = scaler_final.fit_transform(X_train_val)

    rs_final = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=test_param,
        n_iter=N_iter,
        cv=list(outer_cv.split(X_full_scaled, y_train_val, groups)),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    rs_final.fit(X_full_scaled, y_train_val)
    print(f"[sanity check] candidates evaluated: {len(rs_final.cv_results_['params'])}")
    best_params = rs_final.best_params_
    print(f"\n Best params (full train_val search, used for deployment): {best_params}")

    plot_parallel_coords(rs_final.cv_results_, list(test_param.keys()), feature_set, "GBoosting")
    plot_hp_importance(rs_final.cv_results_, list(test_param.keys()), feature_set, "GBoosting")

    return cv_df, best_params


def deploy_gb(X_, y_, param_dist):
    """
    Refit Gradient Boosting on the full given set (typically train_val) with chosen params.
    """
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_)
    final_model = GradientBoostingRegressor(**param_dist, random_state=42)
    final_model.fit(X_scaled, y_)

    return final_model, scaler_final
