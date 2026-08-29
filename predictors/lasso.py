"""
lasso.py
========

Defines the function for hyperparameter search and deployment for Lasso. Since it only has one parameter, a gridsearch like random forest and gradient boosting is not conducted
"""
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

def search_lasso_hp(X_train_val, y_train_val, groups, feature_set):
    """
    Two-stage Lasso alpha search on log-transformed OC.

    Stage 1: nested spatial GroupKFold CV (outer k=5, `LassoCV`'s internal
    k=3 inner search), an unbiased estimate of generalisation performance,
    used for reporting like RF and GB

    Stage 2: a separate `LassoCV` search over the FULL train_val set (its
    own k=5 CV, not nested) to be used only to pick the final alpha for
    deployment. 
    """
    outer_cv = GroupKFold(n_splits=5)
    inner_cv = GroupKFold(n_splits=3)
    fold_results = []

    # Outer loop -- Stage 1, performance estimate only
    for fold_i, (train_index, val_index) in enumerate(outer_cv.split(X_train_val, y_train_val, groups)):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        groups_train = groups[train_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # inner loop
        lasso_cv = LassoCV(
            cv=list(inner_cv.split(X_train_scaled, y_train, groups_train)),
            max_iter=10000,
            random_state=42,
            n_jobs=-1,
        )
        lasso_cv.fit(X_train_scaled, y_train)  # refits on the full outer-fold training set
        best_alpha_fold = lasso_cv.alpha_
        print(f"[fold {fold_i}] alpha grid: {lasso_cv.alphas_.min():.5g} to {lasso_cv.alphas_.max():.5g} "
              f"({len(lasso_cv.alphas_)} values), winner = {best_alpha_fold:.5g}")

        pred_oc = np.exp(lasso_cv.predict(X_val_scaled).ravel())
        true_oc = np.exp(y_val)
        m = compute_metrics(true_oc, pred_oc)
        m.update({"fold": fold_i, "best_alpha": best_alpha_fold})
        fold_results.append(m)

    cv_df = pd.DataFrame(fold_results)
    print(f"\n[L1 (Lasso) — {feature_set}] Outer CV results (Stage 1 — unbiased estimate):")
    print(cv_df[["fold", "r2", "rmse", "rpd", "rpiq"]].to_string(index=False))
    print(f"  Mean R^2   = {cv_df['r2'].mean():.3f} +/- {cv_df['r2'].std():.3f}")
    print(f"  Mean RMSE = {cv_df['rmse'].mean():.2f} +/- {cv_df['rmse'].std():.2f} g/kg")
    print(f"  Mean RPD  = {cv_df['rpd'].mean():.2f} +/- {cv_df['rpd'].std():.2f}")
    print(f"  Mean RPIQ = {cv_df['rpiq'].mean():.2f} +/- {cv_df['rpiq'].std():.2f}")
    print("\n  Per-fold best alpha (expected to differ across folds — not combined):")
    for row in cv_df.itertuples():
        print(f"Fold {row.fold}: alpha = {row.best_alpha:.5g}")

    # full train_val search, for deployment only
    scaler_final = StandardScaler()
    X_full_scaled = scaler_final.fit_transform(X_train_val)
    lasso_cv_final = LassoCV(
        cv=list(outer_cv.split(X_full_scaled, y_train_val, groups)),
        max_iter=10000,
        random_state=42,
        n_jobs=-1,
    )
    lasso_cv_final.fit(X_full_scaled, y_train_val)
    best_alpha = lasso_cv_final.alpha_
    print("\n Deployment search (Stage 2, full train_val, not a performance estimate):")
    print(f" alpha grid: {lasso_cv_final.alphas_.min():.5g} to {lasso_cv_final.alphas_.max():.5g} "
          f"({len(lasso_cv_final.alphas_)} values)")
    print(f" Best alpha (deployment) = {best_alpha:.5g}")

    return cv_df, best_alpha


def deploy_lasso(X_, y_, best_alpha):
    """
    Refit Lasso on the full given set (typically train_val) with a chosen alpha
    """
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_)
    final_model = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
    final_model.fit(X_scaled, y_)

    return final_model, scaler_final

