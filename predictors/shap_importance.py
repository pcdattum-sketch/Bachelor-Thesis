"""
shape_importance.py
========

Defines the function for SHAP analysis for the tree-based models and the linear model 
"""
import shap

def cv_shap_importance_tree(X, y, groups, model_class, best_params, feature_names):
    """
    SHAP importance for a tree-based model (RandomForestRegressor and GradientBoostingRegressor).

    """
    gkf = GroupKFold(n_splits=5)
    all_imp = []  # per-fold mean(|SHAP|) per feature, for the summary stats
    pooled_values = [] # raw out-of-fold SHAP values, one row per original sample
    pooled_data = [] # matching feature values, needed for beeswarm's colour axis

    for train_index, val_index in gkf.split(X, y, groups):
        model = model_class(**best_params, random_state=42)
        model.fit(X[train_index], y[train_index])

        explainer = shap.TreeExplainer(model)
        fold_explanation = explainer(X[val_index]) # Explanation object, for these unseen validation samples, did the feature spush performance up or down 

        all_imp.append(np.abs(fold_explanation.values).mean(axis=0))
        pooled_values.append(fold_explanation.values)
        pooled_data.append(X[val_index])

    arr = np.array(all_imp)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": arr.mean(axis=0),
        "importance_std": arr.std(axis=0),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    shap_explanation = shap.Explanation(
        values=np.concatenate(pooled_values, axis=0),
        data=np.concatenate(pooled_data, axis=0),
        feature_names=feature_names,
    )

    return importance_df, shap_explanation


def cv_shap_importance_lasso(X, y, groups, best_params, feature_names):
    """
    SHAP importance for Lasso

    """
    gkf = GroupKFold(n_splits=5)
    all_imp = [] # per-fold mean(|SHAP|) per feature, for the summary stats
    pooled_values = [] # raw out-of-fold SHAP values, one row per original sample
    pooled_data = [] # matching (scaled) feature values, needed for beeswarm's colour axis

    for train_index, val_index in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_index])
        X_val_scaled = scaler.transform(X[val_index])

        model = Lasso(**best_params, random_state=42)
        model.fit(X_train_scaled, y[train_index])

        explainer = shap.LinearExplainer(model, X_train_scaled)
        fold_explanation = explainer(X_val_scaled)  # for these unseen validation samples, did the feature push the (log-SOC) prediction up or down

        all_imp.append(np.abs(fold_explanation.values).mean(axis=0))
        pooled_values.append(fold_explanation.values)
        pooled_data.append(X_val_scaled)

    arr = np.array(all_imp)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": arr.mean(axis=0),
        "importance_std": arr.std(axis=0),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    shap_explanation = shap.Explanation(
        values=np.concatenate(pooled_values, axis=0),
        data=np.concatenate(pooled_data, axis=0),
        feature_names=feature_names,
    )

    return importance_df, shap_explanation
