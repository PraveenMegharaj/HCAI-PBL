"""Explainability primitives — implemented by hand, no external XAI library.

- Counterfactuals: local sampling + MAD-weighted L1 ranking (Task 4)
- PDP: iterate over feature values, average predict_proba (Task 5)
- ALE: quantile bins, mean local effect within bin, cumsum + centre (Task 5)
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task 4 — Counterfactual generation
# ---------------------------------------------------------------------------

def generate_counterfactuals(x, target_class, model, X, N=10000, k=5,
                             seed=None):
    """Return (top_k_counterfactuals_df, original_point_df) or None.

    Feature types are inferred from the training data:
      * 2 unique values → binary, sample uniformly from the two values
      * ≤10 unique values → categorical, sample uniformly from observed
      * otherwise → continuous, add Gaussian noise scaled to feature std
    """
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=float)
    X_arr = X.values.astype(float)

    mad = np.median(np.abs(X_arr - np.median(X_arr, axis=0)), axis=0)
    mad[mad == 0] = 1.0

    n_features = X_arr.shape[1]
    unique_counts = [len(np.unique(X_arr[:, i])) for i in range(n_features)]
    uniques       = [np.unique(X_arr[:, i])      for i in range(n_features)]
    stds          = X_arr.std(axis=0)

    samples = np.empty((N, n_features), dtype=float)
    for i in range(n_features):
        if unique_counts[i] <= 10:  # binary or categorical
            samples[:, i] = rng.choice(uniques[i], size=N)
        else:                       # continuous — Gaussian around x[i]
            samples[:, i] = x[i] + rng.normal(0, stds[i] * 0.5, size=N)

    preds = model.predict(samples)
    matching = samples[preds == target_class]
    if len(matching) == 0:
        return None

    distances = np.sum(np.abs(matching - x) / mad, axis=1)
    order = np.argsort(distances)[:k]
    cf_df = pd.DataFrame(matching[order], columns=X.columns)
    cf_df['MAD_L1_distance'] = distances[order].round(4)
    original_df = pd.DataFrame([x], columns=X.columns)
    return cf_df, original_df


# ---------------------------------------------------------------------------
# Task 5 — PDP + ALE
# ---------------------------------------------------------------------------

def compute_pdp(model, X, feature_col, feature_vals, n_classes=3):
    """Partial Dependence: for each value v of `feature_col`, set that
    column to v across the whole dataset and average predict_proba.

    Returns shape (n_classes, len(feature_vals)).
    """
    X_arr = X.values.astype(float)
    feat_idx = list(X.columns).index(feature_col)
    pdp_values = np.zeros((n_classes, len(feature_vals)))
    for j, val in enumerate(feature_vals):
        X_mod = X_arr.copy()
        X_mod[:, feat_idx] = val
        pdp_values[:, j] = model.predict_proba(X_mod).mean(axis=0)
    return pdp_values


def compute_ale(model, X, feature_col, n_bins=20, n_classes=3):
    """Accumulated Local Effect — quantile-binned discrete derivative,
    cumulative sum across bins, mean-centered per class.

    For classifiers with a smooth `predict_proba` the ALE is more
    reliable than PDP when the feature is correlated with others,
    because it only uses local perturbations within a bin.

    Returns (bin_centers, ale_values) with ale shape (n_classes, n_bins).
    """
    X_arr = X.values.astype(float)
    feat_idx = list(X.columns).index(feature_col)
    feat_vals = X_arr[:, feat_idx]

    quantiles = np.percentile(feat_vals, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    n_actual_bins = len(quantiles) - 1

    ale_values  = np.zeros((n_classes, n_actual_bins))
    bin_centers = np.zeros(n_actual_bins)

    for b in range(n_actual_bins):
        lower, upper = quantiles[b], quantiles[b + 1]
        bin_centers[b] = (lower + upper) / 2

        mask = (feat_vals >= lower) & (feat_vals <= upper)
        if mask.sum() == 0:
            continue

        X_bin = X_arr[mask].copy()
        X_lower = X_bin.copy(); X_lower[:, feat_idx] = lower
        X_upper = X_bin.copy(); X_upper[:, feat_idx] = upper

        local_effect = (model.predict_proba(X_upper)
                        - model.predict_proba(X_lower)).mean(axis=0)
        ale_values[:, b] = local_effect

    ale_values = np.cumsum(ale_values, axis=1)
    ale_values -= ale_values.mean(axis=1, keepdims=True)
    return bin_centers, ale_values
