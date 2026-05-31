import os
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.conf import settings

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from palmerpenguins import load_penguins


def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return settings.MEDIA_URL + filename


def load_and_prepare():
    """Load and clean the Palmer Penguins dataset."""
    df = load_penguins()
    df = df.dropna()

    # Encode categorical features
    df['island'] = LabelEncoder().fit_transform(df['island'])
    df['sex']    = LabelEncoder().fit_transform(df['sex'])

    feature_cols = ['island', 'bill_length_mm', 'bill_depth_mm',
                    'flipper_length_mm', 'body_mass_g', 'sex', 'year']
    target_col = 'species'

    X = df[feature_cols]
    y = LabelEncoder().fit_transform(df[target_col])
    class_names = sorted(df[target_col].unique().tolist())

    return X, y, feature_cols, class_names

def generate_counterfactuals(x, target_class, model, X, N=10000, k=5):
    """
    x            : original data point (pandas Series)
    target_class : desired predicted class (int)
    model        : trained sklearn model
    X            : full feature dataframe (for MAD computation)
    N            : number of random samples
    k            : number of counterfactuals to return
    """
    x = np.array(x, dtype=float)
    X_arr = X.values.astype(float)

    # Compute MAD for each feature (avoid division by zero)
    mad = np.median(np.abs(X_arr - np.median(X_arr, axis=0)), axis=0)
    mad[mad == 0] = 1.0

    # Identify feature types
    n_features = X_arr.shape[1]
    unique_counts = [len(np.unique(X_arr[:, i])) for i in range(n_features)]

    # Sample N points around x
    samples = []
    for _ in range(N):
        sample = x.copy()
        for i in range(n_features):
            if unique_counts[i] == 2:
                # Binary feature: random flip
                sample[i] = np.random.choice(np.unique(X_arr[:, i]))
            elif unique_counts[i] <= 10:
                # Categorical feature: random choice
                sample[i] = np.random.choice(np.unique(X_arr[:, i]))
            else:
                # Continuous feature: Gaussian noise
                std = np.std(X_arr[:, i])
                sample[i] = x[i] + np.random.normal(0, std * 0.5)
        samples.append(sample)

    samples = np.array(samples)

    # Filter: keep only samples predicted as target class
    preds = model.predict(samples)
    matching = samples[preds == target_class]

    if len(matching) == 0:
        return None

    # Rank by MAD-weighted L1 distance
    distances = np.sum(np.abs(matching - x) / mad, axis=1)
    ranked_idx = np.argsort(distances)[:k]
    counterfactuals = matching[ranked_idx]
    cf_distances = distances[ranked_idx]

    # Build result dataframes
    cf_df = pd.DataFrame(counterfactuals, columns=X.columns)
    cf_df['MAD_L1_distance'] = cf_distances.round(4)
    original_df = pd.DataFrame([x], columns=X.columns)

    return cf_df, original_df


def compute_pdp(model, X, feature_col, feature_vals, n_classes=3):
    """
    Compute PDP for a single feature across all classes.
    Returns array of shape (n_classes, len(feature_vals))
    """
    X_arr = X.values.astype(float)
    feat_idx = list(X.columns).index(feature_col)
    pdp_values = np.zeros((n_classes, len(feature_vals)))

    for j, val in enumerate(feature_vals):
        X_mod = X_arr.copy()
        X_mod[:, feat_idx] = val
        # Get predicted probabilities for each class
        probs = model.predict_proba(X_mod)  # shape (n_samples, n_classes)
        pdp_values[:, j] = probs.mean(axis=0)

    return pdp_values


def compute_ale(model, X, feature_col, n_bins=20, n_classes=3):
    """
    Compute ALE for a single feature across all classes.
    Returns bin_centers and ale_values of shape (n_classes, n_bins)
    """
    X_arr = X.values.astype(float)
    feat_idx = list(X.columns).index(feature_col)
    feat_vals = X_arr[:, feat_idx]

    # Define bins using quantiles
    quantiles = np.percentile(feat_vals,
                              np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    n_actual_bins = len(quantiles) - 1

    ale_values = np.zeros((n_classes, n_actual_bins))
    bin_centers = np.zeros(n_actual_bins)

    for b in range(n_actual_bins):
        lower = quantiles[b]
        upper = quantiles[b + 1]
        bin_centers[b] = (lower + upper) / 2

        # Select points in this bin
        mask = (feat_vals >= lower) & (feat_vals <= upper)
        if mask.sum() == 0:
            continue

        X_bin = X_arr[mask].copy()

        # Predict at lower bound
        X_lower = X_bin.copy()
        X_lower[:, feat_idx] = lower
        probs_lower = model.predict_proba(X_lower)

        # Predict at upper bound
        X_upper = X_bin.copy()
        X_upper[:, feat_idx] = upper
        probs_upper = model.predict_proba(X_upper)

        # Local effect = difference
        local_effect = (probs_upper - probs_lower).mean(axis=0)
        ale_values[:, b] = local_effect

    # Accumulate effects
    ale_values = np.cumsum(ale_values, axis=1)

    # Centre ALE (subtract mean)
    ale_values -= ale_values.mean(axis=1, keepdims=True)

    return bin_centers, ale_values


def index(request):
    context = {}

    best_tree = None 
    best_lr   = None

    X, y, feature_cols, class_names = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Default values
    model_type  = request.POST.get('model_type', 'tree')
    lam         = float(request.POST.get('lam', 0.0))
    context['model_type'] = model_type
    context['lam']        = lam
    context['class_names'] = class_names
    context['feature_cols'] = feature_cols

    uid = str(uuid.uuid4())[:8]

    # ─────────────────────────────────────────
    # DECISION TREE
    # ─────────────────────────────────────────
    if model_type == 'tree':

        # Train trees for all leaf node values
        leaf_range = range(2, 51)
        best_tree  = None
        best_score = None
        best_leaves = None
        tree_results = []

        for max_leaves in leaf_range:
            clf = DecisionTreeClassifier(
                max_leaf_nodes=max_leaves, random_state=42
            )
            clf.fit(X_train, y_train)
            acc      = clf.score(X_test, y_test)
            n_leaves = clf.get_n_leaves()
            score    = acc - lam * n_leaves  # maximise: acc - λ*leaves
            tree_results.append({
                'max_leaves': max_leaves,
                'acc': round(acc, 4),
                'n_leaves': n_leaves,
                'score': round(score, 4)
            })
            if best_score is None or score > best_score:
                best_score  = score
                best_tree   = clf
                best_leaves = n_leaves

        context['accuracy']  = round(best_tree.score(X_test, y_test), 4)
        context['n_leaves']  = best_leaves
        context['best_score'] = round(best_score, 4)

        # ---- Plot 1: Tree visualisation ----
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            best_tree,
            feature_names=feature_cols,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=ax
        )
        ax.set_title(
            f'Decision Tree  |  Leaves: {best_leaves}  |  '
            f'Accuracy: {context["accuracy"]}  |  λ={lam}',
            fontsize=13
        )
        context['tree_url'] = save_plot(fig, f'tree_{uid}.png')

        # ---- Plot 2: Accuracy vs leaves ----
        fig, ax = plt.subplots(figsize=(8, 4))
        leaves_list = [r['n_leaves'] for r in tree_results]
        acc_list    = [r['acc']      for r in tree_results]
        ax.plot(leaves_list, acc_list, marker='o',
                color='steelblue', linewidth=2)
        ax.axvline(x=best_leaves, color='red', linestyle='--',
                   label=f'Best: {best_leaves} leaves')
        ax.set_xlabel('Number of Leaves')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'Accuracy vs Number of Leaves  (λ={lam})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        context['acc_plot_url'] = save_plot(fig, f'acc_leaves_{uid}.png')

    # ─────────────────────────────────────────
    # LOGISTIC REGRESSION
    # ─────────────────────────────────────────
    elif model_type == 'logistic':

        # C = inverse regularization strength (lower C = more regularization)
        C_range     = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                       1, 2, 5, 10, 50, 100]
        best_lr     = None
        best_score  = None
        best_C      = None
        best_nonzero = None
        lr_results  = []

        for C in C_range:
            clf = LogisticRegression(
                C=C, penalty='l1', solver='saga',
                max_iter=5000, random_state=42
            )
            clf.fit(X_train, y_train)
            acc      = clf.score(X_test, y_test)
            nonzero  = int(np.sum(np.abs(clf.coef_) > 1e-4))
            score    = acc - lam * nonzero
            lr_results.append({
                'C': C, 'acc': round(acc, 4),
                'nonzero': nonzero, 'score': round(score, 4)
            })
            if best_score is None or score > best_score:
                best_score   = score
                best_lr      = clf
                best_C       = C
                best_nonzero = nonzero

        context['accuracy']    = round(best_lr.score(X_test, y_test), 4)
        context['nonzero']     = best_nonzero
        context['best_C']      = best_C
        context['best_score']  = round(best_score, 4)

        # ---- Plot: Accuracy vs non-zero coefficients ----
        fig, ax = plt.subplots(figsize=(8, 4))
        nz_list  = [r['nonzero'] for r in lr_results]
        acc_list = [r['acc']     for r in lr_results]
        ax.plot(nz_list, acc_list, marker='o',
                color='steelblue', linewidth=2)
        ax.axvline(x=best_nonzero, color='red', linestyle='--',
                   label=f'Best: {best_nonzero} non-zero coefs')
        ax.set_xlabel('Non-zero Coefficients (complexity)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'Accuracy vs Complexity  (λ={lam})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        context['acc_plot_url'] = save_plot(fig, f'acc_lr_{uid}.png')

        df_full = load_penguins().dropna().reset_index(drop=True)
        numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        display_cols = ['species'] + numerical_cols
        context['data_points'] = df_full[display_cols].head(50).to_dict('records')
        context['data_indices'] = list(range(min(50, len(df_full))))
        context['class_names']  = class_names

        selected_idx = int(request.POST.get('selected_idx', 0))
        target_class = request.POST.get('target_class', None)
        context['selected_idx'] = selected_idx

        if target_class is not None and request.POST.get('action') == 'counterfactual':
            target_class = int(target_class)
            x = X.iloc[selected_idx]
            active_model = best_tree if model_type == 'tree' else best_lr

            result = generate_counterfactuals(x, target_class, active_model, X, N=10000, k=5)

            if result is None:
                context['cf_error'] = (
                    "No counterfactuals found. "
                    "Try a different point or target class."
                )
            else:
                cf_df, original_df = result
                context['cf_table'] = cf_df.round(3).to_html(
                    classes='data-table', index=False
                )
                context['orig_table'] = original_df.round(3).to_html(
                    classes='data-table', index=False
                )
                context['target_class_name']   = class_names[target_class]
                context['original_class_name'] = class_names[
                    active_model.predict([x])[0]
                ]

    
    # -----------------------------------------------------------------
    # FEATURE EFFECT PLOTS (PDP + ALE)
    # -----------------------------------------------------------------

    numerical_features = ['bill_length_mm', 'bill_depth_mm',
                          'flipper_length_mm', 'body_mass_g']
    context['numerical_features'] = numerical_features

    selected_feature = request.POST.get('selected_feature',
                                        numerical_features[0])
    context['selected_feature'] = selected_feature

    if request.POST.get('action') == 'feature_effect':
        active_model = best_tree if model_type == 'tree' else best_lr
        feat_vals = np.linspace(
            X[selected_feature].min(),
            X[selected_feature].max(),
            50
        )
        uid_fe = str(uuid.uuid4())[:8]
        colors_list = ['steelblue', 'tomato', 'seagreen']

        # ---- PDP Plot ----
        pdp_vals = compute_pdp(
            active_model, X, selected_feature, feat_vals,
            n_classes=len(class_names)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for c_idx, cls in enumerate(class_names):
            ax.plot(feat_vals, pdp_vals[c_idx],
                    label=cls, color=colors_list[c_idx],
                    linewidth=2)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Average Predicted Probability')
        ax.set_title(f'PDP — {selected_feature}')
        ax.legend(title='Species')
        ax.grid(True, alpha=0.3)
        context['pdp_url'] = save_plot(fig, f'pdp_{uid_fe}.png')

        # ---- ALE Plot ----
        bin_centers, ale_vals = compute_ale(
            active_model, X, selected_feature,
            n_bins=20, n_classes=len(class_names)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for c_idx, cls in enumerate(class_names):
            ax.plot(bin_centers, ale_vals[c_idx],
                    label=cls, color=colors_list[c_idx],
                    linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--',
                   alpha=0.4, linewidth=1)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Accumulated Local Effect')
        ax.set_title(f'ALE — {selected_feature}')
        ax.legend(title='Species')
        ax.grid(True, alpha=0.3)
        context['ale_url'] = save_plot(fig, f'ale_{uid_fe}.png')

        context['feature_effect_done'] = True



    return render(request, 'project2/index.html', context)