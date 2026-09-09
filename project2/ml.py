"""Data loading, model training, and plot helpers for Project 2."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from django.conf import settings

from palmerpenguins import load_penguins
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


FEATURE_COLS = ['island', 'bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g', 'sex', 'year']
NUMERICAL_FEATURES = ['bill_length_mm', 'bill_depth_mm',
                      'flipper_length_mm', 'body_mass_g']
TARGET_COL = 'species'


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return settings.MEDIA_URL + filename


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_and_prepare():
    """Load Palmer Penguins, drop NAs, label-encode string columns."""
    df = load_penguins().dropna()
    df['island'] = LabelEncoder().fit_transform(df['island'])
    df['sex']    = LabelEncoder().fit_transform(df['sex'])

    X = df[FEATURE_COLS]
    y = LabelEncoder().fit_transform(df[TARGET_COL])
    class_names = sorted(df[TARGET_COL].unique().tolist())
    return X, y, FEATURE_COLS, class_names


def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Task 1 + 2 — Decision tree with complexity control
# ---------------------------------------------------------------------------

def sweep_tree(X_train, y_train, X_test, y_test, lam,
               leaf_range=range(2, 51)):
    """Return (best_tree, best_leaves, best_score, results)."""
    results = []
    best_tree = None
    best_score = None
    best_leaves = None
    for max_leaves in leaf_range:
        clf = DecisionTreeClassifier(max_leaf_nodes=max_leaves,
                                     random_state=42)
        clf.fit(X_train, y_train)
        acc      = clf.score(X_test, y_test)
        n_leaves = clf.get_n_leaves()
        score    = acc - lam * n_leaves
        results.append({'max_leaves': max_leaves, 'acc': round(acc, 4),
                        'n_leaves': n_leaves, 'score': round(score, 4)})
        if best_score is None or score > best_score:
            best_score, best_tree, best_leaves = score, clf, n_leaves
    return best_tree, best_leaves, best_score, results


# ---------------------------------------------------------------------------
# Task 3 — L1-regularised logistic regression
# ---------------------------------------------------------------------------

def sweep_logistic(X_train, y_train, X_test, y_test, lam,
                   C_range=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                            1, 2, 5, 10, 50, 100)):
    """L1-penalised logistic regression sweep.

    Ω(f) = number of non-zero coefficients.
    """
    results = []
    best_lr = None
    best_score = None
    best_C = None
    best_nonzero = None
    for C in C_range:
        clf = LogisticRegression(C=C, penalty='l1', solver='saga',
                                 max_iter=5000, random_state=42)
        clf.fit(X_train, y_train)
        acc     = clf.score(X_test, y_test)
        nonzero = int(np.sum(np.abs(clf.coef_) > 1e-4))
        score   = acc - lam * nonzero
        results.append({'C': C, 'acc': round(acc, 4),
                        'nonzero': nonzero, 'score': round(score, 4)})
        if best_score is None or score > best_score:
            best_score, best_lr = score, clf
            best_C, best_nonzero = C, nonzero
    return best_lr, best_C, best_nonzero, best_score, results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_decision_tree(clf, feature_cols, class_names, leaves, accuracy, lam):
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(clf, feature_names=feature_cols, class_names=class_names,
              filled=True, rounded=True, ax=ax)
    ax.set_title(f'Decision Tree  |  Leaves: {leaves}  |  '
                 f'Accuracy: {accuracy}  |  λ={lam}', fontsize=13)
    return fig


def plot_accuracy_vs_leaves(results, best_leaves, lam):
    fig, ax = plt.subplots(figsize=(8, 4))
    leaves = [r['n_leaves'] for r in results]
    accs   = [r['acc']      for r in results]
    ax.plot(leaves, accs, marker='o', color='steelblue', linewidth=2)
    ax.axvline(x=best_leaves, color='red', linestyle='--',
               label=f'Best: {best_leaves} leaves')
    ax.set_xlabel('Number of Leaves'); ax.set_ylabel('Test Accuracy')
    ax.set_title(f'Accuracy vs Number of Leaves  (λ={lam})')
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def plot_accuracy_vs_nonzero(results, best_nonzero, lam):
    fig, ax = plt.subplots(figsize=(8, 4))
    nz  = [r['nonzero'] for r in results]
    acc = [r['acc']     for r in results]
    ax.plot(nz, acc, marker='o', color='steelblue', linewidth=2)
    ax.axvline(x=best_nonzero, color='red', linestyle='--',
               label=f'Best: {best_nonzero} non-zero coefs')
    ax.set_xlabel('Non-zero Coefficients (complexity)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'Accuracy vs Complexity  (λ={lam})')
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def plot_pdp(feat_vals, pdp_vals, class_names, feature_name):
    colors = ['steelblue', 'tomato', 'seagreen']
    fig, ax = plt.subplots(figsize=(8, 5))
    for c_idx, cls in enumerate(class_names):
        ax.plot(feat_vals, pdp_vals[c_idx], label=cls,
                color=colors[c_idx], linewidth=2)
    ax.set_xlabel(feature_name); ax.set_ylabel('Average Predicted Probability')
    ax.set_title(f'PDP — {feature_name}')
    ax.legend(title='Species'); ax.grid(True, alpha=0.3)
    return fig


def plot_ale(bin_centers, ale_vals, class_names, feature_name):
    colors = ['steelblue', 'tomato', 'seagreen']
    fig, ax = plt.subplots(figsize=(8, 5))
    for c_idx, cls in enumerate(class_names):
        ax.plot(bin_centers, ale_vals[c_idx], label=cls,
                color=colors[c_idx], linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel(feature_name); ax.set_ylabel('Accumulated Local Effect')
    ax.set_title(f'ALE — {feature_name}')
    ax.legend(title='Species'); ax.grid(True, alpha=0.3)
    return fig
