"""Machine learning helpers for Project 1.

Kept separate from views.py so views stay thin (controller-only) and
so this file can be unit-tested without pulling in Django's request cycle.
"""
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from django.conf import settings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return settings.MEDIA_URL + filename


# ---------------------------------------------------------------------------
# Problem type detection + model / metric registry
# ---------------------------------------------------------------------------

def detect_problem_type(df):
    """Heuristic — last column is the target; object dtype or ≤10 unique
    values → classification, else regression."""
    target = df.iloc[:, -1]
    if target.dtype == 'object' or target.nunique() <= 10:
        return 'classification'
    return 'regression'


def get_models(problem_type):
    if problem_type == 'classification':
        return {
            'KNN':                 {'model': KNeighborsClassifier,   'param': 'n_neighbors',  'param_range': range(1, 21)},
            'Decision Tree':       {'model': DecisionTreeClassifier, 'param': 'max_depth',    'param_range': range(1, 21)},
            'Random Forest':       {'model': RandomForestClassifier, 'param': 'n_estimators', 'param_range': range(10, 110, 10)},
            'Logistic Regression': {'model': LogisticRegression,     'param': 'C',            'param_range': [0.01, 0.1, 1, 10, 100]},
        }
    return {
        'KNN':               {'model': KNeighborsRegressor,   'param': 'n_neighbors',  'param_range': range(1, 21)},
        'Decision Tree':     {'model': DecisionTreeRegressor, 'param': 'max_depth',    'param_range': range(1, 21)},
        'Random Forest':     {'model': RandomForestRegressor, 'param': 'n_estimators', 'param_range': range(10, 110, 10)},
        'Linear Regression': {'model': LinearRegression,      'param': None,           'param_range': [None]},
    }


def get_metrics(problem_type):
    return ['Accuracy', 'F1 Score'] if problem_type == 'classification' else ['R2 Score', 'MSE']


def compute_score(y_true, y_pred, metric):
    if metric == 'Accuracy':
        return round(accuracy_score(y_true, y_pred), 4)
    if metric == 'F1 Score':
        return round(f1_score(y_true, y_pred, average='weighted'), 4)
    if metric == 'R2 Score':
        return round(r2_score(y_true, y_pred), 4)
    if metric == 'MSE':
        return round(mean_squared_error(y_true, y_pred), 4)
    return 0.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def build_histograms(df, feature_cols):
    n_cols = len(feature_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, feature_cols):
        df[col].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
        ax.set_title(col, fontsize=10)
    fig.suptitle('Feature Distributions', fontsize=13)
    fig.tight_layout()
    return fig


def build_scatter(df, feature_cols, target_col, problem_type):
    fig, ax = plt.subplots(figsize=(6, 5))
    x_col = feature_cols[0]
    y_col = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    if problem_type == 'classification':
        classes = df[target_col].unique()
        colors = plt.cm.Set1.colors
        for i, cls in enumerate(classes):
            subset = df[df[target_col] == cls]
            ax.scatter(subset[x_col], subset[y_col], label=str(cls),
                       color=colors[i % len(colors)], alpha=0.7)
        ax.legend(title=target_col)
    else:
        scatter = ax.scatter(df[x_col], df[y_col],
                             c=df[target_col], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=target_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')
    return fig


def build_heatmap(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    numeric_df = df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr().round(2), annot=True,
                cmap='coolwarm', ax=ax, linewidths=0.5)
    ax.set_title('Correlation Heatmap')
    return fig


def build_score_plot(results, param_name, metric, model_name, best_param):
    fig, ax = plt.subplots(figsize=(7, 4))
    params = [r['param'] for r in results]
    scores = [r['score'] for r in results]
    ax.plot(params, scores, marker='o', color='steelblue', linewidth=2)
    ax.axvline(x=best_param, color='red', linestyle='--',
               label=f'Best: {param_name}={best_param}')
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(f'{model_name} — {metric} vs {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def prepare_dataframe(csv_file_obj):
    """Read a CSV, drop leading id/index column if any, return dataframe."""
    df = pd.read_csv(csv_file_obj)
    if df.columns[0].lower() in ('id', 'index'):
        df = df.drop(columns=df.columns[0])
    return df


def load_cached_csv(filename):
    return pd.read_csv(os.path.join(settings.MEDIA_ROOT, filename))


def store_csv(df):
    """Persist df to media/ and return the generated basename."""
    uid = str(uuid.uuid4())[:8]
    filename = f'data_{uid}.csv'
    df.to_csv(os.path.join(settings.MEDIA_ROOT, filename), index=False)
    return filename, uid


def train_sweep(df, model_name, problem_type, test_size, metric):
    """Run the full hyperparameter sweep and return (results, best_score, best_param, param_name)."""
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]

    X = df[feature_cols]
    y = df[target_col]

    if problem_type == 'classification' and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    cfg = get_models(problem_type)[model_name]
    ModelClass = cfg['model']
    param_name = cfg['param']
    param_range = cfg['param_range']

    results = []
    best_score = None
    best_param = None
    for val in param_range:
        if param_name is None:
            clf = ModelClass()
        else:
            kwargs = {param_name: val}
            if 'random_state' in ModelClass().get_params():
                kwargs['random_state'] = 42
            clf = ModelClass(**kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = compute_score(y_test, y_pred, metric)
        results.append({'param': val, 'score': score})
        if best_score is None or score > best_score:
            best_score = score
            best_param = val
    return results, best_score, best_param, param_name
