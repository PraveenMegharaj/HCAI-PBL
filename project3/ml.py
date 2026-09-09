"""Baseline classifier + dataset loading for Project 3 (AG News)."""
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from django.conf import settings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


CACHE_DIR    = os.path.join(settings.MEDIA_ROOT, 'project3')
os.makedirs(CACHE_DIR, exist_ok=True)

CLASS_NAMES  = ['World', 'Sports', 'Business', 'Sci/Tech']
MODEL_PATH   = os.path.join(CACHE_DIR, 'baseline_model.pkl')
METRICS_PATH = os.path.join(CACHE_DIR, 'baseline_metrics.pkl')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ag_news():
    """Load AG News train/test as pandas dataframes (text, label)."""
    train_csv = os.path.join(CACHE_DIR, 'ag_news_train.csv')
    test_csv  = os.path.join(CACHE_DIR, 'ag_news_test.csv')

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        return pd.read_csv(train_csv), pd.read_csv(test_csv)

    try:
        from datasets import load_dataset
        ds = load_dataset('fancyzhx/ag_news')
        train_df = pd.DataFrame(ds['train'])
        test_df  = pd.DataFrame(ds['test'])
    except ImportError:
        import requests
        urls = {
            train_csv: ("https://raw.githubusercontent.com/mhjabreel/"
                        "CharCnn_Keras/master/data/ag_news_csv/train.csv"),
            test_csv:  ("https://raw.githubusercontent.com/mhjabreel/"
                        "CharCnn_Keras/master/data/ag_news_csv/test.csv"),
        }
        for path, url in urls.items():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(path, 'wb') as f:
                f.write(r.content)

        def _load(path):
            df = pd.read_csv(path, header=None,
                             names=['label', 'title', 'description'])
            df['text']  = (df['title'].fillna('') + '. ' +
                           df['description'].fillna(''))
            df['label'] = df['label'] - 1
            return df[['text', 'label']]

        train_df = _load(train_csv)
        test_df  = _load(test_csv)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv,   index=False)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Task 1 — Baseline classifier
# ---------------------------------------------------------------------------

def train_baseline():
    """TF-IDF (1-2 grams, 20k features, sublinear TF) + Logistic Regression."""
    train_df, test_df = load_ag_news()

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=3,
        sublinear_tf=True, strip_accents='unicode')
    X_train = vectorizer.fit_transform(train_df['text'].astype(str))
    X_test  = vectorizer.transform(test_df['text'].astype(str))
    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, solver='lbfgs')
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True)

    metrics = {
        'accuracy':   round(float(accuracy), 4),
        'n_train':    int(len(train_df)),
        'n_test':     int(len(test_df)),
        'n_features': int(X_train.shape[1]),
        'cm':         cm.tolist(),
        'per_class':  {c: {
                          'precision': round(float(report[c]['precision']), 4),
                          'recall':    round(float(report[c]['recall']),    4),
                          'f1':        round(float(report[c]['f1-score']),  4),
                          'support':   int(report[c]['support']),
                      } for c in CLASS_NAMES},
        'y_test':  y_test.tolist(),
        'y_pred':  y_pred.tolist(),
        'y_proba': y_proba.tolist(),
    }
    bundle = {'vectorizer': vectorizer, 'model': clf}

    with open(MODEL_PATH,   'wb') as f: pickle.dump(bundle, f)
    with open(METRICS_PATH, 'wb') as f: pickle.dump(metrics, f)
    return metrics, bundle


def load_cached_baseline():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
        return None, None
    with open(MODEL_PATH,   'rb') as f: bundle  = pickle.load(f)
    with open(METRICS_PATH, 'rb') as f: metrics = pickle.load(f)
    return metrics, bundle


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight', dpi=110)
    plt.close(fig)
    return settings.MEDIA_URL + filename


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar=False, linewidths=0.4)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    return fig


def plot_per_class_bar(per_class_dict, metric_key, title, color='steelblue'):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names  = list(per_class_dict.keys())
    values = [per_class_dict[n][metric_key] for n in names]
    bars = ax.bar(names, values, color=color, edgecolor='white')
    ax.set_ylim(0, 1.05); ax.set_ylabel(metric_key); ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=9)
    return fig


def plot_deferral_sweep(sweep):
    fig, ax = plt.subplots(figsize=(7, 4))
    taus = [s['tau'] for s in sweep]
    sys  = [s['system_accuracy'] for s in sweep]
    dfr  = [s['defer_rate']      for s in sweep]
    ax.plot(taus, sys, marker='o', color='#2e8b57',
            label='system accuracy', linewidth=2)
    ax.plot(taus, dfr, marker='s', color='#8b2e8b',
            label='defer rate',      linewidth=2)
    ax.set_xlabel('τ (confidence threshold)')
    ax.set_ylabel('rate / accuracy')
    ax.set_title('Task 3 — deferral policy sweep')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    return fig


def plot_active_learning_curve(curve):
    fig, ax = plt.subplots(figsize=(7, 4))
    budgets = [c['budget']          for c in curve]
    sys     = [c['system_accuracy'] for c in curve]
    dfr     = [c['defer_rate']      for c in curve]
    ax.plot(budgets, sys, marker='o', color='#2e8b57',
            label='system accuracy', linewidth=2)
    ax.plot(budgets, dfr, marker='s', color='#8b2e8b',
            label='defer rate',      linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('# expert queries (budget)')
    ax.set_ylabel('rate / accuracy')
    ax.set_title('Task 4 — system accuracy vs expert budget')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    return fig
