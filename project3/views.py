import os
import uuid
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render
from django.conf import settings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)


CACHE_DIR = os.path.join(settings.MEDIA_ROOT, 'project3')
os.makedirs(CACHE_DIR, exist_ok=True)

CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
MODEL_PATH   = os.path.join(CACHE_DIR, 'baseline_model.pkl')
METRICS_PATH = os.path.join(CACHE_DIR, 'baseline_metrics.pkl')


# =========================================================================
# Data loading
# =========================================================================

def load_ag_news():
    """Load AG News train/test as pandas dataframes with columns text, label.

    Tries in order: local CSV cache -> HuggingFace `datasets` library ->
    fetch original CSV mirror over HTTP. Caches result to media/project3/.
    """
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
            train_csv: "https://raw.githubusercontent.com/mhjabreel/"
                       "CharCnn_Keras/master/data/ag_news_csv/train.csv",
            test_csv:  "https://raw.githubusercontent.com/mhjabreel/"
                       "CharCnn_Keras/master/data/ag_news_csv/test.csv",
        }
        for path, url in urls.items():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(path, 'wb') as f:
                f.write(r.content)

        def _load_raw(path):
            df = pd.read_csv(path, header=None,
                             names=['label', 'title', 'description'])
            df['text']  = df['title'].fillna('') + '. ' + df['description'].fillna('')
            df['label'] = df['label'] - 1   # original labels are 1..4
            return df[['text', 'label']]

        train_df = _load_raw(train_csv)
        test_df  = _load_raw(test_csv)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv,   index=False)
    return train_df, test_df


# =========================================================================
# Task 1 — Baseline classifier
# =========================================================================

def train_baseline():
    """Train TF-IDF + LogisticRegression on full AG News train split.

    Returns (metrics_dict, model_bundle). Caches both to disk so the page
    loads instantly on subsequent requests.
    """
    train_df, test_df = load_ag_news()

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        strip_accents='unicode',
    )
    X_train = vectorizer.fit_transform(train_df['text'].astype(str))
    X_test  = vectorizer.transform(test_df['text'].astype(str))
    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    clf = LogisticRegression(
        C=1.0, max_iter=1000, n_jobs=-1, solver='lbfgs'
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES,
        digits=4, output_dict=True
    )

    metrics = {
        'accuracy':    round(float(accuracy), 4),
        'n_train':     int(len(train_df)),
        'n_test':      int(len(test_df)),
        'n_features':  int(X_train.shape[1]),
        'cm':          cm.tolist(),
        'per_class':   {c: {
                            'precision': round(float(report[c]['precision']), 4),
                            'recall':    round(float(report[c]['recall']),    4),
                            'f1':        round(float(report[c]['f1-score']),  4),
                            'support':   int(report[c]['support']),
                        } for c in CLASS_NAMES},
        'y_test':      y_test.tolist(),
        'y_pred':      y_pred.tolist(),
    }
    bundle = {'vectorizer': vectorizer, 'model': clf}

    with open(MODEL_PATH,   'wb') as f:
        pickle.dump(bundle, f)
    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)

    return metrics, bundle


def load_cached_baseline():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
        return None, None
    with open(MODEL_PATH,   'rb') as f:
        bundle = pickle.load(f)
    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)
    return metrics, bundle


# =========================================================================
# Task 2 — Simulated expert
# =========================================================================

class SimulatedExpert:
    """An expert whose competence depends on the TRUE class of the input.

    Motivation: real human experts specialise. A sports journalist labels
    sports articles with high accuracy but is unreliable on business news.
    We model this by giving the expert a set of `competent_classes` on
    which they answer correctly with high probability, and every other
    class becomes noisy.

    Parameters
    ----------
    competent_classes : iterable[int]
        Class labels the expert specialises in.
    competent_acc : float
        P(correct | true label in competent_classes).
    other_acc : float
        P(correct | true label NOT in competent_classes).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, competent_classes, competent_acc=0.95,
                 other_acc=0.40, n_classes=4, seed=42):
        self.competent_classes = set(int(c) for c in competent_classes)
        self.competent_acc     = float(competent_acc)
        self.other_acc         = float(other_acc)
        self.n_classes         = int(n_classes)
        self.rng               = np.random.default_rng(seed)

    def predict(self, y_true):
        y_true = np.asarray(y_true)
        preds  = np.empty_like(y_true)
        for i, y in enumerate(y_true):
            is_competent = int(y) in self.competent_classes
            p_correct    = self.competent_acc if is_competent else self.other_acc
            if self.rng.random() < p_correct:
                preds[i] = y
            else:
                others = [c for c in range(self.n_classes) if c != int(y)]
                preds[i] = self.rng.choice(others)
        return preds

    def describe(self):
        comp = sorted(self.competent_classes)
        comp_names = [CLASS_NAMES[c] for c in comp]
        other_names = [n for i, n in enumerate(CLASS_NAMES) if i not in self.competent_classes]
        return {
            'competent_classes': comp_names,
            'other_classes':     other_names,
            'competent_acc':     self.competent_acc,
            'other_acc':         self.other_acc,
        }


def evaluate_expert(expert, y_true):
    y_pred = expert.predict(y_true)
    y_true = np.asarray(y_true)

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = y_true == c
        per_class[name] = {
            'support':  int(mask.sum()),
            'accuracy': round(float((y_pred[mask] == c).mean()) if mask.any() else 0.0, 4),
        }
    return {
        'overall_accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'per_class':        per_class,
        'cm':               confusion_matrix(y_true, y_pred).tolist(),
        'y_pred':           y_pred.tolist(),
    }


# =========================================================================
# Plot helpers
# =========================================================================

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
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    return fig


def plot_per_class_bar(per_class_dict, metric_key, title, color='steelblue'):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names  = list(per_class_dict.keys())
    values = [per_class_dict[n][metric_key] for n in names]
    bars = ax.bar(names, values, color=color, edgecolor='white')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=9)
    return fig


# =========================================================================
# Django view
# =========================================================================

def index(request):
    context = {'class_names': CLASS_NAMES}

    # Load or train baseline
    metrics, bundle = load_cached_baseline()

    if request.method == 'POST' and request.POST.get('action') == 'train_baseline':
        try:
            metrics, bundle = train_baseline()
            context['train_success'] = True
        except Exception as e:
            context['error'] = f"Baseline training failed: {e}"

    if metrics is not None:
        uid = str(uuid.uuid4())[:8]
        context['baseline'] = {
            'accuracy':   metrics['accuracy'],
            'n_train':    metrics['n_train'],
            'n_test':     metrics['n_test'],
            'n_features': metrics['n_features'],
            'per_class':  metrics['per_class'],
        }
        context['baseline_cm_url'] = save_plot(
            plot_confusion_matrix(metrics['cm'],
                                  f"Baseline confusion matrix  (acc={metrics['accuracy']:.4f})"),
            f'p3_baseline_cm_{uid}.png',
        )

        # Task 2 — simulated expert evaluation
        # Read expert config from POST (or use defaults)
        competent_str = request.POST.get('competent_classes', '0,1')  # World & Sports
        try:
            competent = [int(c) for c in competent_str.split(',') if c.strip() != '']
        except ValueError:
            competent = [0, 1]
        competent_acc = float(request.POST.get('competent_acc', 0.95))
        other_acc     = float(request.POST.get('other_acc', 0.40))

        expert = SimulatedExpert(competent, competent_acc=competent_acc,
                                 other_acc=other_acc, n_classes=len(CLASS_NAMES))

        y_test = np.array(metrics['y_test'])
        expert_metrics = evaluate_expert(expert, y_test)

        context['expert'] = {
            'description':      expert.describe(),
            'overall_accuracy': expert_metrics['overall_accuracy'],
            'per_class':        expert_metrics['per_class'],
            'competent_str':    competent_str,
            'competent_acc':    competent_acc,
            'other_acc':        other_acc,
        }
        context['expert_cm_url'] = save_plot(
            plot_confusion_matrix(
                expert_metrics['cm'],
                f"Expert confusion matrix  (acc={expert_metrics['overall_accuracy']:.4f})",
            ),
            f'p3_expert_cm_{uid}.png',
        )
        context['expert_bar_url'] = save_plot(
            plot_per_class_bar(
                expert_metrics['per_class'], 'accuracy',
                'Expert per-class accuracy', color='#8b2e8b',
            ),
            f'p3_expert_bar_{uid}.png',
        )

    return render(request, 'project3/index.html', context)
