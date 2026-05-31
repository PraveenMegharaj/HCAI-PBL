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


def index(request):
    context = {}

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

    return render(request, 'project2/index.html', context)