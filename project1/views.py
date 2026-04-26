import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from django.shortcuts import render
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, r2_score)
from sklearn.preprocessing import LabelEncoder


def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return settings.MEDIA_URL + filename


def detect_problem_type(df):
    target = df.iloc[:, -1]
    if target.dtype == 'object' or target.nunique() <= 10:
        return 'classification'
    return 'regression'


def get_models(problem_type):
    if problem_type == 'classification':
        return {
            'KNN':                {'model': KNeighborsClassifier,  'param': 'n_neighbors', 'param_range': range(1, 21)},
            'Decision Tree':      {'model': DecisionTreeClassifier,'param': 'max_depth',   'param_range': range(1, 21)},
            'Random Forest':      {'model': RandomForestClassifier,'param': 'n_estimators','param_range': range(10, 110, 10)},
            'Logistic Regression':{'model': LogisticRegression,    'param': 'C',           'param_range': [0.01, 0.1, 1, 10, 100]},
        }
    else:
        return {
            'KNN':              {'model': KNeighborsRegressor,  'param': 'n_neighbors', 'param_range': range(1, 21)},
            'Decision Tree':    {'model': DecisionTreeRegressor,'param': 'max_depth',   'param_range': range(1, 21)},
            'Random Forest':    {'model': RandomForestRegressor,'param': 'n_estimators','param_range': range(10, 110, 10)},
            'Linear Regression':{'model': LinearRegression,     'param': None,          'param_range': [None]},
        }


def get_metrics(problem_type):
    if problem_type == 'classification':
        return ['Accuracy', 'F1 Score']
    else:
        return ['R2 Score', 'MSE']


def compute_score(y_true, y_pred, metric):
    if metric == 'Accuracy':
        return round(accuracy_score(y_true, y_pred), 4)
    elif metric == 'F1 Score':
        return round(f1_score(y_true, y_pred, average='weighted'), 4)
    elif metric == 'R2 Score':
        return round(r2_score(y_true, y_pred), 4)
    elif metric == 'MSE':
        return round(mean_squared_error(y_true, y_pred), 4)
    return 0


def index(request):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action', 'upload')

        # ─────────────────────────────────────────
        # ACTION 1: Upload & Visualize
        # ─────────────────────────────────────────
        if action == 'upload':
            csv_file = request.FILES.get('csv_file')
            problem_type_override = request.POST.get('problem_type', 'auto')

            if csv_file and csv_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(csv_file)

                    # Drop ID column if present
                    if df.columns[0].lower() in ['id', 'index']:
                        df = df.drop(columns=df.columns[0])

                    feature_cols = df.columns[:-1].tolist()
                    target_col = df.columns[-1]

                    # Detect problem type
                    if problem_type_override == 'auto':
                        problem_type = detect_problem_type(df)
                    else:
                        problem_type = problem_type_override

                    # Save CSV to session via media
                    uid = str(uuid.uuid4())[:8]
                    csv_path = os.path.join(settings.MEDIA_ROOT, f'data_{uid}.csv')
                    df.to_csv(csv_path, index=False)

                    context['csv_filename'] = f'data_{uid}.csv'
                    context['feature_cols'] = feature_cols
                    context['target_col'] = target_col
                    context['n_samples'] = len(df)
                    context['n_features'] = len(feature_cols)
                    context['problem_type'] = problem_type
                    context['table_html'] = df.head(10).to_html(classes='data-table', index=False)
                    context['stats_html'] = df.describe().round(2).to_html(classes='data-table')

                    # Histograms
                    n_cols = len(feature_cols)
                    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
                    if n_cols == 1:
                        axes = [axes]
                    for ax, col in zip(axes, feature_cols):
                        df[col].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
                        ax.set_title(col, fontsize=10)
                    fig.suptitle('Feature Distributions', fontsize=13)
                    fig.tight_layout()
                    context['hist_url'] = save_plot(fig, f'hist_{uid}.png')

                    # Scatter plot
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
                    context['scatter_url'] = save_plot(fig, f'scatter_{uid}.png')

                    # Heatmap
                    fig, ax = plt.subplots(figsize=(7, 5))
                    numeric_df = df.select_dtypes(include='number')
                    sns.heatmap(numeric_df.corr().round(2), annot=True,
                                cmap='coolwarm', ax=ax, linewidths=0.5)
                    ax.set_title('Correlation Heatmap')
                    context['heatmap_url'] = save_plot(fig, f'heatmap_{uid}.png')

                    # Pass model choices for training form
                    context['models'] = list(get_models(problem_type).keys())
                    context['metrics'] = get_metrics(problem_type)
                    context['success'] = True

                except Exception as e:
                    context['error'] = f"Error reading file: {str(e)}"
            else:
                context['error'] = "Please upload a valid CSV file."

        # ─────────────────────────────────────────
        # ACTION 2: Train Model
        # ─────────────────────────────────────────
        elif action == 'train':
            csv_filename = request.POST.get('csv_filename')
            problem_type = request.POST.get('problem_type')
            model_name = request.POST.get('model_name')
            test_size = float(request.POST.get('test_size', 0.2))
            metric = request.POST.get('metric')

            # Reload CSV
            csv_path = os.path.join(settings.MEDIA_ROOT, csv_filename)
            df = pd.read_csv(csv_path)

            feature_cols = df.columns[:-1].tolist()
            target_col = df.columns[-1]

            X = df[feature_cols]
            y = df[target_col]

            # Encode target if classification and string
            le = None
            if problem_type == 'classification' and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Get model config
            models_config = get_models(problem_type)
            model_config = models_config[model_name]
            ModelClass = model_config['model']
            param_name = model_config['param']
            param_range = model_config['param_range'] if 'param_range' in model_config else model_config['range']

            # Train for each hyperparameter value
            results = []
            best_score = None
            best_param = None

            for param_val in param_range:
                if param_name is None:
                    clf = ModelClass()
                else:
                    clf = ModelClass(**{param_name: param_val, 'random_state': 42}
                                     if 'random_state' in ModelClass().get_params()
                                     else {param_name: param_val})
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = compute_score(y_test, y_pred, metric)
                results.append({'param': param_val, 'score': score})

                if best_score is None or score > best_score:
                    best_score = score
                    best_param = param_val

            # Plot score vs hyperparameter
            uid = str(uuid.uuid4())[:8]
            if param_name is not None:
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
                context['score_plot_url'] = save_plot(fig, f'score_{uid}.png')

            # Results table
            results_df = pd.DataFrame(results)
            if param_name:
                results_df.columns = [param_name, metric]
            context['results_html'] = results_df.to_html(classes='data-table', index=False)
            context['best_score'] = best_score
            context['best_param'] = best_param
            context['best_param_name'] = param_name
            context['model_name'] = model_name
            context['metric'] = metric
            context['problem_type'] = problem_type
            context['csv_filename'] = csv_filename
            context['feature_cols'] = feature_cols
            context['target_col'] = target_col
            context['models'] = list(models_config.keys())
            context['metrics'] = get_metrics(problem_type)
            context['train_success'] = True

    return render(request, "project1/index.html", context)