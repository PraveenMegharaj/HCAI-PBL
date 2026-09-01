"""Project 1 views — thin controllers.

All ML logic lives in ml.py; input validation in forms.py; upload metadata
is logged in models.py (UploadedDataset). This view only orchestrates.
"""
import uuid

import pandas as pd

from django.shortcuts import render

from . import ml
from .forms import UploadForm, TrainForm
from .models import UploadedDataset


def index(request):
    context = {}

    if request.method != 'POST':
        return render(request, 'project1/index.html', context)

    action = request.POST.get('action', 'upload')

    if action == 'upload':
        _handle_upload(request, context)
    elif action == 'train':
        _handle_train(request, context)

    return render(request, 'project1/index.html', context)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _handle_upload(request, context):
    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        context['error'] = 'Please upload a valid CSV file.'
        return

    csv_file = form.cleaned_data['csv_file']
    override = form.cleaned_data.get('problem_type') or 'auto'

    try:
        df = ml.prepare_dataframe(csv_file)
        feature_cols = df.columns[:-1].tolist()
        target_col   = df.columns[-1]

        problem_type = (ml.detect_problem_type(df)
                        if override == 'auto' else override)

        filename, uid = ml.store_csv(df)

        UploadedDataset.objects.create(
            filename=filename,
            original_name=csv_file.name,
            n_samples=len(df),
            n_features=len(feature_cols),
            target_col=target_col,
            problem_type=problem_type,
        )

        context.update({
            'csv_filename': filename,
            'feature_cols': feature_cols,
            'target_col':   target_col,
            'n_samples':    len(df),
            'n_features':   len(feature_cols),
            'problem_type': problem_type,
            'table_html':   df.head(10).to_html(classes='data-table', index=False),
            'stats_html':   df.describe().round(2).to_html(classes='data-table'),
            'hist_url':     ml.save_plot(ml.build_histograms(df, feature_cols), f'hist_{uid}.png'),
            'scatter_url':  ml.save_plot(ml.build_scatter(df, feature_cols, target_col, problem_type), f'scatter_{uid}.png'),
            'heatmap_url':  ml.save_plot(ml.build_heatmap(df), f'heatmap_{uid}.png'),
            'models':       list(ml.get_models(problem_type).keys()),
            'metrics':      ml.get_metrics(problem_type),
            'success':      True,
        })

    except Exception as e:
        context['error'] = f'Error reading file: {e}'


def _handle_train(request, context):
    form = TrainForm(request.POST)
    if not form.is_valid():
        context['error'] = f'Invalid training form: {form.errors.as_text()}'
        return

    csv_filename = form.cleaned_data['csv_filename']
    problem_type = form.cleaned_data['problem_type']
    model_name   = form.cleaned_data['model_name']
    test_size    = float(form.cleaned_data['test_size'])
    metric       = form.cleaned_data['metric']

    df = ml.load_cached_csv(csv_filename)
    results, best_score, best_param, param_name = ml.train_sweep(
        df, model_name, problem_type, test_size, metric)

    uid = uuid.uuid4().hex[:8]
    if param_name is not None:
        fig = ml.build_score_plot(results, param_name, metric, model_name, best_param)
        context['score_plot_url'] = ml.save_plot(fig, f'score_{uid}.png')

    results_df = pd.DataFrame(results)
    if param_name:
        results_df.columns = [param_name, metric]

    context.update({
        'results_html':   results_df.to_html(classes='data-table', index=False),
        'best_score':     best_score,
        'best_param':     best_param,
        'best_param_name': param_name,
        'model_name':     model_name,
        'metric':         metric,
        'problem_type':   problem_type,
        'csv_filename':   csv_filename,
        'feature_cols':   df.columns[:-1].tolist(),
        'target_col':     df.columns[-1],
        'models':         list(ml.get_models(problem_type).keys()),
        'metrics':        ml.get_metrics(problem_type),
        'train_success':  True,
    })
