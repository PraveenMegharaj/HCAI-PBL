"""Project 2 views — thin controllers.

Every ML step lives in ml.py; every explainability primitive in
explainability.py; POST inputs are validated with forms.py.
"""
import uuid

import numpy as np
from django.shortcuts import render
from palmerpenguins import load_penguins

from . import ml, explainability
from .forms import ControlsForm, CounterfactualForm, FeatureEffectForm


def index(request):
    context = {}

    X, y, feature_cols, class_names = ml.load_and_prepare()
    X_train, X_test, y_train, y_test = ml.split(X, y)

    controls = ControlsForm(request.POST or None)
    if request.method == 'POST' and controls.is_valid():
        model_type = controls.cleaned_data['model_type']
        lam        = controls.cleaned_data['lam']
    else:
        model_type = 'tree'
        lam        = 0.0

    context.update({'model_type': model_type, 'lam': lam,
                    'class_names': class_names, 'feature_cols': feature_cols})

    uid = uuid.uuid4().hex[:8]
    best_tree = best_lr = None

    # -----------------------------------------------------------------
    # Task 1 + 2 · Decision tree with complexity control
    # -----------------------------------------------------------------
    if model_type == 'tree':
        best_tree, best_leaves, best_score, results = ml.sweep_tree(
            X_train, y_train, X_test, y_test, lam)
        acc = round(best_tree.score(X_test, y_test), 4)

        context.update({
            'accuracy':      acc,
            'n_leaves':      best_leaves,
            'best_score':    round(best_score, 4),
            'tree_url':      ml.save_plot(
                                ml.plot_decision_tree(
                                    best_tree, feature_cols, class_names,
                                    best_leaves, acc, lam),
                                f'tree_{uid}.png'),
            'acc_plot_url':  ml.save_plot(
                                ml.plot_accuracy_vs_leaves(results, best_leaves, lam),
                                f'acc_leaves_{uid}.png'),
        })

    # -----------------------------------------------------------------
    # Task 3 · L1-regularised logistic regression
    # -----------------------------------------------------------------
    elif model_type == 'logistic':
        best_lr, best_C, best_nonzero, best_score, results = ml.sweep_logistic(
            X_train, y_train, X_test, y_test, lam)
        acc = round(best_lr.score(X_test, y_test), 4)

        context.update({
            'accuracy':     acc,
            'nonzero':      best_nonzero,
            'best_C':       best_C,
            'best_score':   round(best_score, 4),
            'acc_plot_url': ml.save_plot(
                                ml.plot_accuracy_vs_nonzero(results, best_nonzero, lam),
                                f'acc_lr_{uid}.png'),
        })

    # -----------------------------------------------------------------
    # Task 4 · Counterfactual explanations — available for both models
    # -----------------------------------------------------------------
    df_full = load_penguins().dropna().reset_index(drop=True)
    display_cols = ['species'] + ml.NUMERICAL_FEATURES
    context['data_points']  = df_full[display_cols].head(50).to_dict('records')
    context['data_indices'] = list(range(min(50, len(df_full))))

    selected_idx = int(request.POST.get('selected_idx', 0))
    context['selected_idx'] = selected_idx

    if request.POST.get('action') == 'counterfactual':
        cf_form = CounterfactualForm(request.POST)
        if cf_form.is_valid():
            target_class = cf_form.cleaned_data['target_class']
            x = X.iloc[cf_form.cleaned_data['selected_idx']]
            active_model = best_tree if model_type == 'tree' else best_lr
            result = explainability.generate_counterfactuals(
                x, target_class, active_model, X, N=10000, k=5)
            if result is None:
                context['cf_error'] = ('No counterfactuals found. '
                                       'Try a different point or target class.')
            else:
                cf_df, original_df = result
                context['cf_table'] = cf_df.round(3).to_html(
                    classes='data-table', index=False)
                context['orig_table'] = original_df.round(3).to_html(
                    classes='data-table', index=False)
                context['target_class_name']   = class_names[target_class]
                context['original_class_name'] = class_names[
                    active_model.predict([x])[0]]
        else:
            context['cf_error'] = f'Invalid counterfactual request: {cf_form.errors.as_text()}'

    # -----------------------------------------------------------------
    # Task 5 · PDP + ALE
    # -----------------------------------------------------------------
    context['numerical_features'] = ml.NUMERICAL_FEATURES
    selected_feature = request.POST.get('selected_feature',
                                        ml.NUMERICAL_FEATURES[0])
    context['selected_feature'] = selected_feature

    if request.POST.get('action') == 'feature_effect':
        fe_form = FeatureEffectForm(request.POST)
        if fe_form.is_valid():
            active_model = best_tree if model_type == 'tree' else best_lr
            feat_vals = np.linspace(X[selected_feature].min(),
                                    X[selected_feature].max(), 50)
            uid_fe = uuid.uuid4().hex[:8]

            pdp = explainability.compute_pdp(
                active_model, X, selected_feature, feat_vals,
                n_classes=len(class_names))
            context['pdp_url'] = ml.save_plot(
                ml.plot_pdp(feat_vals, pdp, class_names, selected_feature),
                f'pdp_{uid_fe}.png')

            centers, ale = explainability.compute_ale(
                active_model, X, selected_feature,
                n_bins=20, n_classes=len(class_names))
            context['ale_url'] = ml.save_plot(
                ml.plot_ale(centers, ale, class_names, selected_feature),
                f'ale_{uid_fe}.png')

            context['feature_effect_done'] = True

    return render(request, 'project2/index.html', context)
