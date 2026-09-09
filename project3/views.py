"""Project 3 views — thin controller.

Task 1 in ml.py, Task 2 in experts.py, Task 3 in deferral.py,
Task 4 in active_learning.py, Task 5 uses models.py + Django sessions.
"""
import uuid

import numpy as np
from django.shortcuts import render

from . import ml, active_learning, deferral, experts
from .forms import DeferralForm, ExpertForm, HumanExpertSubmitForm
from .models import HumanExpertLabel


def index(request):
    context = {'class_names': ml.CLASS_NAMES}

    metrics, bundle = ml.load_cached_baseline()

    # -------- Task 1: (re)train baseline
    if request.method == 'POST' and request.POST.get('action') == 'train_baseline':
        try:
            metrics, bundle = ml.train_baseline()
            context['train_success'] = True
        except Exception as e:
            context['error'] = f'Baseline training failed: {e}'

    if metrics is None:
        return render(request, 'project3/index.html', context)

    uid = uuid.uuid4().hex[:8]
    context['baseline'] = {
        'accuracy':   metrics['accuracy'],
        'n_train':    metrics['n_train'],
        'n_test':     metrics['n_test'],
        'n_features': metrics['n_features'],
        'per_class':  metrics['per_class'],
    }
    context['baseline_cm_url'] = ml.save_plot(
        ml.plot_confusion_matrix(
            metrics['cm'],
            f"Baseline confusion matrix  (acc={metrics['accuracy']:.4f})"),
        f'p3_baseline_cm_{uid}.png')

    # -------- Task 2: simulated expert
    expert_form = ExpertForm(request.POST or None)
    if expert_form.is_valid():
        competent     = expert_form.parsed_classes()
        competent_acc = expert_form.cleaned_data['competent_acc']
        other_acc     = expert_form.cleaned_data['other_acc']
        competent_str = expert_form.cleaned_data['competent_classes']
    else:
        competent, competent_acc, other_acc = [0, 1], 0.95, 0.40
        competent_str = '0,1'

    expert = experts.SimulatedExpert(
        competent, competent_acc=competent_acc, other_acc=other_acc,
        n_classes=len(ml.CLASS_NAMES))

    y_test = np.array(metrics['y_test'])
    expert_metrics = experts.evaluate_expert(expert, y_test)

    context['expert'] = {
        'description':      expert.describe(),
        'overall_accuracy': expert_metrics['overall_accuracy'],
        'per_class':        expert_metrics['per_class'],
        'competent_str':    competent_str,
        'competent_acc':    competent_acc,
        'other_acc':        other_acc,
    }
    context['expert_cm_url'] = ml.save_plot(
        ml.plot_confusion_matrix(
            expert_metrics['cm'],
            f"Expert confusion matrix  (acc={expert_metrics['overall_accuracy']:.4f})"),
        f'p3_expert_cm_{uid}.png')
    context['expert_bar_url'] = ml.save_plot(
        ml.plot_per_class_bar(expert_metrics['per_class'], 'accuracy',
                              'Expert per-class accuracy', color='#8b2e8b'),
        f'p3_expert_bar_{uid}.png')

    y_pred_model  = np.array(metrics['y_pred'])
    y_proba       = np.array(metrics['y_proba'])
    y_pred_expert = np.array(expert_metrics['y_pred'])

    # -------- Task 3: learning-to-defer
    sweep = deferral.sweep_deferral(y_test, y_pred_model, y_proba, y_pred_expert)
    context['defer_sweep_url'] = ml.save_plot(
        ml.plot_deferral_sweep(sweep), f'p3_defer_sweep_{uid}.png')

    tau_default = 0.6
    if request.POST.get('action') == 'defer_eval':
        df = DeferralForm(request.POST)
        if df.is_valid():
            tau_default = df.cleaned_data['tau']
    context['defer'] = {
        'tau':    tau_default,
        'result': deferral.eval_deferral(y_test, y_pred_model, y_proba,
                                         y_pred_expert, tau_default),
        'sweep':  sweep,
    }

    # -------- Task 4: active learning curve
    curve = active_learning.active_learning_curve(
        y_test, y_pred_model, y_proba, y_pred_expert)
    context['al_curve_url'] = ml.save_plot(
        ml.plot_active_learning_curve(curve), f'p3_al_curve_{uid}.png')
    context['al_curve'] = curve

    # -------- Task 5: interactive expert
    _handle_task5(request, y_test, context)

    return render(request, 'project3/index.html', context)


# ---------------------------------------------------------------------------
# Task 5 — participant plays the expert
# ---------------------------------------------------------------------------

def _handle_task5(request, y_test, context):
    if not request.session.session_key:
        request.session.save()
    sess_key = request.session.session_key

    if request.POST.get('action') == 'human_expert_next':
        request.session['t5_idx'] = int(
            np.random.default_rng().integers(0, len(y_test)))

    idx = int(request.session.get('t5_idx', 0))

    if request.POST.get('action') == 'human_expert_submit':
        form = HumanExpertSubmitForm(request.POST)
        if form.is_valid():
            user_label = form.cleaned_data['user_label']
            true_label = int(y_test[idx])
            HumanExpertLabel.objects.create(
                session_key=sess_key,
                article_idx=idx,
                user_label=user_label,
                true_label=true_label,
                is_correct=(user_label == true_label))

    _, test_df = ml.load_ag_news()
    context['t5_sample_idx']  = idx
    context['t5_sample_text'] = test_df.iloc[idx]['text']

    log_qs = HumanExpertLabel.objects.filter(
        session_key=sess_key).order_by('-submitted_at')[:20]
    log = [{
        'idx':        r.article_idx,
        'user_label': r.user_label,
        'true_label': r.true_label,
        'correct':    r.is_correct,
    } for r in log_qs]
    context['t5_log'] = log

    if log:
        n_ok = sum(1 for r in log if r['correct'])
        context['t5_running_acc'] = round(n_ok / len(log), 3)
    else:
        context['t5_running_acc'] = None
