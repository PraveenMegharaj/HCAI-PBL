"""Task 3 — Confidence-threshold learning-to-defer policy.

Design choice: with both the classifier's probabilities and the expert's
labels available at evaluation time, the simplest well-motivated deferral
rule is:

    DEFER TO EXPERT  iff  max(model_proba) < τ

The classifier's top-class probability is a calibrated-enough measure
of its own confidence, and the expert's competence structure means
deferring on uncertain cases only helps when those cases fall in the
expert's strong region.  The τ threshold controls the trade-off between
deferring often (more expert queries) and rarely (more model errors).
"""
import numpy as np


def eval_deferral(y_true, y_pred_model, y_proba, y_pred_expert, tau):
    """System-level metrics for a confidence-threshold policy at a given τ."""
    y_true        = np.asarray(y_true)
    y_pred_model  = np.asarray(y_pred_model)
    y_pred_expert = np.asarray(y_pred_expert)
    y_proba       = np.asarray(y_proba)

    max_p  = y_proba.max(axis=1)
    defer  = max_p < tau
    system = np.where(defer, y_pred_expert, y_pred_model)

    return {
        'tau':             float(tau),
        'defer_rate':      round(float(defer.mean()), 4),
        'system_accuracy': round(float((system == y_true).mean()), 4),
        'model_only_acc':  round(float((y_pred_model  == y_true).mean()), 4),
        'expert_only_acc': round(float((y_pred_expert == y_true).mean()), 4),
        'on_deferred_expert_acc': round(
            float((y_pred_expert[defer] == y_true[defer]).mean())
            if defer.any() else 0.0, 4),
        'on_kept_model_acc': round(
            float((y_pred_model[~defer] == y_true[~defer]).mean())
            if (~defer).any() else 0.0, 4),
    }


def sweep_deferral(y_true, y_pred_model, y_proba, y_pred_expert, n_taus=21):
    """Sweep τ across [0, 1] and return all per-τ metrics."""
    taus = np.linspace(0.0, 1.0, n_taus)
    return [eval_deferral(y_true, y_pred_model, y_proba, y_pred_expert, t)
            for t in taus]
