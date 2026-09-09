"""Task 4 — Active learning for expert competence discovery.

Setting: no expert labels available initially.  Budget of B expert queries.

Strategy: LEAST-CONFIDENT UNCERTAINTY SAMPLING.  Pick the B test-set
indices where max(model_proba) is smallest and query the expert there.

Rationale: those are exactly the points where a potential deferral would
help most, so getting expert labels there is the most information-
efficient way to estimate expert competence.  We estimate expert accuracy
conditioned on the model's predicted class, then defer whenever that
estimated accuracy exceeds the model's confidence.
"""
import numpy as np

from .ml import CLASS_NAMES


def active_learning_curve(y_true, y_pred_model, y_proba, y_pred_expert,
                          budgets=(50, 100, 250, 500, 1000, 2000)):
    """Sweep budgets and return curve of (budget, defer_rate, system_acc)."""
    y_true        = np.asarray(y_true)
    y_pred_model  = np.asarray(y_pred_model)
    y_pred_expert = np.asarray(y_pred_expert)
    y_proba       = np.asarray(y_proba)

    max_p = y_proba.max(axis=1)
    order = np.argsort(max_p)  # most uncertain first

    curve = []
    for B in budgets:
        queried_idx = order[:B]

        est_acc_per_pred_class = {}
        for c in range(len(CLASS_NAMES)):
            mask = y_pred_model[queried_idx] == c
            if mask.any():
                est_acc_per_pred_class[c] = float(
                    (y_pred_expert[queried_idx][mask] ==
                     y_true[queried_idx][mask]).mean())
            else:
                est_acc_per_pred_class[c] = 0.0

        est_exp_acc = np.array([est_acc_per_pred_class[int(c)]
                                for c in y_pred_model])
        defer       = est_exp_acc > max_p
        system_pred = np.where(defer, y_pred_expert, y_pred_model)
        system_acc  = float((system_pred == y_true).mean())

        curve.append({
            'budget':          int(B),
            'defer_rate':      round(float(defer.mean()), 4),
            'system_accuracy': round(system_acc, 4),
            'per_class_est':   {CLASS_NAMES[k]: round(v, 4)
                                for k, v in est_acc_per_pred_class.items()},
        })
    return curve
