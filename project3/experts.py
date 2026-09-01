"""Task 2 — Simulated experts with region-specific competence."""
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

from .ml import CLASS_NAMES


class SimulatedExpert:
    """Expert competent on a specific subset of classes.

    Motivation — real human experts specialise. A sports journalist labels
    sports articles accurately and is unreliable on business news. We
    parametrise this with `competent_classes` (subset of labels), a high
    accuracy on those, and a lower accuracy elsewhere.
    """
    def __init__(self, competent_classes, competent_acc=0.95,
                 other_acc=0.40, n_classes=4, seed=42):
        self.competent_classes = set(int(c) for c in competent_classes)
        self.competent_acc = float(competent_acc)
        self.other_acc     = float(other_acc)
        self.n_classes     = int(n_classes)
        self.rng           = np.random.default_rng(seed)

    def predict(self, y_true):
        y_true = np.asarray(y_true)
        preds  = np.empty_like(y_true)
        for i, y in enumerate(y_true):
            p = (self.competent_acc if int(y) in self.competent_classes
                 else self.other_acc)
            if self.rng.random() < p:
                preds[i] = y
            else:
                others = [c for c in range(self.n_classes) if c != int(y)]
                preds[i] = self.rng.choice(others)
        return preds

    def describe(self):
        comp = sorted(self.competent_classes)
        return {
            'competent_classes': [CLASS_NAMES[c] for c in comp],
            'other_classes':     [n for i, n in enumerate(CLASS_NAMES)
                                  if i not in self.competent_classes],
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
            'accuracy': round(float((y_pred[mask] == c).mean())
                              if mask.any() else 0.0, 4),
        }
    return {
        'overall_accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'per_class':        per_class,
        'cm':               confusion_matrix(y_true, y_pred).tolist(),
        'y_pred':           y_pred.tolist(),
    }
