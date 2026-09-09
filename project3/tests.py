"""Tests for Project 3 helpers."""
import numpy as np
from django.test import TestCase

from . import active_learning, deferral, experts


class SimulatedExpertTests(TestCase):
    def test_competent_class_high_accuracy(self):
        # Expert competent on class 0 with 100% accuracy → always right.
        e = experts.SimulatedExpert([0], competent_acc=1.0, other_acc=0.0)
        y = np.zeros(50, dtype=int)
        preds = e.predict(y)
        self.assertTrue((preds == y).all())

    def test_other_classes_low_accuracy(self):
        # Expert weak on class 2 with 0% accuracy → never right.
        e = experts.SimulatedExpert([0], competent_acc=1.0, other_acc=0.0)
        y = np.full(50, 2, dtype=int)
        preds = e.predict(y)
        self.assertTrue((preds != y).all())


class DeferralTests(TestCase):
    def test_tau_zero_never_defers(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred_model = np.array([0, 0, 0, 0])
        y_proba = np.array([[0.9, 0.05, 0.03, 0.02]] * 4)
        y_pred_expert = np.array([0, 1, 2, 3])
        r = deferral.eval_deferral(y_true, y_pred_model, y_proba,
                                   y_pred_expert, tau=0.0)
        self.assertEqual(r['defer_rate'], 0.0)
        self.assertEqual(r['system_accuracy'], 0.25)  # only model on class 0 is right

    def test_tau_one_always_defers(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred_model = np.array([0, 0, 0, 0])
        y_proba = np.array([[0.9, 0.05, 0.03, 0.02]] * 4)
        y_pred_expert = np.array([0, 1, 2, 3])
        r = deferral.eval_deferral(y_true, y_pred_model, y_proba,
                                   y_pred_expert, tau=1.0)
        self.assertEqual(r['defer_rate'], 1.0)
        self.assertEqual(r['system_accuracy'], 1.0)


class ActiveLearningTests(TestCase):
    def test_curve_shape(self):
        n = 500
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, size=n)
        y_pred = rng.integers(0, 4, size=n)
        y_proba = rng.random(size=(n, 4))
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        y_pred_expert = rng.integers(0, 4, size=n)
        curve = active_learning.active_learning_curve(
            y_true, y_pred, y_proba, y_pred_expert,
            budgets=(10, 50, 100))
        self.assertEqual(len(curve), 3)
        for row in curve:
            self.assertIn('budget', row)
            self.assertIn('system_accuracy', row)
