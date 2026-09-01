"""Tests for Project 2 ML + explainability helpers."""
import numpy as np
from django.test import TestCase

from . import ml, explainability


class LoadPenguinsTests(TestCase):
    def test_load_shapes(self):
        X, y, feature_cols, class_names = ml.load_and_prepare()
        self.assertEqual(X.shape[1], len(feature_cols))
        self.assertEqual(len(y), len(X))
        self.assertEqual(sorted(class_names), ['Adelie', 'Chinstrap', 'Gentoo'])

    def test_split(self):
        X, y, *_ = ml.load_and_prepare()
        X_tr, X_te, y_tr, y_te = ml.split(X, y)
        self.assertEqual(len(y_tr) + len(y_te), len(y))


class SweepTests(TestCase):
    def test_tree_sweep_returns_best(self):
        X, y, *_ = ml.load_and_prepare()
        X_tr, X_te, y_tr, y_te = ml.split(X, y)
        clf, leaves, score, results = ml.sweep_tree(
            X_tr, y_tr, X_te, y_te, lam=0.0, leaf_range=range(2, 6))
        self.assertIsNotNone(clf)
        self.assertGreater(leaves, 0)
        self.assertGreaterEqual(len(results), 3)


class ExplainabilityTests(TestCase):
    def test_pdp_shape(self):
        from sklearn.tree import DecisionTreeClassifier
        X, y, *_ = ml.load_and_prepare()
        clf = DecisionTreeClassifier(max_leaf_nodes=8, random_state=42).fit(X, y)
        pdp = explainability.compute_pdp(
            clf, X, 'bill_length_mm',
            feature_vals=np.linspace(30, 60, 10), n_classes=3)
        self.assertEqual(pdp.shape, (3, 10))

    def test_ale_returns_arrays(self):
        from sklearn.tree import DecisionTreeClassifier
        X, y, *_ = ml.load_and_prepare()
        clf = DecisionTreeClassifier(max_leaf_nodes=8, random_state=42).fit(X, y)
        centers, ale = explainability.compute_ale(
            clf, X, 'bill_length_mm', n_bins=10, n_classes=3)
        self.assertEqual(ale.shape[0], 3)
        self.assertEqual(len(centers), ale.shape[1])
