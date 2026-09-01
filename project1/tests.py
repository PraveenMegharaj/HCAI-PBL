"""Tests for Project 1 ML helpers.

These exercise the pure logic in ml.py without spinning up Django's
request cycle, so `python manage.py test project1` runs fast.
"""
from io import StringIO

import pandas as pd
from django.test import TestCase

from . import ml


class DetectProblemTypeTests(TestCase):
    def test_string_target_is_classification(self):
        df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': ['a', 'b', 'a', 'b']})
        self.assertEqual(ml.detect_problem_type(df), 'classification')

    def test_few_unique_numeric_is_classification(self):
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [0, 1, 0]})
        self.assertEqual(ml.detect_problem_type(df), 'classification')

    def test_many_unique_numeric_is_regression(self):
        df = pd.DataFrame({'x': range(50), 'y': [i * 1.5 for i in range(50)]})
        self.assertEqual(ml.detect_problem_type(df), 'regression')


class RegistryTests(TestCase):
    def test_classification_registry_shape(self):
        m = ml.get_models('classification')
        self.assertIn('KNN', m)
        self.assertIn('Logistic Regression', m)

    def test_regression_registry_shape(self):
        m = ml.get_models('regression')
        self.assertIn('KNN', m)
        self.assertIn('Linear Regression', m)

    def test_metrics_shape(self):
        self.assertEqual(ml.get_metrics('classification'), ['Accuracy', 'F1 Score'])
        self.assertEqual(ml.get_metrics('regression'),     ['R2 Score', 'MSE'])


class PrepareDataframeTests(TestCase):
    def test_drops_leading_id_column(self):
        csv = StringIO('id,x,y\n1,10,0\n2,20,1\n')
        df = ml.prepare_dataframe(csv)
        self.assertEqual(list(df.columns), ['x', 'y'])
