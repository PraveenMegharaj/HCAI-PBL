"""Tests for Project 4 — feature extraction + Plackett-Luce."""
import numpy as np
import pandas as pd

from django.test import TestCase

from . import ml


class PlackettLuceTests(TestCase):
    def test_two_item_matches_bradley_terry(self):
        # For n=2 the Plackett-Luce probability is the Bradley-Terry
        # softmax; log-likelihood of ranking [0, 1] with utilities [u0, u1]
        # should be u0 - log(exp(u0) + exp(u1)).
        w = np.array([1.0, -1.0])
        X = np.array([[1.0, 0.0], [0.0, 1.0]])  # utilities → [1, -1]
        ll = ml.plackett_luce_log_likelihood(w, [X])
        expected = 1.0 - np.log(np.exp(1.0) + np.exp(-1.0))
        self.assertAlmostEqual(ll, expected, places=6)

    def test_higher_utility_first_more_likely(self):
        w = np.array([1.0, 0.0])
        X1 = np.array([[2.0, 0.0], [1.0, 0.0], [0.5, 0.0]])   # utilities 2, 1, 0.5
        X2 = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])   # reverse
        self.assertGreater(
            ml.plackett_luce_log_likelihood(w, [X1]),
            ml.plackett_luce_log_likelihood(w, [X2]))

    def test_fit_recovers_direction(self):
        # Synthetic: true weight vector recovers a preference direction.
        rng = np.random.default_rng(0)
        d = 4
        w_true = np.array([1.0, -1.0, 0.5, 0.0])

        rankings = []
        for _ in range(200):
            X = rng.normal(size=(5, d))
            utils = X @ w_true + rng.normal(size=5) * 0.1
            order = np.argsort(-utils)
            rankings.append(X[order])

        w_hat = ml.fit_plackett_luce(rankings, d, lr=0.1,
                                     n_steps=200, l2=0.001)
        # Direction is right — cosine similarity should be strongly positive
        cos = (w_hat @ w_true) / (np.linalg.norm(w_hat) * np.linalg.norm(w_true))
        self.assertGreater(cos, 0.5)


class FeatureExtractionTests(TestCase):
    def test_extract_shapes(self):
        df = pd.DataFrame({
            'movie_title':    ['A', 'B', 'C'],
            'imdb_score':     [7.5, 6.0, 8.1],
            'duration':       [110, 95, 140],
            'title_year':     [2005, 1999, 2018],
            'budget':         [50e6, 10e6, 100e6],
            'gross':          [80e6, 5e6, 200e6],
            'genres':         ['Action|Adventure', 'Drama|Romance', 'Sci-Fi|Thriller'],
            'language':       ['English', 'French', 'English'],
            'content_rating': ['PG-13', 'R', 'PG'],
        })
        X, names = ml.extract_features(df)
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(X.shape[1], len(names))
        # Genre features are 0/1
        for j, name in enumerate(names):
            if name.startswith('genre_'):
                self.assertTrue(set(X[:, j].tolist()) <= {0.0, 1.0})


class RandomHelpersTests(TestCase):
    def test_random_pair_disjoint(self):
        df = pd.DataFrame({'movie_title': list('abcdefghij')})
        rng = np.random.default_rng(0)
        a, b = ml.random_pair(df, rng)
        self.assertNotEqual(a, b)

    def test_random_ranking_set_distinct(self):
        df = pd.DataFrame({'movie_title': list('abcdefghij')})
        rng = np.random.default_rng(0)
        ids = ml.random_ranking_set(df, size=5, rng=rng)
        self.assertEqual(len(set(ids)), 5)
