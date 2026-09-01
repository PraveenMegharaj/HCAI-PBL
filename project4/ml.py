"""Project 4 — dataset loading, feature extraction, preference model.

Task 1 — Feature representation
    We extract a compact, interpretable feature vector for each movie:

        [imdb_score, duration_z, year_z, log_budget_z, log_gross_z,
         genre_1, ..., genre_K, is_english, rating_G, rating_PG,
         rating_PG13, rating_R]

    Rationale:
    - Numerical popularity / quality signals (`imdb_score`, `duration`,
      `title_year`, `log(budget)`, `log(gross)`) capture the coarse
      quality and era of the film. They're standardised so the linear
      utility U(x)=wᵀx has coefficients on the same scale.
    - Genres are multi-hot over the top-K most frequent genres — a
      classic content-based recommender feature; each dimension is
      interpretable ("does this user like Action?"). We drop rare
      genres to avoid noisy dimensions.
    - English vs other language is one binary feature — a coarse
      proxy for the film's cultural background.
    - Content rating is one-hot for G/PG/PG-13/R — captures target
      audience (family vs adult).

    This yields ≈20 features, small enough to elicit a preference
    vector with a handful of pairwise comparisons, and each dimension
    has a clear semantic meaning — useful for a viva.

Task 2 — Plackett-Luce ranking model
    The Bradley-Terry model gives the probability that item i is
    preferred to item j as

        P(i ≻ j) = exp(U(i)) / (exp(U(i)) + exp(U(j))).

    Its natural extension to a full ranking i₁ ≻ i₂ ≻ … ≻ iₙ is the
    Plackett-Luce model:

        P(i₁ ≻ … ≻ iₙ) = ∏_{k=1}^{n} exp(U(i_k)) / Σ_{j=k}^{n} exp(U(i_j)).

    Intuitively: at each step you pick the top-ranked item from what
    is left with a soft-max over the remaining utilities. When n=2
    this collapses back to Bradley-Terry, so it is a genuine extension.
"""
import os

import numpy as np
import pandas as pd

from django.conf import settings


CACHE_DIR = os.path.join(settings.MEDIA_ROOT, 'project4')
os.makedirs(CACHE_DIR, exist_ok=True)

CSV_PATH   = os.path.join(CACHE_DIR, 'movie_metadata.csv')
DATASET_URL = ('https://raw.githubusercontent.com/sundeepblue/'
               'movie_rating_prediction/master/movie_metadata.csv')

TOP_GENRES = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance',
              'Adventure', 'Crime', 'Sci-Fi', 'Fantasy', 'Horror',
              'Family', 'Mystery', 'Biography', 'Animation']

CONTENT_RATINGS = ['G', 'PG', 'PG-13', 'R']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_movies():
    """Load the IMDB 5000 movie metadata CSV (download on first access)."""
    if not os.path.exists(CSV_PATH):
        import requests
        r = requests.get(DATASET_URL, timeout=120)
        r.raise_for_status()
        with open(CSV_PATH, 'wb') as f:
            f.write(r.content)
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['movie_title', 'imdb_score', 'genres'])
    df = df.reset_index(drop=True)
    df['movie_title'] = df['movie_title'].str.strip()
    return df


# ---------------------------------------------------------------------------
# Task 1 — Feature extraction
# ---------------------------------------------------------------------------

def _z(series):
    """Z-score a series with NaNs → replace missing with 0."""
    s = pd.to_numeric(series, errors='coerce')
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return np.zeros(len(series))
    return ((s - mu) / sd).fillna(0.0).values


def extract_features(df):
    """Return (X, feature_names) where X is (n_movies, n_features)."""
    n = len(df)

    imdb  = pd.to_numeric(df['imdb_score'], errors='coerce').fillna(
                pd.to_numeric(df['imdb_score'], errors='coerce').mean()).values
    dur_z = _z(df.get('duration', pd.Series([np.nan] * n)))
    yr_z  = _z(df.get('title_year', pd.Series([np.nan] * n)))
    budget = pd.to_numeric(df.get('budget', pd.Series([np.nan] * n)),
                           errors='coerce')
    gross  = pd.to_numeric(df.get('gross',  pd.Series([np.nan] * n)),
                           errors='coerce')
    log_budget_z = _z(np.log1p(budget.fillna(0).clip(lower=0)))
    log_gross_z  = _z(np.log1p(gross.fillna(0).clip(lower=0)))

    genre_cols = []
    genre_series = df['genres'].fillna('').str.split('|')
    for g in TOP_GENRES:
        genre_cols.append(genre_series.apply(lambda gs: int(g in gs)).values)
    genres_mat = np.column_stack(genre_cols) if genre_cols else np.zeros((n, 0))

    lang = df.get('language', pd.Series(['English'] * n)).fillna('').astype(str)
    is_english = (lang.str.lower() == 'english').astype(int).values

    rating = df.get('content_rating', pd.Series([''] * n)).fillna('').astype(str)
    rating_cols = []
    for r in CONTENT_RATINGS:
        rating_cols.append((rating == r).astype(int).values)
    ratings_mat = np.column_stack(rating_cols)

    X = np.column_stack([
        imdb, dur_z, yr_z, log_budget_z, log_gross_z,
        genres_mat, is_english, ratings_mat,
    ]).astype(float)

    names = (['imdb_score', 'duration_z', 'year_z',
              'log_budget_z', 'log_gross_z']
             + [f'genre_{g}' for g in TOP_GENRES]
             + ['is_english']
             + [f'rating_{r}' for r in CONTENT_RATINGS])

    assert X.shape[1] == len(names), f'{X.shape[1]} vs {len(names)}'
    return X, names


# ---------------------------------------------------------------------------
# Task 2 — Plackett-Luce likelihood + estimation
# ---------------------------------------------------------------------------

def plackett_luce_log_likelihood(w, X_rankings):
    """Sum of log Plackett-Luce probabilities over a list of rankings.

    X_rankings : list of ndarrays, each shape (n_k, d), rows ordered
                 from most-preferred to least-preferred.
    """
    total = 0.0
    for X in X_rankings:
        utils = X @ w
        for k in range(len(utils) - 1):
            log_denom = np.logaddexp.reduce(utils[k:])
            total += utils[k] - log_denom
    return total


def _pl_gradient(w, X_rankings):
    """Gradient of the Plackett-Luce log-likelihood (for use in gradient descent)."""
    grad = np.zeros_like(w)
    for X in X_rankings:
        utils = X @ w
        for k in range(len(utils) - 1):
            sub_utils = utils[k:]
            sub_X     = X[k:]
            weights   = np.exp(sub_utils - np.max(sub_utils))
            weights  /= weights.sum()
            grad += X[k] - weights @ sub_X
    return grad


def fit_plackett_luce(rankings, d, lr=0.05, n_steps=200, l2=0.01, seed=0):
    """Very small gradient-ascent MLE for the preference vector w ∈ R^d.

    `rankings` is a list of ndarrays, each already ordered most-to-least
    preferred. Regularised with an isotropic Gaussian prior (l2 term).
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=d)
    for _ in range(n_steps):
        g = _pl_gradient(w, rankings) - l2 * w
        w = w + lr * g / max(1, len(rankings))
    return w


# ---------------------------------------------------------------------------
# Sampling helpers used by the user-study interface
# ---------------------------------------------------------------------------

def random_pair(df, rng):
    idx = rng.choice(len(df), size=2, replace=False)
    return int(idx[0]), int(idx[1])


def random_ranking_set(df, size, rng):
    idx = rng.choice(len(df), size=size, replace=False)
    return [int(i) for i in idx]


def summary_for_ids(df, ids):
    """Return a list of small dicts for the given movie row indices."""
    out = []
    for i in ids:
        row = df.iloc[int(i)]
        out.append({
            'id':       int(i),
            'title':    row['movie_title'],
            'year':     int(row['title_year']) if pd.notna(row.get('title_year')) else None,
            'genres':   row['genres'].replace('|', ', ') if pd.notna(row.get('genres')) else '',
            'director': row.get('director_name') if pd.notna(row.get('director_name')) else '',
            'imdb':     float(row['imdb_score']) if pd.notna(row.get('imdb_score')) else None,
            'runtime':  int(row['duration']) if pd.notna(row.get('duration')) else None,
        })
    return out
