# Human-Centric AI — Project-Based Learning (TUHH)

Four interactive Django applications, one per course project, accessible from a shared
home page. Each app explores a different facet of building machine-learning systems
that keep a human in the loop.

## Group

| Name             | Matriculation |
|------------------|---------------|
| Praveen Megharaj | 672067        |
| Niharika Kiran   | 672070        |

## Setup

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. apply database migrations
python manage.py migrate

# 4. run the server
python manage.py runserver
```

Then open **http://127.0.0.1:8000/home/** and navigate to any project.

**Notes on first run**
- Projects 3 and 4 download their datasets automatically on first access
  (AG News for P3, IMDB-5000 for P4) and cache them under `media/`. This needs an
  internet connection the first time and may take a moment.
- The `media/` directory is created automatically; generated plots and cached
  models/datasets are written there.

## Projects

### Project 1 · Supervised Learning Interface
Upload a CSV (last column = target), explore it (preview, descriptive stats,
histograms, class-coloured scatter, correlation heatmap), and train a model with an
interactive hyperparameter sweep. Supports both classification and regression with
automatic problem-type detection and a manual override. Four models per problem type.

### Project 2 · Explainability
Palmer Penguins with complexity-controlled models. A λ slider selects the model
maximising `acc_test − λ·Ω(f)` (Ω = number of leaves for the decision tree, number of
non-zero coefficients for L1 logistic regression). Includes hand-implemented
counterfactual explanations (MAD-weighted L1 ranking) and hand-implemented **PDP and
ALE** feature-effect plots — no external XAI library. The counterfactual and
feature-effect regions are linked to the selected model type and λ.

### Project 3 · Active Learning for Learning-to-Defer
AG News topic classification (TF-IDF + logistic regression baseline), a simulated
region-competent expert, a confidence-threshold deferral policy, and least-confident
uncertainty sampling for expert-competence discovery under a query budget. Includes an
optional interactive interface where the user plays the expert (Task 5).
**A PDF report is downloadable from the project page** (button at the bottom).

### Project 4 · Preference Elicitation
A runnable user-study interface (IMDB-5000) comparing two elicitation designs —
pairwise choice vs. ranking a set of ten — under a Plackett-Luce extension of the
Bradley-Terry model. Flow: landing → consent → study → complete. Responses are stored
anonymously per participant. **The landing page provides the PDF report download and
the link to start the study.**

## Regenerating the report PDFs (optional)

The reports are already committed under `static/project3/report.pdf` and
`static/project4/report.pdf`. To rebuild them from the pipeline:

```bash
python manage.py build_report        # Project 3
python manage.py build_report_p4     # Project 4
```

## Admin

Study/labelling data (P3 Task 5, P4 responses) is visible in the Django admin:

```bash
python manage.py createsuperuser
# then visit http://127.0.0.1:8000/admin/
```

## Tests

```bash
python manage.py test
```

## Project structure

Each app keeps views as thin controllers, with ML logic in dedicated modules
(`ml.py`, `explainability.py`, `deferral.py`, `active_learning.py`), input validation
in `forms.py`, persistence in `models.py`, and unit tests in `tests.py`.
