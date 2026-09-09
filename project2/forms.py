"""Django forms for Project 2 — used for validation."""
from django import forms

from .ml import NUMERICAL_FEATURES


MODEL_TYPE_CHOICES = [
    ('tree',     'Decision Tree'),
    ('logistic', 'Logistic Regression (L1)'),
]


class ControlsForm(forms.Form):
    """Model family + λ complexity penalty."""
    model_type = forms.ChoiceField(choices=MODEL_TYPE_CHOICES, initial='tree')
    lam        = forms.FloatField(min_value=0.0, max_value=0.1, initial=0.0)


class CounterfactualForm(forms.Form):
    """Counterfactual request tied to a specific data point + target class."""
    model_type   = forms.ChoiceField(choices=MODEL_TYPE_CHOICES)
    lam          = forms.FloatField(min_value=0.0, max_value=0.1)
    selected_idx = forms.IntegerField(min_value=0)
    target_class = forms.IntegerField(min_value=0, max_value=2)


class FeatureEffectForm(forms.Form):
    """PDP/ALE request for one numerical feature."""
    model_type       = forms.ChoiceField(choices=MODEL_TYPE_CHOICES)
    lam              = forms.FloatField(min_value=0.0, max_value=0.1)
    selected_feature = forms.ChoiceField(
        choices=[(f, f) for f in NUMERICAL_FEATURES])
