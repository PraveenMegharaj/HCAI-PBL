"""Django forms for Project 4 — validation."""
from django import forms


AGE_GROUP_CHOICES = [
    ('',        '—'),
    ('under18', 'Under 18'),
    ('18-24',   '18-24'),
    ('25-34',   '25-34'),
    ('35-49',   '35-49'),
    ('50plus',  '50+'),
]

EXPERIENCE_CHOICES = [
    ('',       '—'),
    ('none',   'None'),
    ('some',   'Some (course / tutorial)'),
    ('a_lot',  'A lot (professional)'),
]


class ConsentForm(forms.Form):
    consented    = forms.BooleanField(required=True,
                                      label='I have read the information above '
                                            'and I consent to take part.')
    age_group    = forms.ChoiceField(choices=AGE_GROUP_CHOICES, required=False)
    prior_ml_exp = forms.ChoiceField(choices=EXPERIENCE_CHOICES, required=False)


class PairwiseResponseForm(forms.Form):
    trial_index     = forms.IntegerField(min_value=0)
    movie_a_id      = forms.IntegerField(min_value=0)
    movie_b_id      = forms.IntegerField(min_value=0)
    chosen_movie_id = forms.IntegerField(min_value=0)
    response_ms     = forms.IntegerField(required=False, min_value=0)


class RankingResponseForm(forms.Form):
    trial_index     = forms.IntegerField(min_value=0)
    shown_ids_json  = forms.CharField()   # comma-separated list of ids
    ranked_ids_json = forms.CharField()   # comma-separated list of ids
    response_ms     = forms.IntegerField(required=False, min_value=0)
