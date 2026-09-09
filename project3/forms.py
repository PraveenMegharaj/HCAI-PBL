"""Django forms for Project 3 — used for validation."""
from django import forms


COMPETENT_CLASSES_CHOICES = [
    ('0,1', 'World + Sports'),
    ('2,3', 'Business + Sci/Tech'),
    ('0',   'World only'),
    ('1',   'Sports only'),
    ('2',   'Business only'),
    ('3',   'Sci/Tech only'),
]


class ExpertForm(forms.Form):
    """Task 2 — pick the expert's competence region."""
    competent_classes = forms.ChoiceField(
        choices=COMPETENT_CLASSES_CHOICES, initial='0,1')
    competent_acc = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.95)
    other_acc     = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.40)

    def parsed_classes(self):
        return [int(c) for c in self.cleaned_data['competent_classes'].split(',')]


class DeferralForm(forms.Form):
    """Task 3 — evaluate at a specific τ."""
    tau = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.6)


class HumanExpertSubmitForm(forms.Form):
    """Task 5 — participant labelling an article."""
    user_label = forms.IntegerField(min_value=0, max_value=3)
