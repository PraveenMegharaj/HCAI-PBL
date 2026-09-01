"""Django forms for Project 1. Used for validation of POST data.

Templates render the widgets by hand so the app keeps its custom styling —
these forms only validate `cleaned_data` in the view.
"""
from django import forms


PROBLEM_TYPE_CHOICES = [
    ('auto',           'Auto-detect'),
    ('classification', 'Classification'),
    ('regression',     'Regression'),
]

TEST_SIZE_CHOICES = [
    ('0.1', '10%'),
    ('0.2', '20%'),
    ('0.3', '30%'),
    ('0.4', '40%'),
]


class UploadForm(forms.Form):
    csv_file = forms.FileField(required=True)
    problem_type = forms.ChoiceField(
        choices=PROBLEM_TYPE_CHOICES, initial='auto', required=False)

    def clean_csv_file(self):
        f = self.cleaned_data['csv_file']
        if not f.name.lower().endswith('.csv'):
            raise forms.ValidationError('File must be a .csv')
        return f


class TrainForm(forms.Form):
    csv_filename = forms.CharField(max_length=200)
    problem_type = forms.ChoiceField(
        choices=[('classification', 'Classification'),
                 ('regression',     'Regression')])
    model_name   = forms.CharField(max_length=100)
    test_size    = forms.ChoiceField(choices=TEST_SIZE_CHOICES, initial='0.2')
    metric       = forms.CharField(max_length=50)
