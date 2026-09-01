"""Django models for Project 1.

An `UploadedDataset` row is created each time the user uploads a CSV.
Even though we do not currently query these rows from the UI, having a
Django model gives us:

* a clean record of what the app has seen (visible in /admin)
* a place to hang future features (delete/reload/rename uploads)
* an example of using models.py, as the professor's PDF suggests
"""
from django.db import models


class UploadedDataset(models.Model):
    filename      = models.CharField(max_length=200,
                                     help_text="stored basename inside MEDIA_ROOT")
    original_name = models.CharField(max_length=200)
    n_samples     = models.IntegerField()
    n_features    = models.IntegerField()
    target_col    = models.CharField(max_length=100)
    problem_type  = models.CharField(
        max_length=20,
        choices=[('classification', 'Classification'),
                 ('regression',     'Regression')])
    uploaded_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return (f'{self.original_name} · {self.problem_type} · '
                f'{self.n_samples} × {self.n_features}')
