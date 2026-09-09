"""Task 4 storage — participants, pairwise choices, and rankings."""
import uuid

from django.db import models


class Participant(models.Model):
    """One row per person who starts the user study."""
    DESIGN_PAIRWISE = 'pairwise'
    DESIGN_RANKING  = 'ranking'
    DESIGN_CHOICES = [
        (DESIGN_PAIRWISE, 'Design 1 · Pairwise choice'),
        (DESIGN_RANKING,  'Design 2 · Rank a set of 10'),
    ]

    participant_id = models.UUIDField(default=uuid.uuid4, unique=True,
                                      editable=False)
    condition      = models.CharField(max_length=16, choices=DESIGN_CHOICES)
    consented      = models.BooleanField(default=False)
    age_group      = models.CharField(max_length=32, blank=True)
    prior_ml_exp   = models.CharField(max_length=32, blank=True)
    started_at     = models.DateTimeField(auto_now_add=True)
    completed_at   = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f'{str(self.participant_id)[:8]} · {self.condition}'


class PairwiseResponse(models.Model):
    """Design 1 — a single 'A vs B, I chose X' event."""
    participant     = models.ForeignKey(Participant, on_delete=models.CASCADE,
                                        related_name='pairwise_responses')
    trial_index     = models.IntegerField()
    movie_a_id      = models.IntegerField()
    movie_b_id      = models.IntegerField()
    chosen_movie_id = models.IntegerField()
    response_ms     = models.IntegerField(null=True, blank=True)
    submitted_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['participant', 'trial_index']
        indexes  = [models.Index(fields=['participant', 'trial_index'])]

    def __str__(self):
        return (f'{self.participant_id} · t{self.trial_index} · '
                f'{self.movie_a_id} vs {self.movie_b_id} → {self.chosen_movie_id}')


class RankingResponse(models.Model):
    """Design 2 — one full ranking of 10 movies."""
    participant     = models.ForeignKey(Participant, on_delete=models.CASCADE,
                                        related_name='ranking_responses')
    trial_index     = models.IntegerField()
    shown_ids_json  = models.JSONField()   # list of movie row indices
    ranked_ids_json = models.JSONField()   # same ids, ordered top → bottom
    response_ms     = models.IntegerField(null=True, blank=True)
    submitted_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['participant', 'trial_index']
        indexes  = [models.Index(fields=['participant', 'trial_index'])]

    def __str__(self):
        return (f'{self.participant_id} · t{self.trial_index} · '
                f'{len(self.ranked_ids_json)} movies ranked')
