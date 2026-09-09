"""Task 5 — persist human-expert labelling sessions.

Task 5 is the interactive interface where the user plays the role of the
expert. We used to keep the running log in `request.session`; now each
labelling event is also stored in the database so the app has a proper
audit trail (visible in /admin) and so the log survives across sessions
if a participant returns with the same session key.
"""
from django.db import models


class HumanExpertLabel(models.Model):
    session_key   = models.CharField(max_length=64, db_index=True)
    article_idx   = models.IntegerField()
    user_label    = models.IntegerField()
    true_label    = models.IntegerField()
    is_correct    = models.BooleanField()
    submitted_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-submitted_at']

    def __str__(self):
        return (f'{self.session_key[:8]}… #{self.article_idx} · '
                f'user={self.user_label} vs true={self.true_label} · '
                f'{"✓" if self.is_correct else "✗"}')
