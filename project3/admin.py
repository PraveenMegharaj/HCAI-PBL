from django.contrib import admin

from .models import HumanExpertLabel


@admin.register(HumanExpertLabel)
class HumanExpertLabelAdmin(admin.ModelAdmin):
    list_display  = ('session_key', 'article_idx', 'user_label',
                     'true_label', 'is_correct', 'submitted_at')
    list_filter   = ('is_correct', 'user_label', 'true_label')
    search_fields = ('session_key',)
    ordering      = ('-submitted_at',)
