from django.contrib import admin

from .models import Participant, PairwiseResponse, RankingResponse


class PairwiseResponseInline(admin.TabularInline):
    model = PairwiseResponse
    extra = 0


class RankingResponseInline(admin.TabularInline):
    model = RankingResponse
    extra = 0


@admin.register(Participant)
class ParticipantAdmin(admin.ModelAdmin):
    list_display  = ('participant_id', 'condition', 'consented',
                     'age_group', 'prior_ml_exp',
                     'started_at', 'completed_at')
    list_filter   = ('condition', 'consented', 'age_group')
    search_fields = ('participant_id',)
    ordering      = ('-started_at',)
    inlines       = [PairwiseResponseInline, RankingResponseInline]


@admin.register(PairwiseResponse)
class PairwiseResponseAdmin(admin.ModelAdmin):
    list_display = ('participant', 'trial_index', 'movie_a_id',
                    'movie_b_id', 'chosen_movie_id', 'response_ms',
                    'submitted_at')
    list_filter  = ('participant__condition',)


@admin.register(RankingResponse)
class RankingResponseAdmin(admin.ModelAdmin):
    list_display = ('participant', 'trial_index', 'response_ms',
                    'submitted_at')
    list_filter  = ('participant__condition',)
