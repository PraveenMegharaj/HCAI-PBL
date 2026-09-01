"""Project 4 — Preference elicitation user study.

Views:
  landing   — cover page with PDF download and 'Start' button
  consent   — informed-consent form
  study     — the elicitation loop (Design 1 or Design 2, chosen at
              random when the participant is created)
  complete  — thank-you page
"""
import json
import random

import numpy as np
from django.shortcuts import redirect, render
from django.utils import timezone

from . import ml
from .forms import ConsentForm, PairwiseResponseForm, RankingResponseForm
from .models import Participant, PairwiseResponse, RankingResponse


N_TRIALS_PAIRWISE = 20          # Design 1
N_TRIALS_RANKING  = 5           # Design 2
RANKING_SET_SIZE  = 10


# ---------------------------------------------------------------------------
# Landing
# ---------------------------------------------------------------------------

def landing(request):
    return render(request, 'project4/landing.html', {
        'n_trials_pairwise': N_TRIALS_PAIRWISE,
        'n_trials_ranking':  N_TRIALS_RANKING,
        'ranking_set_size':  RANKING_SET_SIZE,
    })


# ---------------------------------------------------------------------------
# Consent → creates Participant, assigns condition, redirects to study
# ---------------------------------------------------------------------------

def consent(request):
    if request.method == 'POST':
        form = ConsentForm(request.POST)
        if form.is_valid():
            condition = _assign_condition()
            p = Participant.objects.create(
                condition=condition,
                consented=True,
                age_group=form.cleaned_data.get('age_group') or '',
                prior_ml_exp=form.cleaned_data.get('prior_ml_exp') or '',
            )
            request.session['p4_participant_id'] = str(p.participant_id)
            request.session['p4_trial_index'] = 0
            return redirect('project4:study')
    else:
        form = ConsentForm()
    return render(request, 'project4/consent.html', {'form': form})


def _assign_condition():
    """Balanced random assignment: pick whichever condition currently has
    fewer participants (breaking ties randomly)."""
    n_pw = Participant.objects.filter(
        condition=Participant.DESIGN_PAIRWISE).count()
    n_rk = Participant.objects.filter(
        condition=Participant.DESIGN_RANKING).count()
    if n_pw < n_rk:
        return Participant.DESIGN_PAIRWISE
    if n_rk < n_pw:
        return Participant.DESIGN_RANKING
    return random.choice([Participant.DESIGN_PAIRWISE, Participant.DESIGN_RANKING])


# ---------------------------------------------------------------------------
# The study loop
# ---------------------------------------------------------------------------

def study(request):
    participant = _current_participant(request)
    if participant is None:
        return redirect('project4:consent')

    df = ml.load_movies()
    rng = np.random.default_rng()

    if request.method == 'POST':
        _save_response(request, participant, df)
        request.session['p4_trial_index'] = int(request.session.get(
            'p4_trial_index', 0)) + 1

    trial_index = int(request.session.get('p4_trial_index', 0))
    total_trials = (N_TRIALS_PAIRWISE
                    if participant.condition == Participant.DESIGN_PAIRWISE
                    else N_TRIALS_RANKING)

    if trial_index >= total_trials:
        participant.completed_at = timezone.now()
        participant.save(update_fields=['completed_at'])
        return redirect('project4:complete')

    ctx = {
        'participant':  participant,
        'trial_index':  trial_index,
        'total_trials': total_trials,
        'progress_pct': int(100 * trial_index / total_trials),
    }

    if participant.condition == Participant.DESIGN_PAIRWISE:
        a, b = ml.random_pair(df, rng)
        movies = ml.summary_for_ids(df, [a, b])
        ctx.update({
            'movie_a': movies[0],
            'movie_b': movies[1],
        })
        return render(request, 'project4/design1_pairwise.html', ctx)

    ids = ml.random_ranking_set(df, RANKING_SET_SIZE, rng)
    ctx['movies'] = ml.summary_for_ids(df, ids)
    ctx['shown_ids_csv'] = ','.join(str(i) for i in ids)
    return render(request, 'project4/design2_ranking.html', ctx)


def complete(request):
    participant = _current_participant(request)
    ctx = {'participant': participant}
    if participant is not None:
        ctx['n_pairwise'] = participant.pairwise_responses.count()
        ctx['n_rankings'] = participant.ranking_responses.count()
    # Clear session so refreshing complete/ starts a fresh study
    for k in ('p4_participant_id', 'p4_trial_index'):
        if k in request.session:
            del request.session[k]
    return render(request, 'project4/complete.html', ctx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_participant(request):
    pid = request.session.get('p4_participant_id')
    if not pid:
        return None
    try:
        return Participant.objects.get(participant_id=pid)
    except Participant.DoesNotExist:
        return None


def _save_response(request, participant, df):
    if participant.condition == Participant.DESIGN_PAIRWISE:
        form = PairwiseResponseForm(request.POST)
        if form.is_valid():
            PairwiseResponse.objects.create(
                participant=participant,
                trial_index=form.cleaned_data['trial_index'],
                movie_a_id=form.cleaned_data['movie_a_id'],
                movie_b_id=form.cleaned_data['movie_b_id'],
                chosen_movie_id=form.cleaned_data['chosen_movie_id'],
                response_ms=form.cleaned_data.get('response_ms'),
            )
    else:
        form = RankingResponseForm(request.POST)
        if form.is_valid():
            shown  = [int(x) for x in form.cleaned_data['shown_ids_json'].split(',')  if x.strip()]
            ranked = [int(x) for x in form.cleaned_data['ranked_ids_json'].split(',') if x.strip()]
            RankingResponse.objects.create(
                participant=participant,
                trial_index=form.cleaned_data['trial_index'],
                shown_ids_json=shown,
                ranked_ids_json=ranked,
                response_ms=form.cleaned_data.get('response_ms'),
            )
