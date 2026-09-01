"""Generate the Project 4 PDF report.

Usage:
    python manage.py build_report_p4

Writes to static/project4/report.pdf.
"""
import os

from django.conf import settings
from django.core.management.base import BaseCommand

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


OUT_DIR  = os.path.join(settings.BASE_DIR, 'static', 'project4')
OUT_PATH = os.path.join(OUT_DIR, 'report.pdf')


class Command(BaseCommand):
    help = 'Build the Project 4 PDF report at static/project4/report.pdf'

    def handle(self, *args, **options):
        os.makedirs(OUT_DIR, exist_ok=True)

        doc = SimpleDocTemplate(OUT_PATH, pagesize=A4,
                                leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='H1', parent=styles['Heading1'],
                                  spaceBefore=10, spaceAfter=10, fontSize=18,
                                  textColor=colors.HexColor('#1e3a5f')))
        styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'],
                                  spaceBefore=14, spaceAfter=6, fontSize=13,
                                  textColor=colors.HexColor('#2563eb')))
        styles.add(ParagraphStyle(name='H3', parent=styles['Heading3'],
                                  spaceBefore=10, spaceAfter=4, fontSize=11,
                                  textColor=colors.HexColor('#374151')))
        body = styles['BodyText']

        story = []
        p = story.append

        # ---------- Cover ----------
        p(Paragraph('Project 4 — Preference Elicitation for Movie Recommendation', styles['H1']))
        p(Paragraph('<i>Human-Centric AI · TUHH · Summer semester 2026</i>', body))
        p(Spacer(1, 0.4 * cm))
        p(Paragraph(
            'This report describes the design of a user study comparing two '
            'preference-elicitation interfaces on the IMDB 5000 movie '
            'dataset. Task 1 defines a compact feature representation of a '
            'movie; Task 2 extends the Bradley-Terry model to full rankings '
            'via Plackett-Luce; Task 3 specifies the experimental protocol '
            'that would be used to run the study; Task 4 is the Django '
            'interface that a participant would use.', body))
        p(Spacer(1, 0.3 * cm))

        # ---------- Task 1 ----------
        p(Paragraph('Task 1 — Feature representation', styles['H2']))
        p(Paragraph(
            'A movie is represented by a low-dimensional numerical vector '
            '<b>x ∈ ℝ<sup>d</sup></b> so that the user utility '
            '<b>U(x) = w<sup>T</sup>x</b> is well-defined and every '
            'coefficient of <b>w</b> has an interpretable meaning. '
            'The chosen features are:', body))
        rows = [
            ['Group', 'Feature', 'Rationale'],
            ['Quality',  'imdb_score',   'Coarse quality signal used by the public.'],
            ['Physical', 'duration_z',   'Length preference (short vs long movies).'],
            ['Era',      'year_z',       'Preference for old vs recent films.'],
            ['Economics','log_budget_z, log_gross_z',
                                         'Blockbuster vs indie separation.'],
            ['Content',  '14 genre indicators (multi-hot)',
                                         'The core content-based signal for a '
                                         'movie recommender; each dimension is '
                                         'a yes/no on a well-known genre.'],
            ['Culture',  'is_english',   'English vs foreign-language production.'],
            ['Audience', '4 content-rating one-hots (G / PG / PG-13 / R)',
                                         'Target audience (family vs adult).'],
        ]
        t = Table(rows, colWidths=[3 * cm, 5 * cm, 8.5 * cm], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eff6ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d1d5db')),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        p(t)
        p(Spacer(1, 0.3 * cm))
        p(Paragraph(
            'The resulting vector has d ≈ 24 dimensions — small enough '
            'that a handful of pairwise comparisons or one or two rankings '
            'of ten movies contain useful information about <b>w</b>. '
            'We deliberately avoid director/actor identity features '
            '(hundreds of near-zero dimensions with sparse support) and '
            'text tag features (opaque to the participant).', body))
        p(Paragraph(
            'All numerical features are z-scored so that no single '
            'feature dominates the utility because of scale. Missing '
            'values are replaced by 0 (i.e. the mean, after '
            'standardisation).', body))

        # ---------- Task 2 ----------
        p(PageBreak())
        p(Paragraph('Task 2 — Ranking model (Plackett-Luce extension of Bradley-Terry)', styles['H2']))
        p(Paragraph(
            'The Bradley-Terry model defines the probability that item <i>i</i> '
            'is preferred to item <i>j</i> as', body))
        p(Paragraph(
            '<b>P(i ≻ j) = exp(U(i)) / (exp(U(i)) + exp(U(j)))</b>.', body))
        p(Paragraph(
            'Design 2 collects full orderings i<sub>1</sub> ≻ i<sub>2</sub> ≻ … ≻ '
            'i<sub>n</sub> of ten movies, so we need to extend the '
            'model. The natural extension is the <b>Plackett-Luce</b> '
            'model:', body))
        p(Paragraph(
            '<b>P(i<sub>1</sub> ≻ … ≻ i<sub>n</sub>) = ∏<sub>k=1</sub><sup>n</sup> '
            'exp(U(i<sub>k</sub>)) / Σ<sub>j=k</sub><sup>n</sup> '
            'exp(U(i<sub>j</sub>))</b>.', body))
        p(Paragraph(
            'Intuition: at each step k, the top-ranked remaining item is '
            'selected via a softmax over the utilities of the surviving '
            'items. When n = 2 the product collapses to a single term and '
            'we recover exactly the Bradley-Terry probability — so '
            'Plackett-Luce is a genuine extension of it.', body))
        p(Paragraph(
            'Given a set of observed rankings, the parameters <b>w</b> of '
            'the utility function are estimated by <b>maximum likelihood</b>. '
            'The log-likelihood is the sum, across trials, of the '
            'per-step log-softmax terms above; the gradient with respect '
            'to <b>w</b> is', body))
        p(Paragraph(
            '<b>∂ℓ/∂w = Σ<sub>k</sub> [ x<sub>i<sub>k</sub></sub> − '
            '(Σ<sub>j=k</sub><sup>n</sup> p<sub>jk</sub> · x<sub>j</sub>) ]</b>', body))
        p(Paragraph(
            'where <b>p<sub>jk</sub></b> is the softmax weight of item j '
            'at step k. In the code (<code>ml.py</code>) this is '
            'implemented as a small gradient-ascent MLE with an '
            'isotropic Gaussian prior (L2 regulariser) — sufficient for '
            'the online, low-data setting of preference elicitation.', body))
        p(Paragraph(
            'Pairwise responses from Design 1 are trivially rankings of '
            'size 2, so the same estimator handles both interfaces — '
            'making the two datasets directly comparable in the analysis.', body))

        # ---------- Task 3 ----------
        p(PageBreak())
        p(Paragraph('Task 3 — User-study design', styles['H2']))
        p(Paragraph('Research question', styles['H3']))
        p(Paragraph(
            '<i>Do participants elicit their preferences more accurately, '
            'per unit of elapsed time, using pairwise choices (Design 1) '
            'or by ranking a set of ten movies (Design 2)?</i>', body))
        p(Paragraph('Hypotheses', styles['H3']))
        p(Paragraph(
            '<b>H0.</b> The two interfaces yield preference vectors of '
            'equal predictive accuracy on a held-out set of pairwise '
            'comparisons.', body))
        p(Paragraph(
            '<b>H1.</b> Design 2 yields higher accuracy per unit elapsed '
            'time, because ten-way rankings contain more information per '
            'response than a single pairwise choice.', body))
        p(Paragraph(
            '<b>H2.</b> Design 2 has higher subjective load per response '
            'than Design 1, so the two interfaces trade off ergonomics '
            'against information density.', body))

        p(Paragraph('Design', styles['H3']))
        p(Paragraph(
            '<b>Between-subjects</b> design with two conditions '
            '(Design 1 · pairwise, Design 2 · rank-10). Between-subjects '
            'avoids carry-over of learned preferences across interfaces '
            'and matches how a real recommender would be deployed (a new '
            'user is exposed to one interface at a time).', body))
        p(Paragraph(
            'Assignment to condition is random, balanced by the app '
            '(<code>_assign_condition</code> in <code>views.py</code> '
            'picks the currently under-represented condition, breaking '
            'ties uniformly at random).', body))

        p(Paragraph('Participants and recruitment', styles['H3']))
        p(Paragraph(
            'Target sample size: <b>N = 60</b> (30 per condition), '
            'sufficient for detecting a medium-sized effect '
            '(Cohen\'s d ≈ 0.65) with power 0.80 at α = 0.05 in an '
            'unpaired t-test.', body))
        p(Paragraph(
            'Recruitment via TUHH student mailing lists and course '
            'noticeboards. Eligibility: adult participants who watch at '
            'least one movie per month (self-report). No compensation '
            'beyond a debrief at the end. Ethics: this is a low-risk '
            'anonymous online study; informed consent is collected on '
            'the consent page.', body))

        p(Paragraph('Independent variable', styles['H3']))
        p(Paragraph(
            'Interface condition (Design 1 vs Design 2).', body))

        p(Paragraph('Dependent variables', styles['H3']))
        p(Paragraph(
            '<b>Primary.</b> Accuracy of the estimated preference vector '
            '<b>ŵ</b> on a held-out set of 20 pairwise comparisons drawn '
            'from a separate slice of the IMDB dataset. For each held-out '
            'pair (i, j), the prediction is the item with higher '
            '<b>ŵ<sup>T</sup>x</b>; accuracy is the fraction of correct '
            'predictions.', body))
        p(Paragraph(
            '<b>Secondary.</b> Mean time per response (ms), reported to '
            'the server via <code>performance.now()</code>. Time-per-'
            'response is a proxy for cognitive load.', body))
        p(Paragraph(
            '<b>Subjective.</b> Optional single-item load rating '
            '(1-7 Likert, "How mentally demanding was the task?") at the '
            'end of the session.', body))

        p(Paragraph('Procedure', styles['H3']))
        p(Paragraph(
            '1. Landing page — read description and download report if '
            'interested.<br/>'
            '2. Consent page — read information, tick consent box, '
            'optionally fill age group and prior ML/AI experience.<br/>'
            '3. Random condition assignment.<br/>'
            '4. Elicitation phase — the assigned interface, repeated '
            'for a fixed number of trials (20 pairwise choices, or 5 '
            'rankings of ten). Progress bar shown at the top.<br/>'
            '5. Debrief — thank-you page showing the participant\'s '
            'condition and response count.', body))

        p(Paragraph('Analysis plan', styles['H3']))
        p(Paragraph(
            'Fit <b>ŵ</b> per participant using the Plackett-Luce MLE '
            'described in Task 2 on the responses collected in that '
            'condition. Compute accuracy on the held-out pairs. Compare '
            'the two condition means with an independent-samples '
            't-test (Welch\'s if variances differ), report Cohen\'s d '
            'and a 95% confidence interval on the difference. If the '
            'accuracy distributions are strongly non-normal, fall back '
            'to a Mann-Whitney U test.', body))
        p(Paragraph(
            'Secondary analysis on time-per-response and subjective '
            'load with the same test. Multiple comparisons are corrected '
            'with Holm-Bonferroni across the three DVs.', body))

        p(Paragraph('Threats to validity and mitigations', styles['H3']))
        p(Paragraph(
            '<b>Learning effects.</b> Both conditions have a short '
            'practice-free start; the between-subjects design prevents '
            'transfer.<br/>'
            '<b>Fatigue.</b> Trial counts are set so total time stays '
            'well under 10 minutes.<br/>'
            '<b>Movie familiarity heterogeneity.</b> Random sampling '
            'from the IMDB 5000 yields a mix of well-known and obscure '
            'movies; participants may prefer the movies they recognise '
            'regardless of features. Reported as a limitation.', body))

        # ---------- Task 4 ----------
        p(PageBreak())
        p(Paragraph('Task 4 — Interface implementation', styles['H2']))
        p(Paragraph(
            'The user-facing interface is a small Django app '
            '(<code>project4/</code>) with four screens:', body))
        p(Paragraph(
            '1. <b>Landing</b> (/project4/) — brief description, PDF '
            'download button, "Start the user study" link.<br/>'
            '2. <b>Consent</b> (/project4/consent/) — informed-consent '
            'text and checkbox, optional demographic categories.<br/>'
            '3. <b>Study</b> (/project4/study/) — either the pairwise '
            'or the ranking screen depending on assigned condition. '
            'Progress bar at top, response time captured via '
            '<code>performance.now()</code>.<br/>'
            '4. <b>Complete</b> (/project4/complete/) — thank-you page '
            'and per-participant response count.', body))
        p(Paragraph(
            'Data is persisted in three Django models: <b>Participant</b>, '
            '<b>PairwiseResponse</b>, and <b>RankingResponse</b>. Every '
            'response is stored anonymously (only a random '
            '<code>participant_id</code> UUID, no personal identifier). '
            'The researcher inspects the data via <code>/admin</code>.', body))
        p(Paragraph(
            'The ranking UI uses up/down buttons on each row (vanilla '
            'JavaScript, no external library) so it works with keyboard '
            'and on mobile. Item numbering re-renders automatically as '
            'items move.', body))
        p(Paragraph(
            'Code is split into <code>ml.py</code> (data + features + '
            'Plackett-Luce), <code>models.py</code>, <code>forms.py</code>, '
            '<code>urls.py</code>, <code>views.py</code> (thin '
            'controllers), <code>admin.py</code> and <code>tests.py</code> '
            '(pure-logic tests).', body))

        # ---------- Instructions to reproduce ----------
        p(Paragraph('Reproducing this document', styles['H3']))
        p(Paragraph(
            'This report is generated by <code>python manage.py '
            'build_report_p4</code>, which uses ReportLab. The '
            'management command lives in '
            '<code>project4/management/commands/build_report.py</code>.', body))

        doc.build(story)
        self.stdout.write(self.style.SUCCESS(
            f'Report written to {OUT_PATH}'))
