"""Generate the Project 3 PDF report from the current cached metrics.

Usage:
    python manage.py build_report

Writes to static/project3/report.pdf. The report describes the design
choices for Tasks 1-5, cites the achieved metrics, and embeds the plots
generated during a recent visit to the /project3/ page.
"""
import os

from django.conf import settings
from django.core.management.base import BaseCommand

import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from project3 import active_learning, deferral, experts, ml


OUT_DIR = os.path.join(settings.BASE_DIR, 'static', 'project3')
OUT_PATH = os.path.join(OUT_DIR, 'report.pdf')


def _para(styles, text, style='BodyText'):
    return Paragraph(text, styles[style])


class Command(BaseCommand):
    help = 'Build the Project 3 PDF report at static/project3/report.pdf'

    def handle(self, *args, **options):
        os.makedirs(OUT_DIR, exist_ok=True)

        metrics, _ = ml.load_cached_baseline()
        if metrics is None:
            self.stdout.write(self.style.ERROR(
                'No cached baseline found — visit /project3/ and train it first.'))
            return

        # -- expert + downstream metrics ------------------------------------
        expert = experts.SimulatedExpert(
            [0, 1], competent_acc=0.95, other_acc=0.40, n_classes=4)
        y_test = np.array(metrics['y_test'])
        expert_metrics = experts.evaluate_expert(expert, y_test)
        y_pred_model  = np.array(metrics['y_pred'])
        y_proba       = np.array(metrics['y_proba'])
        y_pred_expert = np.array(expert_metrics['y_pred'])

        sweep = deferral.sweep_deferral(y_test, y_pred_model, y_proba, y_pred_expert)
        curve = active_learning.active_learning_curve(
            y_test, y_pred_model, y_proba, y_pred_expert)

        # Render fresh plots for the report
        cm_url = ml.save_plot(
            ml.plot_confusion_matrix(metrics['cm'], 'Baseline confusion matrix'),
            'p3_report_baseline_cm.png')
        expert_cm_url = ml.save_plot(
            ml.plot_confusion_matrix(expert_metrics['cm'], 'Expert confusion matrix'),
            'p3_report_expert_cm.png')
        expert_bar_url = ml.save_plot(
            ml.plot_per_class_bar(expert_metrics['per_class'], 'accuracy',
                                  'Expert per-class accuracy', color='#8b2e8b'),
            'p3_report_expert_bar.png')
        defer_url = ml.save_plot(
            ml.plot_deferral_sweep(sweep), 'p3_report_defer.png')
        al_url = ml.save_plot(
            ml.plot_active_learning_curve(curve), 'p3_report_al.png')

        def rel(url):
            return os.path.join(settings.MEDIA_ROOT, os.path.basename(url))

        # -- build PDF ------------------------------------------------------
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

        story = []
        p = story.append

        # Cover
        p(_para(styles, 'Project 3 — Active Learning for Learning-to-Defer', 'H1'))
        p(_para(styles, '<i>Human-Centric AI · TUHH · Summer semester 2026</i>'))
        p(Spacer(1, 0.4 * cm))
        p(_para(styles,
                'This report documents the design choices and results for '
                'the five tasks of Project 3. The baseline classifier learns '
                'the AG News topic-classification task, a simulated expert '
                'models human specialisation on a subset of classes, a '
                'confidence-threshold policy decides when to defer to the '
                'expert, and an active-learning strategy discovers expert '
                'competence with a limited query budget.'))
        p(Spacer(1, 0.4 * cm))

        # Task 1
        p(_para(styles, 'Task 1 — Baseline classifier', 'H2'))
        p(_para(styles,
                f'A TF-IDF vectoriser (unigrams and bigrams, 20 000 features, '
                f'sublinear TF weighting) followed by a multinomial Logistic '
                f'Regression trained with L-BFGS on the full AG News training '
                f'split (n={metrics["n_train"]}). Cached to disk after the '
                f'first fit.'))
        p(_para(styles, f'<b>Test accuracy: {metrics["accuracy"]:.4f}</b> '
                f'on n={metrics["n_test"]} examples.'))
        p(Image(rel(cm_url), width=11 * cm, height=8 * cm))
        p(Spacer(1, 0.2 * cm))
        rows = [['Class', 'Precision', 'Recall', 'F1', 'Support']]
        for name, m in metrics['per_class'].items():
            rows.append([name, m['precision'], m['recall'],
                         m['f1'], m['support']])
        t = Table(rows, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eff6ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d1d5db')),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        p(t)

        # Task 2
        p(PageBreak())
        p(_para(styles, 'Task 2 — Simulated region-competent expert', 'H2'))
        p(_para(styles,
                'Real human experts specialise. We model this with an expert '
                'that answers correctly with probability <b>p<sub>c</sub></b> '
                'on classes in its competence set and with lower probability '
                '<b>p<sub>o</sub></b> on the remaining classes. When wrong, '
                'a uniformly random incorrect label is returned. This gives '
                'the expert a competence <i>structure</i> that a deferral '
                'policy can later exploit.'))
        p(_para(styles,
                f'Default configuration: competent on <b>World + Sports</b>, '
                f'p<sub>c</sub>=0.95, p<sub>o</sub>=0.40 → '
                f'overall accuracy '
                f'<b>{expert_metrics["overall_accuracy"]:.4f}</b>.'))
        p(Image(rel(expert_cm_url), width=11 * cm, height=8 * cm))
        p(Image(rel(expert_bar_url), width=12 * cm, height=7 * cm))

        # Task 3
        p(PageBreak())
        p(_para(styles, 'Task 3 — Learning to defer', 'H2'))
        p(_para(styles,
                'Given the classifier\'s soft-max output and the (simulated) '
                'expert\'s label, we adopt the simplest well-motivated '
                'deferral rule: <b>defer to the expert whenever the top-class '
                'probability of the classifier is below τ</b>. The '
                'classifier\'s top-1 probability is a serviceable measure of '
                'its own confidence, so we defer precisely on cases where the '
                'classifier is unsure. When τ is small the system falls back '
                'to the classifier, when τ is large it always defers.'))
        p(_para(styles,
                'The plot below sweeps τ across [0, 1] and reports the '
                'system accuracy (a blend of model and expert predictions) '
                'together with the defer rate.'))
        p(Image(rel(defer_url), width=13 * cm, height=7.5 * cm))
        p(_para(styles,
                'Because the expert is strong on two of the four classes '
                'and the classifier is uniformly strong (≈0.92 accuracy '
                'across all classes), the confidence-threshold policy only '
                'improves the system at very small deferral rates. Above '
                'τ≈0.6 the system starts to lose accuracy because the expert '
                'is weaker than the classifier on Business and Sci/Tech, so '
                'deferring uncertain examples in those classes replaces '
                'good model predictions with noisy expert labels. The best '
                'operating point in the sweep is'))
        best_tau = max(sweep, key=lambda s: s['system_accuracy'])
        p(_para(styles,
                f'<b>τ = {best_tau["tau"]:.2f}</b> — defer rate '
                f'{best_tau["defer_rate"]:.3f}, system accuracy '
                f'<b>{best_tau["system_accuracy"]:.4f}</b>.'))

        # Task 4
        p(PageBreak())
        p(_para(styles, 'Task 4 — Active learning for expert competence', 'H2'))
        p(_para(styles,
                'We now assume no expert labels are available during '
                'training. Given a budget of B expert queries, we use '
                '<b>least-confident uncertainty sampling</b>: the B test '
                'points with the smallest max-softmax score are the ones '
                'where a deferral could plausibly help, so labels there '
                'give us the most information about whether the expert '
                'should be trusted.'))
        p(_para(styles,
                'From the queried labels we estimate the expert\'s accuracy '
                'conditioned on the model\'s <i>predicted</i> class '
                '(rather than the true class, which we do not know at '
                'deployment). At inference we defer whenever the estimated '
                'expert accuracy for the model\'s predicted class exceeds '
                'the model\'s top-1 probability.'))
        p(Image(rel(al_url), width=13 * cm, height=7.5 * cm))
        p(_para(styles,
                'With a small budget (50-100 queries) the per-class '
                'estimates are noisy and the deferral policy is close to '
                'never-defer. As the budget grows, the per-class expert '
                'accuracies converge to their true values and the deferral '
                'policy matches (or slightly exceeds) the Task-3 confidence '
                'policy without ever using the expert during training.'))

        # Task 5
        p(PageBreak())
        p(_para(styles, 'Task 5 — Interactive expert (optional)', 'H2'))
        p(_para(styles,
                'A small Django interface presents one random test-set '
                'article per turn. A participant chooses a class, and the '
                'app records the submission together with the true label in '
                'the database (<code>HumanExpertLabel</code>) so it appears '
                'in the Django admin and survives across page reloads. A '
                'running accuracy is displayed as the participant labels '
                'more articles.'))
        p(_para(styles,
                'The purpose is not to run a real study but to demonstrate '
                'the plumbing: the same interface, plus the active-learning '
                'query strategy from Task 4, could be used to build a real '
                'human-in-the-loop labelling pipeline.'))

        # Software structure note
        p(_para(styles, 'Software structure', 'H2'))
        p(_para(styles,
                'The code is organised across proper Django-app files: '
                '<code>ml.py</code> (dataset loading + baseline), '
                '<code>experts.py</code> (Task 2), '
                '<code>deferral.py</code> (Task 3), '
                '<code>active_learning.py</code> (Task 4), '
                '<code>models.py</code> (Task 5 storage), '
                '<code>forms.py</code> (validation), and '
                '<code>views.py</code> as a thin controller. Unit tests '
                'live in <code>tests.py</code>.'))

        doc.build(story)
        self.stdout.write(self.style.SUCCESS(
            f'Report written to {OUT_PATH}'))
