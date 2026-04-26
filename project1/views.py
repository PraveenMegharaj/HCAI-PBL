import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.conf import settings

def save_plot(fig, filename):
    """Save a matplotlib figure to media folder and return its URL."""
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return settings.MEDIA_URL + filename

def detect_problem_type(df):
    """Auto detect if classification or regression based on target column."""
    target = df.iloc[:, -1]
    if target.dtype == 'object' or target.nunique() <= 10:
        return 'classification'
    return 'regression'

def index(request):
    context = {}

    if request.method == 'POST':
        # ---- Handle file upload ----
        csv_file = request.FILES.get('csv_file')
        problem_type_override = request.POST.get('problem_type', 'auto')

        if csv_file and csv_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(csv_file)

                # Drop ID column if present
                if df.columns[0].lower() in ['id', 'index']:
                    df = df.drop(columns=df.columns[0])

                # Basic info
                feature_cols = df.columns[:-1].tolist()
                target_col = df.columns[-1]
                context['feature_cols'] = feature_cols
                context['target_col'] = target_col
                context['n_samples'] = len(df)
                context['n_features'] = len(feature_cols)

                # Detect problem type
                if problem_type_override == 'auto':
                    problem_type = detect_problem_type(df)
                else:
                    problem_type = problem_type_override
                context['problem_type'] = problem_type

                # Table preview (first 10 rows)
                context['table_html'] = df.head(10).to_html(
                    classes='data-table', index=False
                )

                # Statistics
                context['stats_html'] = df.describe().round(2).to_html(
                    classes='data-table'
                )

                # Unique ID for this session's plots
                uid = str(uuid.uuid4())[:8]

                # ---- Plot 1: Histograms ----
                n_cols = len(feature_cols)
                fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
                if n_cols == 1:
                    axes = [axes]
                for ax, col in zip(axes, feature_cols):
                    df[col].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
                    ax.set_title(col, fontsize=10)
                    ax.set_xlabel('')
                fig.suptitle('Feature Distributions', fontsize=13)
                fig.tight_layout()
                context['hist_url'] = save_plot(fig, f'hist_{uid}.png')

                # ---- Plot 2: Scatter plot (first 2 features) ----
                fig, ax = plt.subplots(figsize=(6, 5))
                x_col = feature_cols[0]
                y_col = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]

                if problem_type == 'classification':
                    classes = df[target_col].unique()
                    colors = plt.cm.Set1.colors
                    for i, cls in enumerate(classes):
                        subset = df[df[target_col] == cls]
                        ax.scatter(
                            subset[x_col], subset[y_col],
                            label=str(cls),
                            color=colors[i % len(colors)],
                            alpha=0.7
                        )
                    ax.legend(title=target_col)
                else:
                    scatter = ax.scatter(
                        df[x_col], df[y_col],
                        c=df[target_col], cmap='viridis', alpha=0.7
                    )
                    plt.colorbar(scatter, ax=ax, label=target_col)

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{x_col} vs {y_col}')
                context['scatter_url'] = save_plot(fig, f'scatter_{uid}.png')

                # ---- Plot 3: Correlation heatmap ----
                fig, ax = plt.subplots(figsize=(7, 5))
                numeric_df = df.select_dtypes(include='number')
                sns.heatmap(
                    numeric_df.corr().round(2),
                    annot=True, cmap='coolwarm',
                    ax=ax, linewidths=0.5
                )
                ax.set_title('Correlation Heatmap')
                context['heatmap_url'] = save_plot(fig, f'heatmap_{uid}.png')

                context['success'] = True

            except Exception as e:
                context['error'] = f"Error reading file: {str(e)}"
        else:
            context['error'] = "Please upload a valid CSV file."

    return render(request, "project1/index.html", context)