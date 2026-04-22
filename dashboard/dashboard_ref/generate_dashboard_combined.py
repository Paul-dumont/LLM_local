#!/usr/bin/env python3
"""Generate a single combined PNG containing four plots:
- pain_onset_date (years ago)
- sleep_disorder_type (categorical)
- hearing_loss_present (boolean)
- headache_intensity (numeric)

Saves `selected_dashboard/combined_dashboard.png`.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import helpers from generate_plots_static
try:
    from dashboard.dashboard_ref.generate_plots_static import (
        extract_years_ago_from_onset,
        normalize_categorical_values,
        to_boolean_label,
        extract_numeric,
        calculate_histogram_bins,
        sanitize_filename,
        FONT_TITLE,
        FONT_LABEL,
        FONT_ANNOT,
        FONT_TICKS,
    )
except Exception as e:
    raise RuntimeError(f"Could not import helpers: {e}")

# Local title size (slightly larger than imported FONT_TITLE)
TITLE_SIZE = int(FONT_TITLE * 1.15)

# Visual settings: centralised variables to control plot colors and pie padding
# - `BAR_COLOR`: color used for bar/hist plots
# - `PALETTE_NAME`: seaborn palette name used for categorical/pie charts
# - `PIE_EXPLODE`: small offset to add spacing between pie slices
BAR_COLOR = '#2b8cbe'
PALETTE_NAME = 'viridis'
PIE_EXPLODE = 0.02


def make_combined_dashboard(csv_path: Path, out_dir: Path, out_name: str = 'combined_dashboard.png'):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, dtype=str)

    # Prepare data
    # 1) pain_onset_date -> years_ago only
    years_ago = []
    if 'pain_onset_date' in df.columns:
        patient_age_series = df['patient_age'] if 'patient_age' in df.columns else pd.Series([None] * len(df))
        years_ago, _ = extract_years_ago_from_onset(df['pain_onset_date'], patient_age_series)

    # 2) sleep_disorder_type -> categorical counts (top 8)
    sleep_counts = None
    if 'sleep_disorder_type' in df.columns:
        exploded = normalize_categorical_values(df['sleep_disorder_type'])
        counts_all = exploded.value_counts()
        sleep_counts = counts_all[~counts_all.index.str.lower().isin({'', 'unknown'})].head(8)

    # 3) hearing_loss_present -> boolean counts
    hearing_counts = None
    if 'hearing_loss_present' in df.columns:
        mapped = df['hearing_loss_present'].map(to_boolean_label)
        hearing_counts_all = mapped.value_counts()
        hearing_counts = hearing_counts_all[~hearing_counts_all.index.map(lambda x: str(x).lower() in {'', 'unknown'})]

    # 4) headache_intensity -> numeric series
    headache_num = None
    if 'headache_intensity' in df.columns:
        headache_num = extract_numeric(df['headache_intensity']).dropna()

    # Create figure: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_style('whitegrid')

    def humanize(col_name: str) -> str:
        """Convert snake_case column name to human friendly form.

        Example: 'pain_onset_date' -> 'Pain onset date'
        """
        if not col_name:
            return ''
        s = str(col_name).replace('_', ' ').strip().lower()
        # Title case each word: 'pain_onset_date' -> 'Pain Onset Date'
        return s.title()

    # Top-left: pain_onset_date years ago (histogram)
    ax = axes[0,0]
    if years_ago:
        bins = np.arange(0, max(years_ago) + 5, 5)
        counts, bin_edges = np.histogram(years_ago, bins=bins)
        widths = np.diff(bin_edges)
        centers = bin_edges[:-1] + widths/2
        ax.bar(centers, counts, width=widths, align='center', color=BAR_COLOR, edgecolor='white')
        # Add yellow bar for mean
        mean_years_ago = np.mean(years_ago)
        ax.axvline(mean_years_ago, color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {mean_years_ago:.1f}')
        ax.legend(fontsize=FONT_ANNOT)
        ax.set_xticks(bin_edges)
        ax.set_xlabel('Years ago', fontsize=FONT_LABEL)
        ax.set_ylabel('Count', fontsize=FONT_LABEL)
        ax.set_title(f"{humanize('pain_onset_date')} 141/500", fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=FONT_TICKS)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FONT_ANNOT)
        ax.axis('off')
        ax.set_title(f"{humanize('pain_onset_date')} 141/500", fontsize=TITLE_SIZE)

    # Top-right: sleep_disorder_type (horizontal bar)
    ax = axes[0,1]
    if sleep_counts is not None and not sleep_counts.empty:
        names = sleep_counts.index[::-1]
        values = sleep_counts.values[::-1]
        colors = sns.color_palette(PALETTE_NAME, n_colors=len(values))[::-1]
        ax.barh(names, values, color=colors, edgecolor='white')
        ax.set_xlabel('Count', fontsize=FONT_LABEL)
        ax.set_title(f"{humanize('sleep_disorder_type')} 407/500", fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=FONT_TICKS)
        # annotate
        maxw = max(values) if len(values)>0 else 0
        # Ensure x-axis shows up to 180 (or a bit more if values exceed 180)
        ax.set_xlim(0, max(180, maxw * 1.12))
        # annotate counts to the right of the bars
        for i, v in enumerate(values):
            xpos = min(v + max(0.5, maxw*0.01), ax.get_xlim()[1] - 1)
            ax.text(xpos, i, str(int(v)), va='center', fontsize=FONT_ANNOT)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FONT_ANNOT)
        ax.axis('off')
        ax.set_title(f"{humanize('sleep_disorder_type')} 407/500", fontsize=TITLE_SIZE)

    # Bottom-left: hearing_loss_present (pie)
    ax = axes[1,0]
    if hearing_counts is not None and not hearing_counts.empty:
        labels = [f"{str(l)} ({int(c)})" for l,c in zip(hearing_counts.index, hearing_counts.values)]
        palette = sns.color_palette(PALETTE_NAME, n_colors=max(2, len(hearing_counts)))[::-1]
        # add small separation between slices for better visual padding (configurable via PIE_EXPLODE)
        explode = [PIE_EXPLODE] * len(hearing_counts)
        wedgeprops = {'edgecolor': 'white', 'linewidth': 1.0}
        wedges, texts, autotexts = ax.pie(hearing_counts.values, labels=labels, autopct='%1.1f%%',
                                         colors=palette, startangle=90, counterclock=False,
                                         explode=explode, wedgeprops=wedgeprops)
        for t in (texts or []) + (autotexts or []):
            try:
                t.set_fontsize(FONT_ANNOT)
            except Exception:
                pass
        ax.axis('equal')
        ax.set_title(f"{humanize('hearing_loss_present')} 268/500", fontsize=TITLE_SIZE)
    else:

        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FONT_ANNOT)
        ax.axis('off')
        ax.set_title(humanize('hearing_loss_present'), fontsize=TITLE_SIZE)

    # Bottom-right: headache_intensity (numeric histogram)
    ax = axes[1,1]
    if headache_num is not None and not headache_num.empty:
        # Create one bar per integer value: bins covering each integer (i..i+1)
        try:
            min_v = int(np.floor(headache_num.min()))
            max_v = int(np.ceil(headache_num.max()))
        except Exception:
            # Fallback to default histogram behavior
            counts, bin_edges = np.histogram(headache_num, bins='auto')
            widths = np.diff(bin_edges)
            centers = bin_edges[:-1] + widths / 2
            ax.bar(centers, counts, width=widths, align='center', color=BAR_COLOR, edgecolor='white')
            # place tick labels under bar centers (rounded)
            ax.set_xticks(centers)
            ax.set_xticklabels([f"{c:.1f}" for c in centers])
            # Add small x-margin so rightmost tick isn't clipped (reduced)
            cur_xlim = ax.get_xlim()
            ax.set_xlim(cur_xlim[0], cur_xlim[1] + max(0.5, 0.08 * (cur_xlim[1] - cur_xlim[0])))
            # Add yellow bar for mean
            mean_headache = np.mean(headache_num)
            ax.axvline(mean_headache, color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {mean_headache:.1f}')
            ax.legend(fontsize=FONT_ANNOT)
        else:
            # bins from min_v to max_v inclusive -> edges min_v .. max_v+1
            bins = np.arange(min_v, max_v + 2)
            counts, _ = np.histogram(headache_num, bins=bins)
            centers = np.arange(min_v, max_v + 1) + 0.5
            ax.bar(centers, counts, width=1.0, align='center', color=BAR_COLOR, edgecolor='white')
            # place tick labels under bar centers and label with integer values
            ax.set_xticks(centers)
            ax.set_xticklabels([str(i) for i in range(min_v, max_v + 1)])
            # Add extra right padding so the last tick (e.g. 10) isn't cut off
            # Reduced padding for a tighter layout
            ax.set_xlim(min_v - 0.5, max_v + 1.2)

        # Add yellow bar for mean
        mean_headache = np.mean(headache_num)
        ax.axvline(mean_headache, color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {mean_headache:.1f}')
        ax.legend(fontsize=FONT_ANNOT)
        ax.set_xlabel('Intensity', fontsize=FONT_LABEL)
        ax.set_ylabel('Count', fontsize=FONT_LABEL)
        ax.set_title(f"{humanize('headache_intensity')} 429/500", fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=FONT_TICKS)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FONT_ANNOT)
        ax.axis('off')
        ax.set_title(humanize('headache_intensity'), fontsize=TITLE_SIZE)

    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Combined dashboard image saved to: {out_path}')
    return out_path


if __name__ == '__main__':
    out_dir = HERE / 'selected_dashboard_model'
    csv_path = HERE / 'patient_data_model.csv'
    if not csv_path.exists():
        print(f'Error: {csv_path} not found')
        raise SystemExit(1)
    make_combined_dashboard(csv_path, out_dir)
