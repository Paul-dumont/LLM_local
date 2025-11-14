#!/usr/bin/env python3
"""Génère des PNG pour chaque indicateur dans `patient_data.csv`.

Usage: python generate_plots_static.py

Sortie: dossier `static_plots/` contenant un PNG par colonne.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tokens to treat as missing/unknown (lowercase)
UNKNOWN_TOKENS = {"", "unknown", "unknow", "na", "n/a", "none", "missing"}


def is_unknown(value) -> bool:
    """Check if a value should be treated as unknown/missing."""
    if pd.isna(value):
        return True
    return str(value).strip().lower() in UNKNOWN_TOKENS


def count_present_values(series: pd.Series) -> int:
    """Count the number of non-unknown values in a series."""
    return series.map(lambda x: not is_unknown(x)).sum()


def to_numeric_value(x):
    """Convert a single value to numeric, returning NaN for unknown/invalid values."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.lower() in UNKNOWN_TOKENS:
        return np.nan
    m = re.search(r"(-?\d+(?:[\.,]\d+)?)", s)
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(',', '.'))
    except:
        return np.nan


def extract_numeric(series: pd.Series) -> pd.Series:
    """Extract numeric values from strings like '50mm', '5.2', 'unknown' -> NaN."""
    return series.map(to_numeric_value)


def to_boolean_value(x):
    """Convert a value to boolean, returning NaN for non-boolean values."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ('true', 'yes', 'y', '1'):
        return True
    if s in ('false', 'no', 'n', '0'):
        return False
    return np.nan


def detect_type(series: pd.Series) -> str:
    """Detect the type of data in a series: numeric, boolean, or categorical."""
    total = len(series)
    if total == 0:
        return 'categorical'

    # Check for numeric data
    num = extract_numeric(series)
    num_count = num.notna().sum()
    if num_count / total >= 0.2:
        return 'numeric'

    # Check for boolean data
    bool_count = series.map(to_boolean_value).notna().sum()
    if bool_count / total >= 0.2:
        return 'boolean'

    return 'categorical'


def calculate_histogram_bins(num: pd.Series, present: int) -> tuple:
    """Calculate optimal bin edges using Freedman-Diaconis rule."""
    q25, q75 = np.percentile(num, [25, 75])
    iqr = q75 - q25
    
    if iqr == 0 or present <= 1:
        nbins = int(np.sqrt(present)) if present > 0 else 6
    else:
        bin_width = 2 * iqr * (present ** (-1 / 3))
        if bin_width <= 0:
            nbins = int(np.sqrt(present))
        else:
            data_range = num.max() - num.min()
            nbins = max(6, int(np.ceil(data_range / bin_width))) if data_range > 0 else 6
    
    return np.histogram(num, bins=nbins)


def plot_histogram_bars(ax, counts, bin_edges, color='#2b8cbe'):
    """Plot histogram bars from precomputed counts and bin edges."""
    widths = np.diff(bin_edges)
    centers = bin_edges[:-1] + widths / 2
    ax.bar(centers, counts, width=widths, align='center', 
           color=color, edgecolor='white', alpha=0.95)
    return widths, centers


def setup_numeric_plot_axes(ax, bin_edges, title: str, mean_value: float, 
                           present: int, total: int):
    """Configure axes, labels, and legends for numeric plots."""
    max_ticks = 10
    if len(bin_edges) <= max_ticks:
        ticks = bin_edges
    else:
        idx = np.linspace(0, len(bin_edges) - 1, max_ticks, dtype=int)
        ticks = bin_edges[idx]
    
    ax.set_xticks(ticks)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(mean_value, color='orange', linestyle='--', 
               lw=1.2, label=f'mean {mean_value:.2f}')
    ax.set_xlabel('value')
    ax.set_ylabel('count')
    ax.set_title(f"{title} — {present}/{total} patients", fontsize=14)
    ax.legend(frameon=False)


def is_max_opening(title: str) -> bool:
    """Detect if a column title refers to maximum opening (with or without pain).

    This matches titles containing both 'maximum' (or 'max') and 'opening', case-insensitive.
    """
    t = str(title).strip().lower()
    return ("maximum" in t or "max" in t) and "opening" in t


def plot_numeric(series: pd.Series, out_path: Path, title: str):
    """Create histogram plot for numeric data with adaptive binning."""
    total = len(series)
    present = count_present_values(series)
    
    num = extract_numeric(series).dropna()
    if num.empty:
        _plot_empty_data(out_path, title, total, 'no numeric data')
        return

    vmin, vmax = num.min(), num.max()
    use_fixed_0_10 = (vmin >= 0) and (vmax <= 10)
    use_fixed_0_5 = (vmin >= 0) and (vmax <= 5)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Special bins for patient_age: 5-year bins from 0 to 70
    if title.lower() == 'patient_age':
        bin_edges = np.arange(0, 75, 5)
        counts, _ = np.histogram(num, bins=bin_edges)
        plot_histogram_bars(ax, counts, bin_edges)
        ax.set_xticks(bin_edges)
        ax.set_xlim(0, 70)
        ax.grid(axis='y', alpha=0.3)
    # Special bins for maximum opening (and maximum opening without pain): 5-year bins 0..70
    elif is_max_opening(title):
        bin_edges = np.arange(0, 75, 5)
        counts, _ = np.histogram(num, bins=bin_edges)
        plot_histogram_bars(ax, counts, bin_edges)
        ax.set_xticks(bin_edges)
        ax.set_xlim(0, 70)
        ax.grid(axis='y', alpha=0.3)
    # Special bins for muscle pain score: 5 bins from 0 to 5 (0-1, 1-2, 2-3, 3-4, 4-5)
    elif 'muscle' in title.lower() and 'pain' in title.lower() and 'score' in title.lower():
        bin_edges = np.linspace(0, 5, 6)  # 5 bins
        counts, _ = np.histogram(num, bins=bin_edges)
        plot_histogram_bars(ax, counts, bin_edges)
        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_xlim(0, 5)
        ax.grid(axis='y', alpha=0.3)
    # If values lie in [0,5], use fixed 5 bins (0..5) with tick every 1
    elif use_fixed_0_5:
        bin_edges = np.linspace(0, 5, 6)
        counts, _ = np.histogram(num, bins=bin_edges)
        plot_histogram_bars(ax, counts, bin_edges)
        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_xlim(0, 5)
        ax.grid(axis='y', alpha=0.3)
    # If values lie in [0,10], use fixed 10 bins (0..10) with tick every 1
    elif use_fixed_0_10:
        bin_edges = np.linspace(0, 10, 11)
        counts, _ = np.histogram(num, bins=bin_edges)
        plot_histogram_bars(ax, counts, bin_edges)
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_xlim(0, 10)
        ax.grid(axis='y', alpha=0.3)
    else:
        # Use Freedman-Diaconis rule for adaptive binning
        counts, bin_edges = calculate_histogram_bins(num, present)
        plot_histogram_bars(ax, counts, bin_edges)
        setup_numeric_plot_axes(ax, bin_edges, title, num.mean(), present, total)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    # For special cases (age, 0-10 range), add mean line and finalize
    mean = num.mean()
    ax.axvline(mean, color='orange', linestyle='--', 
               lw=1.2, label=f'mean {mean:.2f}')
    ax.set_xlabel('value')
    ax.set_ylabel('count')
    ax.set_title(f"{title} — {present}/{total} patients", fontsize=14)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_empty_data(out_path: Path, title: str, total: int, message: str):
    """Create an empty plot with a message when no data is available."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center')
    ax.axis('off')
    fig.suptitle(f"{title} — 0/{total} patients", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def to_boolean_label(x):
    """Convert a value to a boolean label string."""
    if is_unknown(x):
        return 'unknown'
    s = str(x).strip().lower()
    if s in ('true', 'yes', 'y', '1'):
        return 'True'
    if s in ('false', 'no', 'n', '0'):
        return 'False'
    return 'unknown'


def plot_boolean(series: pd.Series, out_path: Path, title: str):
    """Create pie chart for boolean data."""
    total = len(series)
    present = count_present_values(series)
    
    mapped = series.map(to_boolean_label)
    counts_all = mapped.value_counts()
    counts = counts_all[~counts_all.index.str.lower().isin(UNKNOWN_TOKENS)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if counts.empty:
        _plot_empty_data(out_path, title, total, 'no boolean data')
        return
    
    palette = sns.color_palette('viridis', n_colors=max(2, len(counts)))
    labels = [f"{lab} ({int(counts[lab])})" for lab in counts.index]
    ax.pie(
        counts.values,
        labels=labels,
        autopct='%1.1f%%',
        colors=palette,
        startangle=90,
        counterclock=False,
        wedgeprops={'edgecolor': 'white'}
    )
    ax.axis('equal')
    ax.set_title(f"{title} — {present}/{total} patients", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def normalize_categorical_values(series: pd.Series) -> pd.Series:
    """Normalize categorical values by splitting on '|' and cleaning."""
    s = series.dropna().astype(str)
    exploded = s.str.split(r'\s*\|\s*').explode().str.strip()
    exploded = exploded.where(~exploded.str.lower().isin(UNKNOWN_TOKENS), 'unknown')
    exploded = exploded.replace({'': 'unknown'})
    exploded = exploded.str.capitalize()
    return exploded


def annotate_bars(ax, maxw: float):
    """Add count labels to the end of horizontal bars."""
    bars = ax.patches
    if len(bars) == 0:
        return
    
    cur_xlim = ax.get_xlim()
    right_needed = maxw * 1.12 + 1
    if cur_xlim[1] < right_needed:
        ax.set_xlim(cur_xlim[0], right_needed)
    
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(w + max(0.5, maxw * 0.01), y, f"{int(round(w))}", 
                va='center', ha='left', fontsize=9)


def plot_categorical(series: pd.Series, out_path: Path, title: str, top_n: int = 30):
    """Create horizontal bar chart for categorical data."""
    total = len(series)
    present = count_present_values(series)
    
    exploded = normalize_categorical_values(series)
    counts_all = exploded.value_counts()
    counts = counts_all[~counts_all.index.str.lower().isin(UNKNOWN_TOKENS)].head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette('viridis', n_colors=len(counts))
    
    # Plot in reversed order so most common category appears at top
    ax.barh(counts.index[::-1], counts.values[::-1], 
            color=colors[::-1], edgecolor='white')
    ax.set_xlabel('count')
    ax.set_title(f"{title} — {present}/{total} patients", fontsize=14)
    
    # Annotate bars with counts
    try:
        maxw = max([bar.get_width() for bar in ax.patches]) if ax.patches else 0
        if maxw > 0:
            annotate_bars(ax, maxw)
    except Exception:
        pass
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def extract_years_ago_from_onset(series: pd.Series, patient_age_series: pd.Series) -> tuple:
    """Extract years ago and age at onset from pain onset date strings.
    
    Returns:
        tuple: (years_ago_list, age_at_onset_list)
    """
    years_ago = []
    age_at_onset = []

    for idx in series.index:
        onset_val = series.iloc[idx] if idx < len(series) else None
        age_val = patient_age_series.iloc[idx] if idx < len(patient_age_series) else None

        if is_unknown(onset_val):
            continue

        onset_str = str(onset_val).strip()
        match_ago = re.search(r"(\d+)\s*years?\s+ago", onset_str, re.IGNORECASE)

        if match_ago:
            y_ago = int(match_ago.group(1))
            years_ago.append(y_ago)

            # Calculate age at onset if patient age is available
            if not is_unknown(age_val):
                try:
                    current_age = extract_numeric(pd.Series([age_val])).iloc[0]
                    if not pd.isna(current_age):
                        age_at_onset.append(int(round(current_age - y_ago)))
                except Exception:
                    pass

    return years_ago, age_at_onset


def plot_time_histogram(data: list, out_path: Path, title: str, xlabel: str, 
                        total: int, bin_size: int, color):
    """Create a histogram plot for time-based data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    count = len(data)
    
    if count > 0:
        bins = np.arange(0, max(data) + bin_size + 1, bin_size)
        counts, _ = np.histogram(data, bins=bins)
        widths = np.diff(bins)
        centers = bins[:-1] + widths / 2
        ax.bar(centers, counts, width=widths, align='center', 
               color=color, edgecolor='white', alpha=0.95)
        ax.set_xticks(bins)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(f"{title} — {count}/{total} patients", fontsize=14)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} — 0/{total} patients", fontsize=14)
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_pain_onset_dual(series: pd.Series, patient_age_series: pd.Series, 
                        out_path: Path, title: str):
    """Create two separate plots for pain_onset_date:
    - <out_path>_years_ago.png: histogram of years since onset
    - <out_path>_age_at_onset.png: histogram of patient age at onset
    """
    total = len(series)
    present = count_present_values(series)

    years_ago, age_at_onset = extract_years_ago_from_onset(series, patient_age_series)
    colors = sns.color_palette('viridis', n_colors=2)

    # Plot 1: Years ago
    out_path_ago = out_path.with_name(out_path.stem + '_years_ago' + out_path.suffix)
    plot_time_histogram(years_ago, out_path_ago, title, 'Years Ago', 
                       len(years_ago), 5, colors[0])

    # Plot 2: Age at onset
    out_path_age = out_path.with_name(out_path.stem + '_age_at_onset' + out_path.suffix)
    other_non_unknown = max(0, present - len(years_ago))
    plot_time_histogram(age_at_onset, out_path_age, title, 'Age at Onset (years)', 
                       other_non_unknown, 5, colors[1])


def should_skip_column(column_name: str) -> bool:
    """Check if a column should be skipped from plotting."""
    return column_name.lower() == 'patient_id'


def override_column_type(column_name: str, detected_type: str) -> str:
    """Override detected type for specific columns that need special handling."""
    col_l = column_name.lower()
    
    # Force boolean for _present columns
    if col_l.endswith('_present'):
        return 'boolean'
    
    # Specific boolean columns
    if col_l in ('sleep_apnea_diagnosed', 'migraine_history', 'jaw_locking'):
        return 'boolean'
    
    # Force categorical for mixed frequency columns
    if col_l in ('headache_frequency', 'pain_frequency'):
        return 'categorical'
    
    return detected_type


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Create a safe filename from a column name."""
    return re.sub(r"[^0-9A-Za-z_]+", '_', name)[:max_length]


def convert_muscle_pain_to_numeric(series: pd.Series) -> pd.Series:
    """Convert textual muscle pain scores to numeric values (0-5).

    Mapping chosen (symmetric scale):
      - minimal -> 0.0
      - mild    -> 1.0
      - moderate-> 2.5
      - high / hight -> 4.0
      - severe  -> 5.0

    The function supports multi-valued cells separated by '|', ',', or ';'.
    For each row, map all tokens found and return the average (NaN if none mapped).
    """
    mapping = {
        'minimal': 0.0,
        'mild': 1.0,
        'moderate': 2.5,
        'high': 4.0,
        'hight': 4.0,  # tolerate common typo
        'severe': 5.0,
    }

    def row_to_value(x):
        if is_unknown(x):
            return np.nan
        s = str(x).strip()
        # split on common delimiters
        parts = re.split(r"\s*[\|,;]\s*", s)
        vals = []
        for p in parts:
            key = p.strip().lower()
            if key in mapping:
                vals.append(mapping[key])
            else:
                # try to extract numeric directly (in case the dataset already has numbers)
                num = to_numeric_value(p)
                if not pd.isna(num):
                    # assume numbers given on 0..5 scale or 0..10 or 0..100
                    if 0 <= num <= 5:
                        vals.append(float(num))
                    elif 5 < num <= 10:
                        # map 0-10 to 0-5
                        vals.append(float(num) / 2.0)
                    elif 10 < num <= 100:
                        # map 0-100 to 0-5
                        vals.append(float(num) / 20.0)
        if not vals:
            return np.nan
        return float(np.mean(vals))

    return series.map(row_to_value)


def plot_column(df: pd.DataFrame, col: str, out_dir: Path):
    """Plot a single column from the dataframe."""
    col_l = col.strip().lower()
    
    # Special handling for pain_onset_date: dual plot
    if col_l == 'pain_onset_date':
        series = df[col]
        patient_age_series = (df['patient_age'] if 'patient_age' in df.columns 
                             else pd.Series([None] * len(df)))
        safe_name = sanitize_filename(col)
        out_path = out_dir / f"{safe_name}.png"
        plot_pain_onset_dual(series, patient_age_series, out_path, col)
        return
    
    # Standard column plotting
    series = df[col]
    # Special handling for muscle pain score: convert known textual categories to numeric 0-10
    if 'muscle' in col_l and 'pain' in col_l and 'score' in col_l:
        # create numeric series (float) in 0..10 range
        num_series = convert_muscle_pain_to_numeric(series)
        # now treat as numeric and plot
        plot_numeric(num_series, out_dir / f"{sanitize_filename(col)}.png", col)
        return

    detected_type = detect_type(series)
    column_type = override_column_type(col, detected_type)
    
    safe_name = sanitize_filename(col)
    out_path = out_dir / f"{safe_name}.png"
    
    if column_type == 'numeric':
        plot_numeric(series, out_path, col)
    elif column_type == 'boolean':
        plot_boolean(series, out_path, col)
    else:
        plot_categorical(series, out_path, col)


def load_patient_data(csv_path: Path) -> pd.DataFrame:
    """Load patient data from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f'patient_data.csv not found in {csv_path.parent}')
    
    df = pd.read_csv(csv_path, dtype=str)
    print(f'Loaded {df.shape[0]} rows and {df.shape[1]} columns')
    return df


def main():
    """Main function to generate all plots from patient_data.csv."""
    here = Path(__file__).resolve().parent
    csv_path = here / 'patient_data.csv'
    out_dir = here / 'static_plots'
    out_dir.mkdir(exist_ok=True)

    try:
        df = load_patient_data(csv_path)
    except FileNotFoundError as e:
        print(e)
        return

    for col in df.columns:
        if should_skip_column(col):
            continue
        
        print(f'Processing {col}')
        try:
            plot_column(df, col, out_dir)
        except Exception as e:
            print(f'Error plotting {col}: {e}')

    print(f'All plots saved to {out_dir}')


if __name__ == '__main__':
    main()
