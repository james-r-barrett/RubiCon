#!/usr/bin/env python3
"""
Rubisco Condensation Plotting Tool

Outputs:
 1. Combined GMM Density Plot (KDE)
 2. Fraction Condensed (Line Plot)
 3. Convex Hull Concentration (Box Plot)
 4. Combined RDF Analysis (Stats on Minima)
 5. MSD Raincloud Plots (Condensed vs Dilute Diffusivities)

Dependencies:
    - numpy, matplotlib, scipy, sklearn, pandas, ptitprince
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import ptitprince as pt  # pip install ptitprince

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

# Import KDE helper if available
try:
    from analysis_main import fit_density_and_plot
except ImportError:
    fit_density_and_plot = None

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path(
    "/Users/jb1725/Documents/York/PDRA/Manuscripts/"
    "Gaurav Sticker Number/LAMMPs/chlamy"
)

# DEFAULT_OUTPUT_DIR = Path(
#     "/Users/jb1725/Documents/York/PDRA/Manuscripts/"
#     "Gaurav Sticker Number/LAMMPs/chlorella"
# )

# Standard Analysis Parameters
RBM_LIST = [2, 3, 4, 5, 6, 7]
RATIO_LIST = [1, 2, 3, 5]
TARGET_RATIO_FOR_BOXPLOT = 5

# Global Color Palette
RBM_COLORS = {
    2: '#EEEEEF',  # Light Grey
    3: '#FAE5D6',  # Very Light Orange
    4: '#F7D1B7',  # Light Orange
    5: '#F2A773',  # Medium Orange
    7: '#E08E79',  # Darker Orange/Red
    9: '#C14F4F'  # Deep Red
}
FALLBACK_COLOR = '#CCCCCC'

# Filter Lists
HIST_RBM_FILTER = [4, 5, 7]
HIST_RATIO_FILTER = [3, 5]
NEIGHBOUR_THRESH_USED = 4.9

RDF_RBM_FILTER = [3, 4, 5]
RDF_RATIO_FILTER = [5]
RDF_MAX_R = 60.0

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


# -----------------------------------------------------------------------------
# 1. NEIGHBOR COUNT HISTOGRAM
# -----------------------------------------------------------------------------
def plot_neighbor_histogram(output_dir: Path, rbm_filter=HIST_RBM_FILTER, ratio_filter=HIST_RATIO_FILTER):
    logging.info(f"Generating Neighbor Count Histogram...")

    collected_data = []
    for rbm in rbm_filter:
        for ratio in ratio_filter:
            fname = output_dir / f"{rbm}rbm_1_{ratio}_neighbor_counts.npy"
            if fname.exists():
                try:
                    collected_data.append(np.load(fname).flatten())
                except:
                    pass

    if not collected_data:
        logging.warning("No data found for histogram filters.")
        return

    combined = np.concatenate(collected_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    max_val = int(np.max(combined))
    bins = np.arange(-0.5, max_val + 1.5, 1)

    ax.hist(combined, bins=bins, density=True,
            color="#87CEEB", edgecolor="black", alpha=0.7, label="Neighbor Counts")

    ax.axvline(NEIGHBOUR_THRESH_USED, color="red", linestyle="--", linewidth=2,
               label=f"Threshold ({NEIGHBOUR_THRESH_USED})")

    ax.set_xlabel("Number of Neighbors (within 20nm)", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')
    ax.set_title(f"Neighbor Count Distribution\n(RBMs: {rbm_filter}, Ratios: {ratio_filter})")
    ax.legend()
    #ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_xlim(-0.5, max_val + 0.5)

    # Despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "neighbor_count_histogram.pdf", dpi=300)
    plt.close()
    logging.info(f"Saved histogram: neighbor_count_histogram.pdf")


# -----------------------------------------------------------------------------
# 2. FRACTION CONDENSED (Line Plot) - LOG SCALE
# -----------------------------------------------------------------------------
def plot_fraction_condensed(output_dir: Path, rbm_list=RBM_LIST, ratio_list=RATIO_LIST):
    logging.info("Generating Fraction Condensed Line Plot...")

    with (output_dir / "condensed_phase_data_iqr.dat").open("w") as f:
        f.write("#RBM\tRatio\tMedian\tQ1\tQ3\n")

        # Square Aspect Ratio
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_box_aspect(1.25)

        for rbm in rbm_list:
            x_vals, y_med, y_err_lo, y_err_hi = [], [], [], []
            for ratio in ratio_list:
                fname = output_dir / f"{rbm}rbm_1_{ratio}_frame_densities.dat"
                if not fname.exists(): continue
                try:
                    d = np.genfromtxt(fname)
                    if d.size == 0: continue
                    n = d[:, 1] if d.ndim > 1 else np.array([d[1]])

                    nrub = 325 if ratio > 3 else 832
                    fracs = n / nrub
                    med = np.median(fracs)
                    q1, q3 = np.percentile(fracs, [25, 75])

                    x_vals.append(ratio);
                    y_med.append(med)
                    y_err_lo.append(med - q1);
                    y_err_hi.append(q3 - med)
                    f.write(f"{rbm}\t{ratio}\t{med:.3f}\t{q1:.3f}\t{q3:.3f}\n")
                except:
                    continue

            if x_vals:
                color = RBM_COLORS.get(rbm, FALLBACK_COLOR)
                ax.errorbar(x_vals, y_med, yerr=[y_err_lo, y_err_hi],
                            fmt='-o', lw=2, ms=8, capsize=5,
                            color=color, ecolor='black', markeredgecolor='black',
                            label=f"{rbm}RBM")

        # Log Scale Config
        ax.set_xscale('log')
        # Explicit ticks for the ratios we have + extras for visual scaling
        ax.set_xticks([1, 2, 3, 5, 10])
        ax.set_xticklabels([1, 2, 3, 5, 10])
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())  # Hide minor tick labels

        ax.set_xlim(0.8, 12)
        ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel("Ratio (EPYC1 : Rubisco) [Log Scale]", fontweight='bold')
        ax.set_ylabel("Fraction Condensed (Median $\pm$ IQR)", fontweight='bold')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "fraction_condensed_iqr.pdf", dpi=300)
        plt.close()
    logging.info("Saved plot: fraction_condensed_iqr.pdf")


# -----------------------------------------------------------------------------
# 3. CONVEX HULL CONCENTRATION (Box Plot) - LINEAR
# -----------------------------------------------------------------------------
def plot_convex_hull_concentration(output_dir: Path, ratio=TARGET_RATIO_FOR_BOXPLOT, rbm_list=RBM_LIST):
    logging.info(f"Generating Hull Concentration Box Plot (Ratio={ratio})...")
    data_to_plot, labels, box_colors = [], [], []

    for rbm in rbm_list:
        fname = output_dir / f"{rbm}rbm_1_{ratio}_hull_concentration_per_frame.dat"
        if not fname.exists(): continue
        try:
            d = np.genfromtxt(fname, comments='#')
            c = d[:, -1] if d.ndim > 1 else np.array([d[-1]])
            c = c[~np.isnan(c)]
            if len(c) > 0:
                data_to_plot.append(c)
                labels.append(rbm)
                box_colors.append(RBM_COLORS.get(rbm, FALLBACK_COLOR))
        except:
            continue

    if not data_to_plot: return

    fig, ax = plt.subplots(figsize=(7, 5))

    bplot = ax.boxplot(data_to_plot, positions=labels, widths=0.6,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color='black', linewidth=2.5),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5))

    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor('black')

    ax.set_xlabel("RBM (Proportional to Linker Length)", fontweight='bold')
    ax.set_ylabel("Convex-hull concentration (µM)", fontweight='bold')
    ax.set_title(f"Condensed-phase concentration (Ratio = {ratio})", pad=15)

    ax.set_xticks(labels)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"convex_hull_concentration_ratio{ratio}_boxplot_clean.pdf", dpi=300)
    plt.close()
    logging.info(f"Saved plot: convex_hull_concentration_ratio{ratio}_boxplot_clean.pdf")


# -----------------------------------------------------------------------------
# 4. COMBINED RDF ANALYSIS
# -----------------------------------------------------------------------------
def plot_combined_rdf_analysis(output_dir: Path, rbm_filter=RDF_RBM_FILTER, ratio_filter=RDF_RATIO_FILTER):
    logging.info(f"Generating Combined RDF Analysis for RBMs {rbm_filter}...")

    rdf_files = []
    for rbm in rbm_filter:
        for ratio in ratio_filter:
            fname = output_dir / f"{rbm}rbm_1_{ratio}_rdf_condensed.dat"
            if fname.exists(): rdf_files.append(fname)

    if not rdf_files: return

    rdf_list, labels_list, colors_list = [], [], []
    r_common = None
    rdf_files = sorted(rdf_files)

    for f in rdf_files:
        try:
            data = np.loadtxt(f)
            r, g = data[:, 0], data[:, 1]
            if r_common is None: r_common = r.copy()

            if not np.allclose(r, r_common):
                g = interp1d(r, g, bounds_error=False, fill_value="extrapolate")(r_common)

            rdf_list.append(g)

            parts = f.name.split('_')
            rbm_val = int(parts[0].replace('rbm', ''))
            ratio_val = parts[2]

            labels_list.append(f"RBM {rbm_val}, Ratio {ratio_val}")
            colors_list.append(RBM_COLORS.get(rbm_val, FALLBACK_COLOR))
        except:
            pass

    if not rdf_list: return
    rdf_array = np.vstack(rdf_list)
    g_avg = np.mean(rdf_array, axis=0)

    minima_locations = []
    try:
        for g_ind in rdf_array:
            gs = savgol_filter(g_ind, 31, 3)
            pks, _ = find_peaks(gs, height=0.1)
            if len(pks) > 0:
                i1 = pks[0]
                sl = slice(i1 + 1, min(i1 + 200, len(gs)))
                i_min = np.argmin(gs[sl]) + i1 + 1
                minima_locations.append(r_common[i_min])
    except:
        pass

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, g in enumerate(rdf_array):
        ax.plot(r_common, g, color=colors_list[i], alpha=0.8, lw=1.5, label=labels_list[i])

    ax.plot(r_common, g_avg, lw=3, color='black', label="Average RDF", zorder=5)

    if minima_locations:
        m, s = np.mean(minima_locations), np.std(minima_locations)
        ax.axvline(m, color="black", linestyle="--", linewidth=2, label=f"Avg Min: {m:.1f} $\pm$ {s:.1f} nm")
        ax.axvspan(m - s, m + s, color='gray', alpha=0.2, zorder=0)

    ax.axvline(20.0, color="red", linestyle=":", linewidth=2, label="Threshold (20 nm)")

    ax.set_xlabel("r (nm)", fontweight='bold')
    ax.set_ylabel("g(r)", fontweight='bold')
    ax.set_title(f"Combined RDF Analysis (RBMs: {'_'.join(map(str, rbm_filter))})")

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, RDF_MAX_R)
    #ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "rdf_analysis_stats.pdf", dpi=300)
    plt.close()
    logging.info(f"Saved RDF Analysis plot: rdf_analysis_stats.pdf")


# -----------------------------------------------------------------------------
# 5. MSD RAINCLOUD PLOTS
# -----------------------------------------------------------------------------
def plot_msd_rainclouds(output_dir: Path):
    logging.info("Generating MSD Raincloud Plots...")

    def load_phase_data(phase_suffix):
        data, lbls, cols = [], [], []
        for rbm in sorted(RBM_LIST):
            for ratio in RATIO_LIST:
                f = output_dir / f"{rbm}rbm_1_{ratio}_D_{phase_suffix}.dat"
                if not f.exists(): continue
                try:
                    arr = np.loadtxt(f)
                    if arr.size == 0: continue
                    d_vals = arr if arr.ndim == 1 else (arr[:, 0] if arr.shape[1] >= 1 else None)
                    if d_vals is None: continue

                    data.append(d_vals)
                    lbls.append(f"{rbm}x (1:{ratio})")
                    cols.append(RBM_COLORS.get(rbm, FALLBACK_COLOR))
                except:
                    pass
        return data, lbls, cols

    cond_res = load_phase_data("condensed")
    dil_res = load_phase_data("dilute")

    if not cond_res[0] and not dil_res[0]: return

    def make_df(res):
        d_list, labels, colors = res
        if not d_list: return None, None
        vals = np.concatenate(d_list)
        groups = []
        for l, arr in zip(labels, d_list): groups.extend([l] * len(arr))
        vals = np.maximum(vals, 1e-10)
        return pd.DataFrame({"Trajectory": groups, "D": vals, "D_log": np.log10(vals)}), colors

    def plot_raincloud(df, colors, title, fname):
        if df is None or df.empty: return
        labels = df["Trajectory"].unique()
        n_groups = len(labels)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * n_groups + 1)))

        pt.half_violinplot(x="D_log", y="Trajectory", data=df, palette=colors,
                           inner=None, width=0.6, orient="h", ax=ax, alpha=0.7)

        box_data = [df[df["Trajectory"] == l]["D_log"].values for l in labels]
        ax.boxplot(box_data, vert=False, positions=np.arange(n_groups) + 0.15,
                   widths=0.15, showfliers=False,
                   medianprops=dict(color="black", linewidth=2),
                   boxprops=dict(color="black", linewidth=1),
                   whiskerprops=dict(color="black", linewidth=1),
                   capprops=dict(color="black", linewidth=1))

        for i, l in enumerate(labels):
            sub = df[df["Trajectory"] == l]
            y_j = np.random.uniform(i - 0.05, i + 0.05, size=len(sub))

            # Downsample heavy plots
            if len(sub) > 1000:
                indices = np.random.choice(len(sub), 1000, replace=False)
                ax.scatter(sub["D_log"].iloc[indices], y_j[indices], color='grey', s=5, alpha=0.3, edgecolor='none')
            else:
                ax.scatter(sub["D_log"], y_j, color='grey', s=5, alpha=0.3, edgecolor='none')

        ax.set_yticks(np.arange(n_groups))
        ax.set_yticklabels(labels)
        ax.set_xlim(-4, 2)
        ticks = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels(ticks)
        ax.set_xlabel("D (µm²/s, log scale)")
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=300)
        plt.close()
        logging.info(f"Saved Raincloud: {fname}")

    cdf, ccols = make_df(cond_res)
    plot_raincloud(cdf, ccols, "Condensed Phase Diffusivity", "condensed_raincloud.pdf")

    ddf, dcols = make_df(dil_res)
    plot_raincloud(ddf, dcols, "Dilute Phase Diffusivity", "dilute_raincloud.pdf")


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.output_dir)
    if not out.exists(): return

    plot_neighbor_histogram(out)
    plot_fraction_condensed(out)
    plot_convex_hull_concentration(out)
    plot_combined_rdf_analysis(out)
    plot_msd_rainclouds(out)

    logging.info("Plotting complete.")


if __name__ == "__main__":
    main()