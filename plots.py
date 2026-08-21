# =============================================================
# PCHE Nu-f Correlation Benchmark — Visualization
# Author: Aakash Yadav, DTU Mechanical Engineering
# Description: Reads benchmark_results.csv (from benchmark_sweep.py)
#              and produces comparison plots: Nu vs Re, f vs Re,
#              and PEC vs Re, at a representative operating point.
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "benchmark_results.csv"

# Representative SAE J2601 precooling condition for the headline plots
PLOT_T_K     = 253    # -20 C
PLOT_P_BAR   = 700
PLOT_ANGLE   = 45     # deg, applies to zigzag correlations only


def load_and_filter(df):
    """Select rows at the representative condition, one series per correlation."""
    mask_T = df["T_K"] == PLOT_T_K
    mask_P = df["P_bar"] == PLOT_P_BAR
    # angle_deg is NaN for baseline/S-shaped (angle-independent); only filter
    # angle where it's actually used (zigzag correlations)
    mask_angle = df["angle_deg"].isna() | (df["angle_deg"] == PLOT_ANGLE)
    return df[mask_T & mask_P & mask_angle].sort_values("Re")


def make_plot(df, y_col, ylabel, title, out_path, logy=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, group in df.groupby("correlation"):
        ax.plot(group["Re"], group[y_col], marker="o", label=name)
    ax.set_xlabel("Reynolds number, Re")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    subset = load_and_filter(df)

    cond_str = f"T = {PLOT_T_K} K, P = {PLOT_P_BAR} bar, angle = {PLOT_ANGLE} deg"

    make_plot(subset, "Nu", "Nusselt number, Nu",
              f"Nu vs Re  ({cond_str})", "plot_Nu_vs_Re.png")

    make_plot(subset, "f", "Fanning friction factor, f",
              f"f vs Re  ({cond_str})", "plot_f_vs_Re.png", logy=True)

    make_plot(subset, "PEC", "Performance Evaluation Criterion, PEC",
              f"PEC vs Re  ({cond_str})", "plot_PEC_vs_Re.png")

    print("\nDone. PEC > 1.0 indicates better thermo-hydraulic performance")
    print("than the straight-channel (Dittus-Boelter) baseline.")
