# =============================================================
# PCHE Nu-f Correlation Benchmark — Multi-Condition Sweep
# Author: Aakash Yadav, DTU Mechanical Engineering
# Description: Sweeps the correlation library across the full
#              SAE J2601 precooling design space (T, P, angle,
#              Re) and exports tidy results to CSV for analysis
#              and plotting.
# =============================================================

import numpy as np
import pandas as pd

from hydrogen_properties import get_all_properties
from correlations import (
    kim_no_Nu, kim_no_f,
    meshram_Nu, meshram_f,
    ngo_Nu, ngo_f,
    dittus_boelter_Nu, straight_f,
    compute_PEC,
)

# -------------------------------------------------------------
# Design space (edit these to change sweep coverage)
# -------------------------------------------------------------
T_VALUES     = [233, 253, 273, 293, 313, 333]      # K
P_VALUES     = [350, 700]                          # bar
ANGLE_VALUES = [30, 45, 60]                        # degrees (zigzag only)
RE_VALUES    = [2000, 5000, 10000, 20000, 40000]

CORRELATIONS = {
    "Zigzag - Kim & No (2011)": dict(needs_angle=True,
        Nu=lambda Re, Pr, ang: kim_no_Nu(Re, Pr, ang),
        f =lambda Re, ang: kim_no_f(Re, ang)),
    "Zigzag - Meshram (2016)": dict(needs_angle=True,
        Nu=lambda Re, Pr, ang: meshram_Nu(Re, Pr, ang),
        f =lambda Re, ang: meshram_f(Re, ang)),
    "S-shaped - Ngo (2007)": dict(needs_angle=False,
        Nu=lambda Re, Pr, ang: ngo_Nu(Re, Pr),
        f =lambda Re, ang: ngo_f(Re)),
}


def run_sweep():
    """Run the full T x P x angle x Re x correlation sweep."""
    rows = []

    for P_bar in P_VALUES:
        for T_K in T_VALUES:
            props = get_all_properties(T_K, P_bar)
            Pr = props["Pr"]

            for Re in RE_VALUES:
                Nu_ref = dittus_boelter_Nu(Re, Pr)
                f_ref  = straight_f(Re)

                rows.append(dict(
                    T_K=T_K, P_bar=P_bar, angle_deg=np.nan, Re=Re,
                    correlation="Straight - Dittus-Boelter (baseline)",
                    Pr=Pr, Nu=Nu_ref, f=f_ref, PEC=1.0,
                ))

                for name, corr in CORRELATIONS.items():
                    angles = ANGLE_VALUES if corr["needs_angle"] else [np.nan]
                    for ang in angles:
                        a = ang if corr["needs_angle"] else 45  # dummy, unused when needs_angle=False
                        Nu = corr["Nu"](Re, Pr, a)
                        f  = corr["f"](Re, a)
                        PEC = compute_PEC(Nu, f, Nu_ref, f_ref)
                        rows.append(dict(
                            T_K=T_K, P_bar=P_bar, angle_deg=ang, Re=Re,
                            correlation=name,
                            Pr=Pr, Nu=Nu, f=f, PEC=PEC,
                        ))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_sweep()
    out_path = "benchmark_results.csv"
    df.to_csv(out_path, index=False)

    print("=" * 70)
    print(f"Sweep complete: {len(df)} rows")
    print(f"  T:     {T_VALUES} K")
    print(f"  P:     {P_VALUES} bar")
    print(f"  angle: {ANGLE_VALUES} deg (zigzag geometries only)")
    print(f"  Re:    {RE_VALUES}")
    print(f"Saved to {out_path}")
    print("=" * 70)

    print("\nMean PEC by correlation (across full design space):")
    print(df.groupby("correlation")["PEC"].mean().sort_values(ascending=False).round(3))
