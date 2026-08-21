# =============================================================
# PCHE Nu-f Correlation Library — Hydrogen Precooling
# Author: Aakash Yadav, DTU Mechanical Engineering
# Description: Published Nu and f correlations for PCHE
#              channels, aligned with SAE J2601 hydrogen
#              precooling research
#
# Geometries covered:
#   - Zigzag channels  (Kim & No 2011, Meshram 2016)
#   - S-shaped channels (Ngo et al. 2007)
#   - Straight channels (Dittus-Boelter, baseline reference)
#
# Performance metric: PEC (Performance Evaluation Criterion)
#   PEC = (Nu/Nu_ref) / (f/f_ref)^(1/3)
#   A PEC > 1 means better heat transfer per pressure penalty
#   than the straight channel baseline
# =============================================================

import numpy as np
from hydrogen_properties import get_all_properties


# =============================================================
# SECTION 1: ZIGZAG CHANNEL CORRELATIONS
# =============================================================

def kim_no_Nu(Re, Pr, angle_deg=45):
    """
    Nusselt number — Kim & No (2011), Zigzag PCHE.
    Source: Nuclear Engineering and Design, Vol. 241
    Valid : Re 2000–58000, angle 30–60°

    Parameters:
        Re        : Reynolds number
        Pr        : Prandtl number
        angle_deg : Zigzag angle in degrees

    Returns:
        Nu : Nusselt number
    """
    angle_rad = np.radians(angle_deg)
    Nu = 0.0292 * (Re ** 0.8) * (Pr ** 0.4) * (np.sin(angle_rad) ** 0.5)
    return Nu


def kim_no_f(Re, angle_deg=45):
    """
    Friction factor — Kim & No (2011), Zigzag PCHE.

    Parameters:
        Re        : Reynolds number
        angle_deg : Zigzag angle in degrees

    Returns:
        f : Fanning friction factor
    """
    angle_rad = np.radians(angle_deg)
    f = 0.2353 * (Re ** -0.2) * (np.sin(angle_rad) ** 0.5)
    return f


def meshram_Nu(Re, Pr, angle_deg=45):
    """
    Nusselt number — Meshram et al. (2016), Zigzag PCHE.
    Source: Applied Thermal Engineering, Vol. 111
    Valid : Re 1000–10000, angle 15–45°

    Parameters:
        Re        : Reynolds number
        Pr        : Prandtl number
        angle_deg : Zigzag angle in degrees

    Returns:
        Nu : Nusselt number
    """
    angle_rad = np.radians(angle_deg)
    Nu = 0.1696 * (Re ** 0.6338) * (Pr ** 0.4) * (angle_rad ** 0.1136)
    return Nu


def meshram_f(Re, angle_deg=45):
    """
    Friction factor — Meshram et al. (2016), Zigzag PCHE.

    Parameters:
        Re        : Reynolds number
        angle_deg : Zigzag angle in degrees

    Returns:
        f : Fanning friction factor
    """
    angle_rad = np.radians(angle_deg)
    f = 0.1924 * (Re ** -0.1601) * (angle_rad ** 0.2281)
    return f


# =============================================================
# SECTION 2: S-SHAPED CHANNEL CORRELATION
# =============================================================

def ngo_Nu(Re, Pr):
    """
    Nusselt number — Ngo et al. (2007), S-shaped PCHE.
    Source: Nuclear Engineering and Design, Vol. 237
    Valid : Re 4000–85000

    Parameters:
        Re : Reynolds number
        Pr : Prandtl number

    Returns:
        Nu : Nusselt number
    """
    Nu = 2.15 * (Re ** 0.514) * (Pr ** 0.4)
    return Nu


def ngo_f(Re):
    """
    Friction factor — Ngo et al. (2007), S-shaped PCHE.

    Parameters:
        Re : Reynolds number

    Returns:
        f : Fanning friction factor
    """
    f = 0.9753 * (Re ** -0.231)
    return f


# =============================================================
# SECTION 3: STRAIGHT CHANNEL BASELINE (Dittus-Boelter)
# =============================================================

def dittus_boelter_Nu(Re, Pr):
    """
    Nusselt number — Dittus-Boelter correlation.
    Standard baseline for straight microchannels.
    Valid : Re > 10000, 0.6 < Pr < 160

    Parameters:
        Re : Reynolds number
        Pr : Prandtl number

    Returns:
        Nu : Nusselt number
    """
    Nu = 0.023 * (Re ** 0.8) * (Pr ** 0.4)
    return Nu


def straight_f(Re):
    """
    Friction factor — Blasius correlation for straight channels.

    Parameters:
        Re : Reynolds number

    Returns:
        f : Fanning friction factor
    """
    if np.isscalar(Re):
        if Re < 2300:
            return 16.0 / Re          # Laminar
        else:
            return 0.0791 * Re**-0.25  # Turbulent (Blasius)
    else:
        f = np.where(Re < 2300,
                     16.0 / Re,
                     0.0791 * Re**-0.25)
        return f


# =============================================================
# SECTION 4: PERFORMANCE EVALUATION CRITERION (PEC)
# =============================================================

def compute_PEC(Nu, f, Nu_ref, f_ref):
    """
    Performance Evaluation Criterion — Webb & Kim (2005).

    Compares a channel's thermo-hydraulic performance
    against a reference (typically straight channel).

    PEC = (Nu/Nu_ref) / (f/f_ref)^(1/3)

    PEC > 1 : Better than reference (more heat transfer
               per unit pumping power penalty)
    PEC = 1 : Same as reference
    PEC < 1 : Worse than reference

    Parameters:
        Nu     : Nusselt number of test channel
        f      : Friction factor of test channel
        Nu_ref : Nusselt number of reference channel
        f_ref  : Friction factor of reference channel

    Returns:
        PEC : Performance Evaluation Criterion value
    """
    PEC = (Nu / Nu_ref) / ((f / f_ref) ** (1/3))
    return PEC


# =============================================================
# SECTION 5: MASTER BENCHMARK FUNCTION
# =============================================================

def benchmark_at_conditions(T_K=253, P_bar=700, angle_deg=45):
    """
    Run full benchmark at a single operating condition.
    Uses real hydrogen properties from hydrogen_properties.py.

    Parameters:
        T_K       : Hydrogen temperature in Kelvin
        P_bar     : System pressure in bar
        angle_deg : PCHE channel angle in degrees

    Returns:
        dict with Nu, f, PEC for all correlations
    """
    props = get_all_properties(T_K, P_bar)
    Pr    = props["Pr"]

    # Test across a representative Re range
    Re_values = np.array([2000, 5000, 10000, 20000, 40000])

    results = {}

    for Re in Re_values:
        # Reference — straight channel (Dittus-Boelter)
        Nu_ref = dittus_boelter_Nu(Re, Pr)
        f_ref  = straight_f(Re)

        results[Re] = {
            "Straight (Dittus-Boelter)": {
                "Nu": Nu_ref, "f": f_ref,
                "PEC": 1.0   # baseline is always 1.0
            },
            "Zigzag — Kim & No (2011)": {
                "Nu": kim_no_Nu(Re, Pr, angle_deg),
                "f" : kim_no_f(Re, angle_deg),
                "PEC": compute_PEC(
                    kim_no_Nu(Re, Pr, angle_deg),
                    kim_no_f(Re, angle_deg),
                    Nu_ref, f_ref)
            },
            "Zigzag — Meshram (2016)": {
                "Nu": meshram_Nu(Re, Pr, angle_deg),
                "f" : meshram_f(Re, angle_deg),
                "PEC": compute_PEC(
                    meshram_Nu(Re, Pr, angle_deg),
                    meshram_f(Re, angle_deg),
                    Nu_ref, f_ref)
            },
            "S-shaped — Ngo (2007)": {
                "Nu": ngo_Nu(Re, Pr),
                "f" : ngo_f(Re),
                "PEC": compute_PEC(
                    ngo_Nu(Re, Pr),
                    ngo_f(Re),
                    Nu_ref, f_ref)
            },
        }

    return results, props


# =============================================================
# QUICK TEST — run this file directly to verify
# =============================================================

if __name__ == "__main__":

    # SAE J2601 precooling condition:
    # Hydrogen at -20°C (253 K), 700 bar
    T_K      = 253
    P_bar    = 700
    angle    = 45

    print("=" * 70)
    print("PCHE CORRELATION BENCHMARK — High-Pressure Hydrogen Precooling")
    print(f"Conditions: T = {T_K} K ({T_K-273:.0f}°C), P = {P_bar} bar, angle = {angle}°")
    print("=" * 70)

    results, props = benchmark_at_conditions(T_K, P_bar, angle)

    print(f"\nFluid Properties at {T_K} K, {P_bar} bar:")
    print(f"  Pr  = {props['Pr']:.4f}")
    print(f"  rho = {props['rho']:.2f} kg/m³")
    print(f"  mu  = {props['mu']:.2e} Pa·s")
    print(f"  k   = {props['k']:.4f} W/m·K")

    print(f"\n{'Correlation':<30} {'Re':>7} {'Nu':>8} {'f':>8} {'PEC':>8}")
    print("-" * 70)

    for Re, corr_dict in results.items():
        first = True
        for name, vals in corr_dict.items():
            if first:
                print(f"{name:<30} {Re:>7} {vals['Nu']:>8.3f} {vals['f']:>8.4f} {vals['PEC']:>8.3f}")
                first = False
            else:
                print(f"{name:<30} {Re:>7} {vals['Nu']:>8.3f} {vals['f']:>8.4f} {vals['PEC']:>8.3f}")
        print()

    print("=" * 70)
    print("PEC > 1.0 means better than straight channel baseline")
    print("=" * 70)