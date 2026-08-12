import numpy as np

def hydrogen_Pr(T_K, P_bar=700):
    if P_bar >= 600:
        Pr = 0.8921 - 0.001073 * (T_K - 273.15)
    else:
        Pr = 0.7124 - 0.000891 * (T_K - 273.15)
    return max(Pr, 0.5)

def hydrogen_density(T_K, P_bar=700):
    M_H2   = 2.016e-3
    R      = 8.314
    P_Pa   = P_bar * 1e5
    Z = 1.0 + 0.0064 * P_bar - 0.000012 * P_bar * T_K / 300.0
    Z = max(Z, 1.0)
    rho = (P_Pa * M_H2) / (Z * R * T_K)
    return rho

def hydrogen_viscosity(T_K, P_bar=700):
    mu_0 = 8.411e-6 * (T_K / 293.0) ** 0.68
    delta_mu = 1.0 + 0.00142 * P_bar * (293.0 / T_K)
    mu = mu_0 * delta_mu
    return mu

def hydrogen_thermal_conductivity(T_K, P_bar=700):
    k_0 = 0.1687 * (T_K / 293.0) ** 0.72
    k = k_0 * (1.0 + 0.00098 * P_bar)
    return k

def hydrogen_cp(T_K, P_bar=700):
    cp = 13200.0 - 8.5 * (T_K - 273.15) + 0.045 * (T_K - 273.15) ** 2
    cp = cp * (1.0 - 0.00015 * P_bar)
    return max(cp, 10000.0)

def get_all_properties(T_K, P_bar=700):
    return {
        "Pr"  : hydrogen_Pr(T_K, P_bar),
        "rho" : hydrogen_density(T_K, P_bar),
        "mu"  : hydrogen_viscosity(T_K, P_bar),
        "k"   : hydrogen_thermal_conductivity(T_K, P_bar),
        "cp"  : hydrogen_cp(T_K, P_bar),
    }

if __name__ == "__main__":
    print("=" * 55)
    print("Hydrogen Properties — 700 bar, T = 233 K to 333 K")
    print("=" * 55)
    print(f"{'T (K)':<10} {'Pr':<8} {'rho (kg/m³)':<14} {'mu (Pa·s)':<14} {'k (W/mK)':<10}")
    print("-" * 55)
    for T in [233, 253, 273, 293, 313, 333]:
        props = get_all_properties(T, P_bar=700)
        print(f"{T:<10} {props['Pr']:<8.4f} {props['rho']:<14.2f} {props['mu']:<14.2e} {props['k']:<10.4f}")
    print("=" * 55)