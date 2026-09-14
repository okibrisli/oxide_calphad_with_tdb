"""
Liquidus/solidus scan for a binary SiO2-Al2O3 composition using the
C_A_S_Fe_O_M.tdb oxide database.

Composition: 30 mol% Al2O3, 70 mol% SiO2 (oxide basis), ~42 wt% Al2O3.
This sits in the mullite + liquid field, above the SiO2-mullite eutectic
(~1868 K), so liquid should appear well before 2300 K.

Key fix vs. the previous attempt: components are restricted to A, Q, O, VA
only (CaO/MgO/Fe dropped entirely instead of pinned to zero), and the
phase list is restricted to the phases actually relevant to the SiO2-Al2O3
binary. This avoids (1) spurious GAS-phase mass uptake and (2) degenerate/
singular conditions from forcing L, M, FE global fractions to exactly zero
while still carrying phases built from those elements.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

TDB_FILE = "../C_A_S_Fe_O_M.tdb"
dbf = Database(TDB_FILE)

# ------------------------------------------------------------------
# Target bulk composition on an oxide basis: 30 mol% Al2O3, 70 mol% SiO2
# ------------------------------------------------------------------
x_Al2O3_oxide = 0.30
x_SiO2_oxide = 0.70

mol_Al2O3 = x_Al2O3_oxide
mol_SiO2 = x_SiO2_oxide

mol_A = 2.0 * mol_Al2O3          # Al2O3 -> 2 A
mol_Q = 1.0 * mol_SiO2           # SiO2  -> 1 Q
mol_O = 3.0 * mol_Al2O3 + 2.0 * mol_SiO2

total = mol_A + mol_Q + mol_O
X_A = mol_A / total
X_Q = mol_Q / total
X_O = mol_O / total

wt_Al2O3 = mol_Al2O3 * 101.96
wt_SiO2 = mol_SiO2 * 60.08
wt_pct_Al2O3 = 100.0 * wt_Al2O3 / (wt_Al2O3 + wt_SiO2)

print(f"Elemental mole fractions: X_A={X_A:.5f}, X_Q={X_Q:.5f}, X_O={X_O:.5f}")
print(f"Equivalent to about {wt_pct_Al2O3:.1f} wt% Al2O3 / {100-wt_pct_Al2O3:.1f} wt% SiO2")

# ------------------------------------------------------------------
# Restricted components and phases for the true SiO2-Al2O3 binary
# ------------------------------------------------------------------
comps = ["A", "Q", "O", "VA"]

phases_to_use = [
    "CORUNDUM",  # Al2O3
    "AQUARTZ",   # alpha quartz SiO2
    "BQUARTZ",   # beta quartz SiO2
    "ACRIS",     # alpha cristobalite SiO2
    "BCRIS",     # beta cristobalite SiO2
    "MULLITE",   # 3Al2O3.2SiO2 solid solution
    "LIQUID",    # oxide melt
]

base_conds = {
    v.X("A"): X_A,
    v.X("Q"): X_Q,
    v.P: 101325,
    v.N: 1,
}


def liquid_fraction_and_phases(T_kelvin):
    conds = dict(base_conds)
    conds[v.T] = T_kelvin
    try:
        eq = equilibrium(dbf, comps, phases_to_use, conds)
    except Exception as e:
        return None, [f"FAILED: {e}"]

    phase_vals = np.atleast_1d(eq.Phase.values.squeeze())
    np_vals = np.atleast_1d(eq.NP.values.squeeze())

    liq_frac = 0.0
    solids = []
    for ph, amt in zip(phase_vals, np_vals):
        if ph in ("", "_FAKE_") or amt is None or np.isnan(amt):
            continue
        if ph == "LIQUID":
            liq_frac += float(amt)
        else:
            solids.append((str(ph), float(amt)))
    return liq_frac, solids


# ------------------------------------------------------------------
# Scan temperature range; eutectic is near 1868 K, so start below that
# ------------------------------------------------------------------
print("\n T (K)   T (degC)   Liquid frac   Remaining solids")
T_range = np.arange(1700, 2350, 20)
results = {}
for T in T_range:
    lf, solids = liquid_fraction_and_phases(float(T))
    if lf is None:
        lf, solids = liquid_fraction_and_phases(float(T) + 0.5)
    results[T] = (lf if lf is not None else 0.0, solids)
    lf_show = results[T][0]
    solid_str = ", ".join(f"{p}({a:.3f})" for p, a in solids) if solids else "-"
    print(f"{T:7.1f}  {T - 273.15:8.1f}   {lf_show:6.3f}       {solid_str}")

# ------------------------------------------------------------------
# Solidus (first appearance of liquid) and liquidus (fully liquid)
# ------------------------------------------------------------------
threshold = 0.999
solidus_T = None
liquidus_T = None
for T in sorted(results):
    lf, _ = results[T]
    if solidus_T is None and lf > 1e-6:
        solidus_T = T
    if liquidus_T is None and lf >= threshold:
        liquidus_T = T
        break

print()
print(f"Results for {wt_pct_Al2O3:.1f} wt% Al2O3 / {100-wt_pct_Al2O3:.1f} wt% SiO2")
if solidus_T:
    print(f"Approximate solidus temperature:  {solidus_T} K ({solidus_T - 273.15:.0f} degC)")
else:
    print("No liquid appears in the scanned temperature range -- widen T_range")

if liquidus_T:
    print(f"Approximate liquidus temperature: {liquidus_T} K ({liquidus_T - 273.15:.0f} degC)")
else:
    print("Still not fully liquid within scanned range -- widen T_range upward")
