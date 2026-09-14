"""
Revised version: diagnoses (1) which phase is persisting near-liquidus,
and (2) works around isolated solver-convergence failures by computing
one temperature at a time with retries, instead of one big batched call.

Same composition as before: 60 wt% SiO2, 10 wt% Al2O3, 10 wt% CaO,
10 wt% MgO, 10 wt% Fe2O3.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

TDB_FILE = "../C_A_S_Fe_O_M.tdb"
dbf = Database(TDB_FILE)

wt = {"SiO2": 100.0, "Al2O3": 0.0, "CaO": 0.0, "MgO": 0.0, "Fe2O3": 0.0}
M_SiO2, M_Al2O3, M_CaO, M_MgO, M_Fe2O3 = 60.08, 101.96, 56.07, 40.30, 159.69

mol_SiO2 = wt["SiO2"] / M_SiO2
mol_Al2O3 = wt["Al2O3"] / M_Al2O3
mol_CaO = wt["CaO"] / M_CaO
mol_MgO = wt["MgO"] / M_MgO
mol_Fe2O3 = wt["Fe2O3"] / M_Fe2O3
mol_FE = 2 * mol_Fe2O3
mol_O = 3 * mol_Fe2O3
total = mol_SiO2 + mol_Al2O3 + mol_CaO + mol_MgO + mol_FE + mol_O

X_Q = mol_SiO2 / total
X_A = mol_Al2O3 / total
X_L = mol_CaO / total
X_M = mol_MgO / total
X_FE = mol_FE / total
X_O = mol_O / total

comps = ["A", "L", "Q", "M", "FE", "O", "VA"]
phases_to_use = [p for p in dbf.phases.keys() if p not in ("GAS:G",)]

base_conds = {
    v.X("A"): X_A, v.X("Q"): X_Q, v.X("M"): X_M,
    v.X("FE"): X_FE, v.X("O"): X_O,
    v.P: 101325, v.N: 1,
}


def liquid_fraction_and_phases(T_kelvin):
    """Run equilibrium at ONE temperature; return (liquid_frac, solid_phase_list)."""
    conds = dict(base_conds)
    conds[v.T] = T_kelvin
    try:
        eq = equilibrium(dbf, comps, phases_to_use, conds)
    except Exception as e:
        return None, [f"FAILED: {e}"]

    phase_vals = eq.Phase.values.squeeze()
    np_vals = eq.NP.values.squeeze()
    phase_vals = np.atleast_1d(phase_vals)
    np_vals = np.atleast_1d(np_vals)

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


# ---------------------------------------------------------------
# Widen the scan up to 2800 K given MgO's very high melting point,
# and retry with a tiny T nudge if a point looks like it failed.
# ---------------------------------------------------------------
print(" T (K)   T (degC)   Liquid frac   Remaining solids")
T_range = np.arange(1300, 2800, 20)
results = {}
for T in T_range:
    lf, solids = liquid_fraction_and_phases(float(T))
    if lf is None or (lf == 0.0 and len(results) > 0 and
                       list(results.values())[-1][0] > 0.3):
        # looked like a convergence failure -- retry with a nudge
        lf, solids = liquid_fraction_and_phases(float(T) + 0.5)
    results[T] = (lf, solids)
    solid_str = ", ".join(f"{p}({a:.3f})" for p, a in solids) if solids else "-"
    print(f"{T:7.1f}  {T - 273.15:8.1f}   {lf:6.3f}       {solid_str}")

# ---------------------------------------------------------------
# Find the true liquidus: first T (scanning up) where liquid frac
# is close enough to 1.0 that only trace solid remains.
# ---------------------------------------------------------------
threshold = 0.999
liquidus_T = None
for T in sorted(results):
    lf, _ = results[T]
    if lf is not None and lf >= threshold:
        liquidus_T = T
        break

if liquidus_T:
    print(f"\nLiquidus (>{threshold*100:.1f}% liquid): {liquidus_T} K "
          f"({liquidus_T - 273.15:.0f} degC)")
else:
    print("\nStill not fully liquid by 2800 K -- the persistent solid is "
          "genuinely very refractory. Check which phase it is in the "
          "'Remaining solids' column above (likely MONOXIDE/periclase, "
          "M2S/forsterite, or a spinel) -- that tells you which oxide "
          "is over-saturating the melt at this composition.")
