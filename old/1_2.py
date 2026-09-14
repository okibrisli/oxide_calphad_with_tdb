"""
Estimate the melting (liquidus) temperature of a chosen composition
in the CaO-SiO2 binary system using pycalphad and the peer-reviewed,
open-access TDB from:

  Abdul, Mawalala, Pisch, Bannerman (2023),
  "CaO-SiO2 assessment using 3rd generation CALPHAD models",
  Cement and Concrete Research 173, 107309 (CC-BY 4.0).

This file (Supplementary-CaO_SiO2-R2.tdb) is a native, properly
formatted TDB -- not extracted from a PDF -- so it should load into
pycalphad without any manual cleanup.

Composition chosen: wollastonite, CaSiO3
  CaO : SiO2 = 1 : 1  (mole ratio)
  -> X(L) = 0.5 (CaO), X(Q) = 0.5 (SiO2)
  This sits exactly on the WCS (wollastonite) / PCS (pseudowollastonite)
  stoichiometric compound in this database. Reported melting point of
  wollastonite in the literature is about 1544 degC (~1817 K).

Install requirements first:
    pip install pycalphad

Usage:
    python calc_melting_temperature_CaO_SiO2.py
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# ---------------------------------------------------------------
# 1. Load the database
# ---------------------------------------------------------------
TDB_FILE = "../Supplementary-CaO_SiO2-R2.tdb"
dbf = Database(TDB_FILE)

print("Elements in database:", sorted(dbf.elements))
print("Phases in database:", sorted(dbf.phases.keys()))

# ---------------------------------------------------------------
# 2. Define the composition (wollastonite, CaSiO3)
#    Change this single number to explore other CaO-SiO2 compositions,
#    e.g. 0.333 for C3S (Ca3SiO5-ish region), 0.667 for C2S (Ca2SiO4).
#    Since there are only 2 components (L=CaO, Q=SiO2), only one
#    independent mole fraction needs to be fixed.
# ---------------------------------------------------------------
X_SIO2 = 0.5   # Q = SiO2 mole fraction; X(L) = 1 - X_SIO2 is dependent

comps = ["L", "Q", "VA"]
phases_to_use = list(dbf.phases.keys())  # small system, use everything

# ---------------------------------------------------------------
# 3. Scan temperature and find where LIQUID first appears / disappears
# ---------------------------------------------------------------
T_range = np.arange(1300, 2200, 5)  # Kelvin, coarse scan first

conds = {
    v.X("Q"): X_SIO2,
    v.T: T_range,
    v.P: 101325,
    v.N: 1,
}

print("\nRunning coarse equilibrium scan...")
eq = equilibrium(dbf, comps, phases_to_use, conds)


def liquid_mole_fraction(eq_ds):
    """Sum NP (phase amount) over all vertices where Phase == 'LIQUID'."""
    is_liquid = eq_ds.Phase == "LIQUID"
    liq_np = eq_ds.NP.where(is_liquid, other=0.0).sum(dim="vertex")
    return liq_np.values.squeeze()


liq_frac = liquid_mole_fraction(eq)
temps = T_range

print("\n T (K)   T (degC)   Liquid mole fraction")
for t, lf in zip(temps, liq_frac):
    print(f"{t:7.1f}  {t - 273.15:8.1f}   {lf:6.3f}")

# ---------------------------------------------------------------
# 4. Extract the liquidus temperature: the lowest T (scanning down
#    from high T) where the system is still 100% liquid.
# ---------------------------------------------------------------
fully_liquid = np.isclose(liq_frac, 1.0, atol=1e-3)
if fully_liquid.any():
    liquidus_T = temps[fully_liquid][0]  # first True scanning low->high T
    print(f"\nApprox. liquidus temperature: {liquidus_T} K "
          f"({liquidus_T - 273.15:.0f} degC) [coarse, 5 K steps]")

    # ---------------------------------------------------------------
    # 5. Refine with a finer scan bracketing the coarse estimate
    # ---------------------------------------------------------------
    T_fine = np.arange(liquidus_T - 20, liquidus_T + 5, 0.5)
    conds[v.T] = T_fine
    eq_fine = equilibrium(dbf, comps, phases_to_use, conds)
    liq_frac_fine = liquid_mole_fraction(eq_fine)

    fully_liquid_fine = np.isclose(liq_frac_fine, 1.0, atol=1e-3)
    if fully_liquid_fine.any():
        liquidus_T_fine = T_fine[fully_liquid_fine][0]
        print(f"Refined liquidus temperature: {liquidus_T_fine:.1f} K "
              f"({liquidus_T_fine - 273.15:.1f} degC)")
else:
    print("\nNo fully-liquid state found in the scanned range 1300-2200 K; "
          "widen T_range (e.g. up to 2500 K for silica-rich mixtures).")

# ---------------------------------------------------------------
# 6. Bonus: print which solid phase(s) are stable just below the
#    liquidus, useful to confirm you're near the expected compound.
# ---------------------------------------------------------------
if fully_liquid.any():
    T_just_below = liquidus_T - 10
    conds[v.T] = np.array([T_just_below])
    eq_check = equilibrium(dbf, comps, phases_to_use, conds)
    stable_phases = np.unique(eq_check.Phase.values)
    stable_phases = [p for p in stable_phases if p not in ("", "LIQUID")]
    print(f"\nSolid phase(s) stable at {T_just_below:.0f} K "
          f"({T_just_below - 273.15:.0f} degC): {stable_phases}")
