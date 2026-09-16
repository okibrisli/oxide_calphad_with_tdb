"""
Estimate the melting (liquidus) temperature of a chosen composition
in the SiO2-Al2O3-CaO (C-A-S) system using pycalphad and the
CaO-Al2O3-SiO2-FeO-MgO thermodynamic database derived from the
Portland-cement thesis TDB (cleaned version).

Composition chosen: anorthite, CaAl2Si2O8
  CaO : Al2O3 : SiO2 = 1 : 1 : 2  (mole ratio)
  -> mole fractions of the TDB's oxide "elements":
       X(L) = CaO   = 0.25
       X(A) = Al2O3 = 0.25
       X(Q) = SiO2  = 0.50
  This composition sits exactly on the anorthite (CAS2) stoichiometric
  compound in this database, so it is a good sanity-check composition:
  literature liquidus/melting point of anorthite is about 1553 degC
  (1826 K) at 1 atm.

Install requirements first:
    pip install pycalphad

Usage:
    python calc_melting_temperature_CAS.py
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# ---------------------------------------------------------------
# 1. Load the database
# ---------------------------------------------------------------
TDB_FILE = "CMAS_FeO_Portland_thesis.tdb"  # the cleaned file from before
dbf = Database(TDB_FILE)

print("Elements in database:", sorted(dbf.elements))
print("Number of phases:", len(dbf.phases))

# ---------------------------------------------------------------
# 2. Define the composition (anorthite, CaAl2Si2O8)
#    Change these three numbers to explore other C-A-S compositions.
#    They must sum to 1 together with the trace Fe/Mg/O below.
# ---------------------------------------------------------------
X_CAO   = 0.25   # L  = CaO
X_AL2O3 = 0.25   # A  = Al2O3
X_SIO2  = 0.50   # Q  = SiO2   (dependent component, not set directly)

# Keep the Fe/Mg/O sub-system essentially switched off so we stay
# inside the pure SiO2-Al2O3-CaO ternary.
TRACE = 1e-6

comps = ["A", "L", "Q", "FE", "M", "O", "VA"]

# Only keep phases that don't require a real Fe/Mg presence to be
# thermodynamically relevant -- not strictly necessary (equilibrium
# will just suppress them), but restricting the phase list makes the
# calculation much faster.
phases_to_use = [
    "LIQUID", "MULLITE", "CORUNDUM", "AQUARTZ", "BQUARTZ", "ACRIS", "BCRIS",
    "AC2S", "APC2S", "BC2S", "GC2S", "MC3S", "RC3S", "RC3S2", "TC3S",
    "WCS", "PCS", "CAS2", "C2AS", "C3A", "CA", "CA2", "CA6", "C12A7",
    "CAO", "WOLL",
]
phases_to_use = [p for p in phases_to_use if p in dbf.phases]

# ---------------------------------------------------------------
# 3. Scan temperature and find where LIQUID first appears / disappears
# ---------------------------------------------------------------
T_range = np.arange(1300, 2000, 5)  # Kelvin, coarse scan first

conds = {
    v.X("A"): X_AL2O3,
    v.X("Q"): X_SIO2,
    v.X("FE"): TRACE,
    v.X("M"): TRACE,
    v.X("O"): TRACE,
    v.T: T_range,
    v.P: 101325,
    v.N: 1,
}

print("Running coarse equilibrium scan...")
eq = equilibrium(dbf, comps, phases_to_use, conds)

# NP = phase amounts (moles of each phase) at equilibrium
liquid_fraction = eq.NP.sel(vertex=0).where(eq.Phase.isel(vertex=0) == "LIQUID")

# Simpler and more robust: sum NP over all vertices where Phase == 'LIQUID'
def liquid_mole_fraction(eq_ds):
    is_liquid = eq_ds.Phase == "LIQUID"
    liq_np = eq_ds.NP.where(is_liquid, other=0.0).sum(dim="vertex")
    return liq_np.values.squeeze()

liq_frac = liquid_mole_fraction(eq)
temps = T_range

print("\n T (K)   T (degC)   Liquid mole fraction")
for t, lf in zip(temps, liq_frac):
    print(f"{t:7.1f}  {t-273.15:8.1f}   {lf:6.3f}")

# ---------------------------------------------------------------
# 4. Extract the liquidus temperature: the lowest T (scanning down
#    from high T) where the system is still 100% liquid.
# ---------------------------------------------------------------
fully_liquid = np.isclose(liq_frac, 1.0, atol=1e-3)
if fully_liquid.any():
    liquidus_T = temps[fully_liquid][0]  # first True scanning low->high T
    print(f"\nApprox. liquidus temperature: {liquidus_T} K "
          f"({liquidus_T-273.15:.0f} degC) [coarse, 5 K steps]")

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
              f"({liquidus_T_fine-273.15:.1f} degC)")
else:
    print("\nNo fully-liquid state found in the scanned range 1300-2000 K; "
          "widen T_range (e.g. up to 2200-2300 K for silica-rich mixtures).")