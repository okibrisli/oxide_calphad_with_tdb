"""Composition range scan for equilibrium melting assessment with pycalphad.

Edit COMPOSITION_MODE, OXIDE_RANGES and the temperature settings. The script
calculates first liquid, T10, T50, T90 and T99 for every valid composition.
All temperatures shown and exported are degC.
"""

from pathlib import Path
from itertools import product
import csv
import math
import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases


# USER INPUT
TDB_FILE = Path(__file__).with_name("C_A_S_Fe_O_M.tdb")

# Use BALANCE_COMPONENT for the recommended workflow. All oxide ranges except
# BALANCE_OXIDE are scanned independently. BALANCE_OXIDE is calculated as the
# remainder to give exactly 100 wt%. Its range is still enforced.
#
# Use FULL_GRID only when every listed range and increment deliberately gives
# combinations that sum to exactly 100 wt%. This mode can easily yield no rows.
COMPOSITION_MODE = "BALANCE_COMPONENT"
BALANCE_OXIDE = "SiO2"

# Format: "oxide": (minimum_wt_pct, maximum_wt_pct, increment_wt_pct)
# Keep the initial grid small. Calculation time rises rapidly with every added
# component value because every composition needs a complete temperature scan.
OXIDE_RANGES = {
    "CaO": (15.0, 30.0, 5.0),
    "SiO2": (45.0, 80.0, 5.0),
    "Al2O3": (0.0, 10.0, 5.0),
    "Fe2O3": (0.0, 5.0, 2.5),
    "FeO": (0.0, 0.0, 1.0),
    "MgO": (0.0, 5.0, 2.5),
}

PRESSURE_PA = 101325.0
T_MIN_C = 1000.0
T_MAX_C = 1800.0
COARSE_STEP_C = 50.0
REFINEMENT_TOLERANCE_C = 1.0
PHASE_FRACTION_TOLERANCE = 1.0e-6

# T99 is the practical liquidus for deposit and refractory risk screening.
PRACTICAL_LIQUIDUS_FRACTION = 0.99
LIQUID_FRACTION_MILESTONES = (0.10, 0.50, 0.90, 0.99)
REPORTING_PHASE_FRACTION_MIN = 1.0e-4

# A solver result is accepted only when valid phase fractions sum to one.
# Failed points are retried with the higher phase sampling density.
CALCULATION_OPTIONS_TRIALS = ({"pdens": 500}, {"pdens": 1000})
EXCLUDED_PHASES = {"GAS"}

# Results are sorted by this field for the terminal risk summary.
TOP_RESULTS_TO_PRINT = 15


# Oxide to TDB component mapping. A, L, Q and M represent Al2O3, CaO, SiO2
# and MgO. Iron oxides are converted to independent FE and O amounts.
OXIDE_DEFINITIONS = {
    "CaO": (56.077, {"L": 1.0}),
    "SiO2": (60.0843, {"Q": 1.0}),
    "Al2O3": (101.9613, {"A": 1.0}),
    "MgO": (40.3044, {"M": 1.0}),
    "FeO": (71.844, {"FE": 1.0, "O": 1.0}),
    "Fe2O3": (159.6882, {"FE": 2.0, "O": 3.0}),
    "Fe3O4": (231.531, {"FE": 3.0, "O": 4.0}),
}

PHASE_LABELS = {
    "LIQUID": "LIQUID [oxide melt]",
    "ACRIS": "ACRIS [SiO2, alpha cristobalite]",
    "BCRIS": "BCRIS [SiO2, beta cristobalite]",
    "AQUARTZ": "AQUARTZ [SiO2, alpha quartz]",
    "BQUARTZ": "BQUARTZ [SiO2, beta quartz]",
    "AC2S": "AC2S [Ca2SiO4, alpha belite]",
    "APC2S": "APC2S [Ca2SiO4, alpha prime belite]",
    "BC2S": "BC2S [Ca2SiO4, beta belite]",
    "GC2S": "GC2S [Ca2SiO4, gamma belite]",
    "ALPHA": "ALPHA [(Ca,Mg)2SiO4, alpha belite solution]",
    "ALPHA_PRIME": "ALPHA_PRIME [(Ca,Mg)2SiO4, alpha prime belite solution]",
    "OLIVINE": "OLIVINE [(Ca,Mg)2SiO4, olivine solution]",
    "MC3S": "MC3S [Ca3SiO5, monoclinic alite]",
    "TC3S": "TC3S [Ca3SiO5, triclinic alite]",
    "RC3S": "RC3S [Ca3SiO5, rhombohedral alite]",
    "RC3S2": "RC3S2 [Ca3Si2O7, calcium silicate]",
    "WCS": "WCS [CaSiO3, wollastonite]",
    "PCS": "PCS [CaSiO3, pseudowollastonite]",
    "WOLL": "WOLL [(Ca,Mg)SiO3, wollastonite solution]",
    "PROTO": "PROTO [(Ca,Mg)SiO3, protoenstatite type solution]",
    "M2S": "M2S [Mg2SiO4, forsterite]",
    "PMS": "PMS [MgSiO3, protoenstatite]",
    "OMS": "OMS [MgSiO3, orthoenstatite]",
    "CMS": "CMS [MgSiO3, clinoenstatite]",
    "CLINO": "CLINO [CaMgSi2O6, clinopyroxene or diopside solution]",
    "LOW_CLINO": "LOW_CLINO [CaMgSi2O6, low clinopyroxene solution]",
    "ORTHO": "ORTHO [CaMgSi2O6, orthopyroxene type solution]",
    "AKER": "AKER [Ca2MgSi2O7, akermanite]",
    "MER": "MER [Ca3MgSi2O8, merwinite]",
    "CAS2": "CAS2 [CaAl2Si2O8, anorthite]",
    "C2AS": "C2AS [Ca2Al2SiO7, gehlenite]",
    "MULLITE": "MULLITE [Al6Si2O13, mullite solution]",
    "CAO": "CAO [CaO, lime]",
    "MONOXIDE": "MONOXIDE [(Ca,Mg)O, lime periclase solution]",
    "C3A": "C3A [Ca3Al2O6, tricalcium aluminate]",
    "CA": "CA [CaAl2O4, calcium monoaluminate]",
    "CA2": "CA2 [CaAl4O7, calcium dialuminate]",
    "CA6": "CA6 [CaAl12O19, calcium hexaaluminate]",
    "C12A7": "C12A7 [Ca12Al14O33, mayenite]",
    "C2F": "C2F [Ca2Fe2O5, calcium ferrite]",
    "CF": "CF [CaFe2O4, calcium ferrite]",
    "CF2": "CF2 [CaFe4O7, calcium ferrite]",
    "FERRITE": "FERRITE [Ca2(Al,Fe)2O5, ferrite solution]",
    "CORUNDUM": "CORUNDUM [(Al,Fe)2O3, corundum solution]",
    "HALITE": "HALITE [(Fe2+,Fe3+,Va)O, wustite type iron oxide solution]",
    "SPINEL": "SPINEL [Fe3O4 based iron spinel]",
    "BCC_A2": "BCC_A2 [(Fe,O), bcc iron based solution]",
    "FCC_A1": "FCC_A1 [(Fe,O), fcc iron based solution]",
}


def inclusive_range(start, stop, step):
    if step <= 0:
        raise ValueError("Each composition increment must be greater than zero.")
    if stop < start:
        raise ValueError("Each composition maximum must not be below its minimum.")
    count = int(math.floor((stop - start) / step + 1.0e-10))
    return [round(start + index * step, 10) for index in range(count + 1)]


def generate_compositions():
    unknown = set(OXIDE_RANGES) - set(OXIDE_DEFINITIONS)
    if unknown:
        raise ValueError("Unsupported oxides: " + ", ".join(sorted(unknown)))
    if COMPOSITION_MODE not in {"BALANCE_COMPONENT", "FULL_GRID"}:
        raise ValueError("COMPOSITION_MODE must be BALANCE_COMPONENT or FULL_GRID.")
    if COMPOSITION_MODE == "BALANCE_COMPONENT" and BALANCE_OXIDE not in OXIDE_RANGES:
        raise ValueError("BALANCE_OXIDE must appear in OXIDE_RANGES.")

    oxides = list(OXIDE_RANGES)
    values = {
        oxide: inclusive_range(*OXIDE_RANGES[oxide])
        for oxide in oxides
    }
    tolerance = 1.0e-8
    compositions = []

    if COMPOSITION_MODE == "FULL_GRID":
        for combination in product(*(values[oxide] for oxide in oxides)):
            composition = dict(zip(oxides, combination))
            if abs(sum(composition.values()) - 100.0) <= tolerance:
                compositions.append(composition)
    else:
        scanned_oxides = [oxide for oxide in oxides if oxide != BALANCE_OXIDE]
        lower, upper, _ = OXIDE_RANGES[BALANCE_OXIDE]
        for combination in product(*(values[oxide] for oxide in scanned_oxides)):
            composition = dict(zip(scanned_oxides, combination))
            balance_value = 100.0 - sum(composition.values())
            if lower - tolerance <= balance_value <= upper + tolerance:
                composition[BALANCE_OXIDE] = round(balance_value, 10)
                compositions.append({oxide: composition[oxide] for oxide in oxides})

    if not compositions:
        raise ValueError("No valid compositions were generated. Check ranges and balance oxide.")
    return compositions, oxides


def oxide_wt_pct_to_component_mole_fractions(oxide_wt_pct):
    amounts = {}
    for oxide, wt_pct in oxide_wt_pct.items():
        if wt_pct <= 0:
            continue
        molar_mass, stoichiometry = OXIDE_DEFINITIONS[oxide]
        oxide_moles = wt_pct / molar_mass
        for component, coefficient in stoichiometry.items():
            amounts[component] = amounts.get(component, 0.0) + coefficient * oxide_moles
    total = sum(amounts.values())
    if total <= 0:
        raise ValueError("Composition contains no positive oxide amount.")
    return {component: amount / total for component, amount in amounts.items()}


def display_phase_name(name):
    return PHASE_LABELS.get(name, f"{name} [formula not mapped]")


def format_phases(stable_phases):
    values = [
        f"{display_phase_name(name)}: {amount:.5f}"
        for name, amount in stable_phases
        if amount >= REPORTING_PHASE_FRACTION_MIN
    ]
    return "; ".join(values) if values else "No phase above reporting threshold"


def equilibrium_at_temperature(dbf, components, phases, composition, temperature_c):
    real_components = [component for component in components if component != "VA"]
    dependent_component = real_components[-1]
    conditions = {v.P: PRESSURE_PA, v.T: temperature_c + 273.15, v.N: 1.0}
    for component in real_components:
        if component != dependent_component:
            conditions[v.X(component)] = composition.get(component, 0.0)

    problems = []
    for calc_opts in CALCULATION_OPTIONS_TRIALS:
        eq = equilibrium(dbf, components, phases, conditions, calc_opts=calc_opts)
        names = np.asarray(eq.Phase.values).ravel()
        amounts = np.asarray(eq.NP.values, dtype=float).ravel()
        valid = np.isfinite(amounts) & (names != "")
        stable_phases = [(str(name), float(amount)) for name, amount in zip(names[valid], amounts[valid])]
        total_phase_fraction = sum(amount for _, amount in stable_phases)
        if stable_phases and abs(total_phase_fraction - 1.0) <= 1.0e-5:
            liquid_fraction = sum(amount for name, amount in stable_phases if name == "LIQUID")
            return liquid_fraction, stable_phases
        problems.append(f"pdens={calc_opts['pdens']}, total NP={total_phase_fraction:.8f}")

    raise RuntimeError(
        f"Invalid equilibrium at {temperature_c:.3f} degC. " + "; ".join(problems)
    )


def find_first_crossing(scan_rows, threshold):
    for previous, current in zip(scan_rows[:-1], scan_rows[1:]):
        if previous[1] < threshold and current[1] >= threshold:
            return previous[0], current[0]
    return None


def refine_first_crossing(threshold, lower_c, upper_c, evaluate):
    lower = float(lower_c)
    upper = float(upper_c)
    while upper - lower > REFINEMENT_TOLERANCE_C:
        midpoint = 0.5 * (lower + upper)
        if evaluate(midpoint)[0] >= threshold:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def calculate_composition(dbf, oxide_composition):
    component_composition = oxide_wt_pct_to_component_mole_fractions(oxide_composition)
    components = sorted(component_composition) + ["VA"]
    phases = [phase for phase in filter_phases(dbf, components) if phase not in EXCLUDED_PHASES]
    cache = {}

    def evaluate(temperature_c):
        key = round(float(temperature_c), 8)
        if key not in cache:
            cache[key] = equilibrium_at_temperature(
                dbf, components, phases, component_composition, temperature_c
            )
        return cache[key]

    temperatures = np.arange(T_MIN_C, T_MAX_C + 0.5 * COARSE_STEP_C, COARSE_STEP_C)
    scan_rows = []
    for temperature_c in temperatures:
        liquid_fraction, stable_phases = evaluate(temperature_c)
        scan_rows.append((temperature_c, liquid_fraction, stable_phases))

    result = {}
    thresholds = {"first_liquid_degC": PHASE_FRACTION_TOLERANCE}
    thresholds.update({f"T{int(round(value * 100)):02d}_degC": value for value in LIQUID_FRACTION_MILESTONES})

    for column, threshold in thresholds.items():
        bracket = find_first_crossing(scan_rows, threshold)
        if bracket is None:
            result[column] = None
            result[column.replace("degC", "phases")] = "Not bracketed in selected range"
        else:
            temperature_c = refine_first_crossing(threshold, *bracket, evaluate)
            liquid_fraction, stable_phases = evaluate(temperature_c)
            result[column] = temperature_c
            result[column.replace("degC", "liquid_fraction")] = liquid_fraction
            result[column.replace("degC", "phases")] = format_phases(stable_phases)

    maximum = max(scan_rows, key=lambda row: row[1])
    result["maximum_liquid_fraction"] = maximum[1]
    result["maximum_liquid_temperature_degC"] = maximum[0]
    result["maximum_liquid_phases"] = format_phases(maximum[2])
    return result


def write_results(output_file, oxide_names, results):
    metric_columns = [
        "status", "message", "first_liquid_degC", "first_liquid_liquid_fraction", "first_liquid_phases",
        "T10_degC", "T10_liquid_fraction", "T10_phases",
        "T50_degC", "T50_liquid_fraction", "T50_phases",
        "T90_degC", "T90_liquid_fraction", "T90_phases",
        "T99_degC", "T99_liquid_fraction", "T99_phases",
        "maximum_liquid_fraction", "maximum_liquid_temperature_degC", "maximum_liquid_phases",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=oxide_names + metric_columns)
        writer.writeheader()
        writer.writerows(results)


def print_risk_summary(oxide_names, results):
    successful = [row for row in results if row["status"] == "OK" and row.get("first_liquid_degC") is not None]
    successful.sort(key=lambda row: row["first_liquid_degC"])

    print("\nMOST MELT SENSITIVE COMPOSITIONS")
    print("Composition is wt%. T99 is the practical liquidus criterion.")
    print("CaO     SiO2    Al2O3   Fe2O3  FeO     MgO     First liquid degC   T50 degC   T99 degC")
    print("=" * 100)
    for row in successful[:TOP_RESULTS_TO_PRINT]:
        values = [row.get(oxide, 0.0) for oxide in oxide_names]
        print(
            f"{values[0]:6.2f}  {values[1]:6.2f}  {values[2]:6.2f}  "
            f"{values[3]:6.2f}  {values[4]:6.2f}  {values[5]:6.2f}  "
            f"{row['first_liquid_degC']:17.2f}  "
            f"{row.get('T50_degC') or float('nan'):8.2f}  "
            f"{row.get('T99_degC') or float('nan'):8.2f}"
        )

    failures = [row for row in results if row["status"] != "OK"]
    print(f"\nSuccessful compositions: {len(successful)}")
    print(f"Failed compositions: {len(failures)}")
    if failures:
        print("Failed rows are retained in the CSV with the error message.")


def main():
    compositions, oxide_names = generate_compositions()
    dbf = Database(str(TDB_FILE))
    output_file = TDB_FILE.parent / "composition_melting_scan_results.csv"

    print(f"Generated valid compositions: {len(compositions)}")
    print(f"Temperature range: {T_MIN_C:.0f} to {T_MAX_C:.0f} degC")
    print(f"Temperature increment: {COARSE_STEP_C:.1f} degC")
    print(f"Pressure: {PRESSURE_PA:.0f} Pa")

    results = []
    for index, oxide_composition in enumerate(compositions, start=1):
        composition_text = ", ".join(f"{oxide}={value:.2f}" for oxide, value in oxide_composition.items())
        print(f"Calculating {index} of {len(compositions)}: {composition_text}")
        row = dict(oxide_composition)
        try:
            row.update(calculate_composition(dbf, oxide_composition))
            row["status"] = "OK"
            row["message"] = ""
        except Exception as error:
            row["status"] = "FAILED"
            row["message"] = str(error)
        results.append(row)

    write_results(output_file, oxide_names, results)
    print_risk_summary(oxide_names, results)
    print(f"\nFull results written to: {output_file.name}")


if __name__ == "__main__":
    main()
