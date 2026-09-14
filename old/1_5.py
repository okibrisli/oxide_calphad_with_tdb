"""Equilibrium melting assessment from oxide wt% using pycalphad.

Edit OXIDE_WT_PCT and the temperature settings. All displayed temperatures are
degC. The script prints a concise findings table and writes two CSV files:
melting_findings.csv and melting_scan.csv.
"""

from pathlib import Path
import csv
import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases


# USER INPUT
TDB_FILE = Path(__file__).with_name("C_A_S_Fe_O_M.tdb")

OXIDE_WT_PCT = {
    "CaO": 21,
    "SiO2": 69,
    "Al2O3": 5.0,
    "Fe2O3": 3.0,
    "FeO": 0.0,
    "MgO": 2.0,
}

PRESSURE_PA = 101325.0
T_MIN_C = 1000.0
T_MAX_C = 2200.0
COARSE_STEP_C = 50
REFINEMENT_TOLERANCE_C = 0.25
PHASE_FRACTION_TOLERANCE = 1.0e-6

# Engineering definition for almost fully molten material.
PRACTICAL_LIQUIDUS_FRACTION = 0.99

# Temperature points important for softening, flow, infiltration and melting.
LIQUID_FRACTION_MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

# Only phases at or above this fraction appear in the findings table.
REPORTING_PHASE_FRACTION_MIN = 1.0e-4
PRINT_FULL_SCAN_TO_TERMINAL = True

# Condensed charge calculation. GAS is deliberately excluded.
EXCLUDED_PHASES = {"GAS"}
CALCULATION_OPTIONS = {"pdens": 500}


# TDB component mapping. A, L, Q and M represent Al2O3, CaO, SiO2 and MgO.
OXIDE_DEFINITIONS = {
    "CaO": (56.077, {"L": 1.0}),
    "SiO2": (60.0843, {"Q": 1.0}),
    "Al2O3": (101.9613, {"A": 1.0}),
    "MgO": (40.3044, {"M": 1.0}),
    "FeO": (71.844, {"FE": 1.0, "O": 1.0}),
    "Fe2O3": (159.6882, {"FE": 2.0, "O": 3.0}),
    "Fe3O4": (231.531, {"FE": 3.0, "O": 4.0}),
}


def oxide_wt_pct_to_component_mole_fractions(oxide_wt_pct):
    unknown = set(oxide_wt_pct) - set(OXIDE_DEFINITIONS)
    if unknown:
        raise ValueError("Unsupported oxides: " + ", ".join(sorted(unknown)))

    amounts = {}
    total_mass = 0.0
    for oxide, wt_pct in oxide_wt_pct.items():
        if wt_pct < 0:
            raise ValueError(f"Negative value entered for {oxide}: {wt_pct}")
        if wt_pct == 0:
            continue
        molar_mass, stoichiometry = OXIDE_DEFINITIONS[oxide]
        oxide_moles = wt_pct / molar_mass
        total_mass += wt_pct
        for component, coefficient in stoichiometry.items():
            amounts[component] = amounts.get(component, 0.0) + coefficient * oxide_moles

    if total_mass <= 0:
        raise ValueError("At least one oxide value must be greater than zero.")

    total_amount = sum(amounts.values())
    return {component: amount / total_amount for component, amount in amounts.items()}


def equilibrium_at_temperature(dbf, components, phases, composition, temperature_c):
    real_components = [component for component in components if component != "VA"]
    dependent_component = real_components[-1]
    conditions = {v.P: PRESSURE_PA, v.T: temperature_c + 273.15, v.N: 1.0}

    for component in real_components:
        if component != dependent_component:
            conditions[v.X(component)] = composition.get(component, 0.0)

    eq = equilibrium(dbf, components, phases, conditions, calc_opts=CALCULATION_OPTIONS)
    names = np.asarray(eq.Phase.values).ravel()
    amounts = np.asarray(eq.NP.values, dtype=float).ravel()
    valid = np.isfinite(amounts) & (names != "")
    stable_phases = [(str(name), float(amount)) for name, amount in zip(names[valid], amounts[valid])]
    liquid_fraction = sum(amount for name, amount in stable_phases if name == "LIQUID")
    return liquid_fraction, stable_phases


def refine_first_crossing(threshold, lower_c, upper_c, evaluate):
    lower = float(lower_c)
    upper = float(upper_c)
    if evaluate(lower)[0] >= threshold or evaluate(upper)[0] < threshold:
        raise RuntimeError("Invalid bracket for transition refinement.")

    while upper - lower > REFINEMENT_TOLERANCE_C:
        midpoint = 0.5 * (lower + upper)
        if evaluate(midpoint)[0] >= threshold:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def find_first_crossing(results, threshold):
    for previous, current in zip(results[:-1], results[1:]):
        if previous[1] < threshold and current[1] >= threshold:
            return previous[0], current[0]
    return None


def format_phases(stable_phases):
    reported = [
        f"{name}: {amount:.5f}"
        for name, amount in stable_phases
        if amount >= REPORTING_PHASE_FRACTION_MIN
    ]
    return "; ".join(reported) if reported else "No phase above reporting threshold"


def print_table(rows):
    headers = ("Finding", "Temperature degC", "Liquid fraction", "Solid fraction", "Stable phases")
    widths = (27, 18, 17, 16, 0)
    print("\nKEY FINDINGS")
    print(f"{headers[0]:<{widths[0]}} {headers[1]:>{widths[1]}} {headers[2]:>{widths[2]}} {headers[3]:>{widths[3]}}   {headers[4]}")
    print("=" * 150)
    for row in rows:
        temperature = "Not found" if row["temperature_c"] is None else f"{row['temperature_c']:.2f}"
        liquid = "" if row["liquid_fraction"] is None else f"{row['liquid_fraction']:.6f}"
        solid = "" if row["liquid_fraction"] is None else f"{1.0 - row['liquid_fraction']:.6f}"
        print(f"{row['finding']:<{widths[0]}} {temperature:>{widths[1]}} {liquid:>{widths[2]}} {solid:>{widths[3]}}   {row['phases']}")


def write_csv_files(script_folder, findings, scan_rows):
    findings_file = script_folder / "melting_findings.csv"
    with findings_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("finding", "temperature_degC", "liquid_fraction", "nonliquid_fraction", "stable_phases"))
        writer.writeheader()
        for row in findings:
            writer.writerow({
                "finding": row["finding"],
                "temperature_degC": row["temperature_c"],
                "liquid_fraction": row["liquid_fraction"],
                "nonliquid_fraction": None if row["liquid_fraction"] is None else 1.0 - row["liquid_fraction"],
                "stable_phases": row["phases"],
            })

    scan_file = script_folder / "melting_scan.csv"
    with scan_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("temperature_degC", "liquid_fraction", "nonliquid_fraction", "stable_phases"))
        writer.writeheader()
        for temperature_c, liquid_fraction, stable_phases in scan_rows:
            writer.writerow({
                "temperature_degC": f"{temperature_c:.4f}",
                "liquid_fraction": f"{liquid_fraction:.10f}",
                "nonliquid_fraction": f"{1.0 - liquid_fraction:.10f}",
                "stable_phases": format_phases(stable_phases),
            })
    return findings_file, scan_file


def main():
    if not 0 < PRACTICAL_LIQUIDUS_FRACTION <= 1:
        raise ValueError("PRACTICAL_LIQUIDUS_FRACTION must be greater than zero and at most one.")
    if not TDB_FILE.exists():
        raise FileNotFoundError(f"TDB file not found: {TDB_FILE}")

    composition = oxide_wt_pct_to_component_mole_fractions(OXIDE_WT_PCT)
    dbf = Database(str(TDB_FILE))
    components = sorted(composition) + ["VA"]
    phases = [phase for phase in filter_phases(dbf, components) if phase not in EXCLUDED_PHASES]

    print("\nInput oxide composition, wt%")
    for oxide, value in OXIDE_WT_PCT.items():
        print(f"  {oxide:>6s} : {value:10.5f}")
    print(f"  Total  : {sum(OXIDE_WT_PCT.values()):10.5f}")
    print(f"  Pressure: {PRESSURE_PA:.0f} Pa")
    print(f"  Practical liquidus criterion: NP(LIQUID) >= {PRACTICAL_LIQUIDUS_FRACTION:.4f}")

    cache = {}

    def evaluate(temperature_c):
        key = round(float(temperature_c), 8)
        if key not in cache:
            cache[key] = equilibrium_at_temperature(
                dbf, components, phases, composition, temperature_c
            )
        return cache[key]

    scan_temperatures = np.arange(T_MIN_C, T_MAX_C + 0.5 * COARSE_STEP_C, COARSE_STEP_C)
    scan_rows = []
    print("\nCalculating equilibrium temperature scan")
    for temperature_c in scan_temperatures:
        liquid_fraction, stable_phases = evaluate(temperature_c)
        scan_rows.append((temperature_c, liquid_fraction, stable_phases))
        if PRINT_FULL_SCAN_TO_TERMINAL:
            print(f"  {temperature_c:8.1f} degC   NP(LIQUID) = {liquid_fraction:.8f}   {format_phases(stable_phases)}")

    findings = []
    first_liquid_bracket = find_first_crossing(scan_rows, PHASE_FRACTION_TOLERANCE)
    if first_liquid_bracket is None:
        findings.append({"finding": "First liquid boundary", "temperature_c": None, "liquid_fraction": None, "phases": "Not bracketed in selected range"})
    else:
        temperature_c = refine_first_crossing(PHASE_FRACTION_TOLERANCE, *first_liquid_bracket, evaluate)
        liquid_fraction, stable_phases = evaluate(temperature_c)
        findings.append({"finding": "First liquid boundary", "temperature_c": temperature_c, "liquid_fraction": liquid_fraction, "phases": format_phases(stable_phases)})

    for milestone in LIQUID_FRACTION_MILESTONES:
        bracket = find_first_crossing(scan_rows, milestone)
        label = f"T{int(round(milestone * 100)):02d}, liquid fraction"
        if bracket is None:
            findings.append({"finding": label, "temperature_c": None, "liquid_fraction": None, "phases": "Not bracketed in selected range"})
        else:
            temperature_c = refine_first_crossing(milestone, *bracket, evaluate)
            liquid_fraction, stable_phases = evaluate(temperature_c)
            findings.append({"finding": label, "temperature_c": temperature_c, "liquid_fraction": liquid_fraction, "phases": format_phases(stable_phases)})

    true_liquidus_threshold = 1.0 - PHASE_FRACTION_TOLERANCE
    true_liquidus_bracket = find_first_crossing(scan_rows, true_liquidus_threshold)
    if true_liquidus_bracket is None:
        findings.append({"finding": "True equilibrium liquidus", "temperature_c": None, "liquid_fraction": None, "phases": "Not reached. Use T99 as practical liquidus."})
    else:
        temperature_c = refine_first_crossing(true_liquidus_threshold, *true_liquidus_bracket, evaluate)
        liquid_fraction, stable_phases = evaluate(temperature_c)
        findings.append({"finding": "True equilibrium liquidus", "temperature_c": temperature_c, "liquid_fraction": liquid_fraction, "phases": format_phases(stable_phases)})

    maximum = max(scan_rows, key=lambda row: row[1])
    findings.append({"finding": "Maximum coarse scan liquid", "temperature_c": maximum[0], "liquid_fraction": maximum[1], "phases": format_phases(maximum[2])})

    print_table(findings)
    findings_file, scan_file = write_csv_files(TDB_FILE.parent, findings, scan_rows)
    print("\nCSV files written")
    print(f"  {findings_file.name}")
    print(f"  {scan_file.name}")


if __name__ == "__main__":
    main()
