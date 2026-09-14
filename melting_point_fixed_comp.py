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
    "CaO": 48.3,
    "SiO2": 51.7,
    "Al2O3": 0.0,
    "Fe2O3": 0.0,
    "FeO": 0.0,
    "MgO": 0.0,
}

PRESSURE_PA = 101325.0
T_MIN_C = 1200.0
T_MAX_C = 1600.0
COARSE_STEP_C = 20.0
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


# Labels used in terminal output and both CSV files.
# Solution phases are written as formula ranges where appropriate.
PHASE_LABELS = {
    "LIQUID": "LIQUID [oxide melt solution]",

    "ACRIS": "ACRIS [SiO2, alpha cristobalite]",
    "BCRIS": "BCRIS [SiO2, beta cristobalite]",
    "AQUARTZ": "AQUARTZ [SiO2, alpha quartz]",
    "BQUARTZ": "BQUARTZ [SiO2, beta quartz]",

    "AC2S": "AC2S [Ca2SiO4, alpha belite]",
    "APC2S": "APC2S [Ca2SiO4, alpha prime belite]",
    "BC2S": "BC2S [Ca2SiO4, beta belite]",
    "GC2S": "GC2S [Ca2SiO4, gamma belite]",
    "ALPHA": "ALPHA [(Ca,Mg)2SiO4, alpha belite solid solution]",
    "ALPHA_PRIME": "ALPHA_PRIME [(Ca,Mg)2SiO4, alpha prime belite solid solution]",
    "OLIVINE": "OLIVINE [(Ca,Mg)2SiO4, olivine solid solution]",

    "MC3S": "MC3S [Ca3SiO5, monoclinic alite]",
    "TC3S": "TC3S [Ca3SiO5, triclinic alite]",
    "RC3S": "RC3S [Ca3SiO5, rhombohedral alite]",
    "RC3S2": "RC3S2 [Ca3Si2O7, calcium silicate]",

    "WCS": "WCS [CaSiO3, wollastonite]",
    "PCS": "PCS [CaSiO3, pseudowollastonite]",
    "WOLL": "WOLL [(Ca,Mg)SiO3, wollastonite solid solution]",
    "PROTO": "PROTO [(Ca,Mg)SiO3, protoenstatite type solid solution]",

    "M2S": "M2S [Mg2SiO4, forsterite]",
    "PMS": "PMS [MgSiO3, protoenstatite]",
    "OMS": "OMS [MgSiO3, orthoenstatite]",
    "CMS": "CMS [MgSiO3, clinoenstatite]",

    "CLINO": "CLINO [CaMgSi2O6, clinopyroxene or diopside solid solution]",
    "LOW_CLINO": "LOW_CLINO [CaMgSi2O6, low clinopyroxene solid solution]",
    "ORTHO": "ORTHO [CaMgSi2O6, orthopyroxene type solid solution]",

    "AKER": "AKER [Ca2MgSi2O7, akermanite]",
    "MER": "MER [Ca3MgSi2O8, merwinite]",

    "CAS2": "CAS2 [CaAl2Si2O8, anorthite]",
    "C2AS": "C2AS [Ca2Al2SiO7, gehlenite]",
    "MULLITE": "MULLITE [Al6Si2O13, mullite solid solution]",

    "CAO": "CAO [CaO, lime]",
    "MONOXIDE": "MONOXIDE [(Ca,Mg)O, lime periclase solid solution]",

    "C3A": "C3A [Ca3Al2O6, tricalcium aluminate]",
    "CA": "CA [CaAl2O4, calcium monoaluminate]",
    "CA2": "CA2 [CaAl4O7, calcium dialuminate]",
    "CA6": "CA6 [CaAl12O19, calcium hexaaluminate]",
    "C12A7": "C12A7 [Ca12Al14O33, mayenite]",

    "C2F": "C2F [Ca2Fe2O5, calcium ferrite]",
    "CF": "CF [CaFe2O4, calcium ferrite]",
    "CF2": "CF2 [CaFe4O7, calcium ferrite]",
    "FERRITE": "FERRITE [Ca2(Al,Fe)2O5, ferrite solid solution]",
    "CORUNDUM": "CORUNDUM [(Al,Fe)2O3, corundum solid solution]",

    "HALITE": "HALITE [(Fe2+,Fe3+,Va)O, wustite type iron oxide solution]",
    "SPINEL": "SPINEL [Fe3O4 based iron spinel]",
    "BCC_A2": "BCC_A2 [(Fe,O), bcc iron based solid solution]",
    "FCC_A1": "FCC_A1 [(Fe,O), fcc iron based solid solution]",

    "GAS": "GAS [gas phase]",
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


def display_phase_name(phase_name):
    return PHASE_LABELS.get(
        phase_name,
        f"{phase_name} [formula or phase description not mapped]"
    )


def format_phases(stable_phases):
    reported = [
        f"{display_phase_name(name)}: {amount:.5f}"
        for name, amount in stable_phases
        if amount >= REPORTING_PHASE_FRACTION_MIN
    ]
    return "; ".join(reported) if reported else "No phase above reporting threshold"


def print_table(rows):
    headers = ("Finding", "Temperature degC", "Liquid fraction", "Solid fraction", "Stable phases")
    widths = (27, 18, 17, 16, 0)
    print("\nKEY FINDINGS")
    print(f"{headers[0]:<{widths[0]}} {headers[1]:>{widths[1]}} {headers[2]:>{widths[2]}} {headers[3]:>{widths[3]}}   {headers[4]}")
    print("=" * 180)
    for row in rows:
        temperature = "Not found" if row["temperature_c"] is None else f"{row['temperature_c']:.2f}"
        liquid = "" if row["liquid_fraction"] is None else f"{row['liquid_fraction']:.6f}"
        solid = "" if row["liquid_fraction"] is None else f"{1.0 - row['liquid_fraction']:.6f}"
        print(f"{row['finding']:<{widths[0]}} {temperature:>{widths[1]}} {liquid:>{widths[2]}} {solid:>{widths[3]}}   {row['phases']}")


def write_csv_files(script_folder, findings, scan_rows):
    findings_file = script_folder / "melting_findings.csv"
    with findings_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("finding", "temperature_degC", "liquid_fraction", "nonliquid_fraction", "stable_phases")
        )
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
        writer = csv.DictWriter(
            file,
            fieldnames=("temperature_degC", "liquid_fraction", "nonliquid_fraction", "stable_phases")
        )
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
