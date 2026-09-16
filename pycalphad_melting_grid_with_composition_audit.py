"""Equilibrium melting grid assessment using pycalphad.

Edit only the USER SETTINGS section for normal use. The script calculates global
thermodynamic equilibrium at each composition and temperature, reports liquid
phase amount milestones, and writes summary, scan, settings, and solver input
audit CSV files.
"""

from pathlib import Path
import csv
from decimal import Decimal
from itertools import product
import json

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases


# USER SETTINGS
# Thermodynamic database. This file must be in the same folder as this script.
# The TDB controls available phases, solution models, and all Gibbs energy data.
TDB_FILE = Path(__file__).with_name("C_A_S_Fe_O_M.tdb")

# Each oxide has the format: (start wt%, end wt%, increment wt%).
# If start equals end, the oxide is fixed and the increment is ignored.
# If more than one oxide varies, all combinations are calculated.
COMPOSITION_RANGES = {
    "CaO": (15.0, 15.0, 0.0),
    "SiO2": (77.0, 77.0, 1.0),
    "Al2O3": (5.0, 5.0, 0.0),
    "Fe2O3": (1.0, 1.0, 0.0),
    "FeO": (0.0, 0.0, 0.0),
    "MgO": (2.0, 2.0, 0.0),
}

# Constant pressure for every equilibrium state.
# 101325 Pa is one standard atmosphere, approximately 1.01325 bar.
PRESSURE_PA = 101325.0

# Coarse temperature scan limits, in degC. The range should encompass the
# first liquid boundary and the desired near complete melting criterion.
T_MIN_C = 1000.0
T_MAX_C = 1200.0
COARSE_STEP_C = 100.0

# Bisection endpoint tolerance, in degC, used after a liquid fraction target
# has been bracketed by the coarse scan.
REFINEMENT_TOLERANCE_C = 1.0

# Numerical liquid amount threshold used for first liquid and strict liquidus.
# First liquid is NP(LIQUID) >= this value.
# Strict liquidus is NP(LIQUID) >= 1 minus this value.
PHASE_FRACTION_TOLERANCE = 1.0e-6

# Engineering near complete melting threshold. T99 is used as practical liquidus.
PRACTICAL_LIQUIDUS_FRACTION = 0.99

# Liquid phase amount milestones reported in summary and CSV output.
LIQUID_FRACTION_MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

# Phases below this NP amount remain in the equilibrium calculation but are
# omitted from text reporting to keep phase assemblages readable.
REPORTING_PHASE_FRACTION_MIN = 1.0e-4

# Print every coarse temperature point for every composition if True.
# Keep False for large composition grids.
PRINT_EACH_GRID_SCAN = False

# Safety limit for the number of independently generated compositions.
MAX_CALCULATIONS = 500

# Output folder and files. Existing files with the same names are overwritten.
OUTPUT_DIRECTORY = Path("pycalphad_grid_results")
SUMMARY_FILE = OUTPUT_DIRECTORY / "pycalphad_melting_grid_summary.csv"
SCAN_FILE = OUTPUT_DIRECTORY / "pycalphad_melting_grid_scan.csv"
SETTINGS_FILE = OUTPUT_DIRECTORY / "pycalphad_melting_grid_settings.csv"
COMPOSITION_AUDIT_FILE = OUTPUT_DIRECTORY / "pycalphad_solver_input_compositions.csv"

# Exclude GAS for a closed condensed charge calculation. Remove GAS from this
# set only if your TDB and process model require gas phase equilibrium.
EXCLUDED_PHASES = {"GAS"}

# pycalphad calculation mesh density for phase model evaluation. A higher value
# can improve robustness around complex solution phases at increased runtime.
CALCULATION_OPTIONS = {"pdens": 500}


# INTERNAL CHEMISTRY MAPPING
# Oxide molar masses, g mol minus one, and conversion to the TDB component basis.
# Do not modify unless the component basis of the selected TDB is understood.
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
    "WCS": "WCS [CaSiO3, wollastonite]",
    "PCS": "PCS [CaSiO3, pseudowollastonite]",
    "WOLL": "WOLL [(Ca,Mg)SiO3, wollastonite solid solution]",
    "PROTO": "PROTO [(Ca,Mg)SiO3, protoenstatite type solid solution]",
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
    "FERRITE": "FERRITE [Ca2(Al,Fe)2O5, ferrite solid solution]",
    "CORUNDUM": "CORUNDUM [(Al,Fe)2O3, corundum solid solution]",
    "HALITE": "HALITE [(Fe2+,Fe3+,Va)O, wustite type iron oxide solution]",
    "SPINEL": "SPINEL [Fe3O4 based iron spinel]",
    "BCC_A2": "BCC_A2 [(Fe,O), bcc iron based solid solution]",
    "FCC_A1": "FCC_A1 [(Fe,O), fcc iron based solid solution]",
}


def oxide_values(start, end, step, oxide):
    start = Decimal(str(start))
    end = Decimal(str(end))
    step = Decimal(str(step))
    if start == end:
        return [float(start)]
    if step == 0 or (end - start) * step <= 0:
        raise ValueError(f"Invalid start, end, or step for {oxide}.")
    count = (end - start) / step
    if count != count.to_integral_value():
        raise ValueError(f"Range for {oxide} is not exactly divisible by its step.")
    return [float(start + index * step) for index in range(int(count) + 1)]


def composition_grid():
    oxides = list(COMPOSITION_RANGES)
    value_lists = [oxide_values(*COMPOSITION_RANGES[oxide], oxide) for oxide in oxides]
    for combination in product(*value_lists):
        yield dict(zip(oxides, combination))


def oxide_to_components(oxide_wt_pct):
    unknown = set(oxide_wt_pct) - set(OXIDE_DEFINITIONS)
    if unknown:
        raise ValueError("Unsupported oxides: " + ", ".join(sorted(unknown)))
    amounts = {}
    for oxide, value in oxide_wt_pct.items():
        if value < 0:
            raise ValueError(f"Negative value entered for {oxide}: {value}")
        if value == 0:
            continue
        molar_mass, stoichiometry = OXIDE_DEFINITIONS[oxide]
        moles = value / molar_mass
        for component, coefficient in stoichiometry.items():
            amounts[component] = amounts.get(component, 0.0) + coefficient * moles
    total = sum(amounts.values())
    if total <= 0:
        raise ValueError("At least one oxide value must be positive.")
    return {component: amount / total for component, amount in amounts.items()}


def normalized_solver_oxide_composition(raw_composition):
    """Normalize raw oxide wt% to 100 wt% without changing component ratios.

    pycalphad receives normalized TDB component mole fractions, not oxide wt%.
    This oxide normalization is an engineering readable equivalent of the exact
    same bulk composition supplied to the solver.
    """
    raw_total = sum(raw_composition.values())
    if raw_total <= 0:
        raise ValueError("Raw oxide total must be positive.")
    return {
        oxide: 100.0 * value / raw_total
        for oxide, value in raw_composition.items()
    }


def active_tdb_components(compositions):
    components = set()
    for composition in compositions:
        components.update(oxide_to_components(composition))
    return sorted(components) + ["VA"]


def equilibrium_state(dbf, components, phases, component_composition, temperature_c):
    real_components = [component for component in components if component != "VA"]
    dependent_component = real_components[-1]
    conditions = {v.P: PRESSURE_PA, v.T: temperature_c + 273.15, v.N: 1.0}
    for component in real_components:
        if component != dependent_component:
            conditions[v.X(component)] = component_composition.get(component, 0.0)

    result = equilibrium(dbf, components, phases, conditions, calc_opts=CALCULATION_OPTIONS)
    names = np.asarray(result.Phase.values).ravel()
    amounts = np.asarray(result.NP.values, dtype=float).ravel()
    valid = np.isfinite(amounts) & (names != "")
    stable_phases = [(str(name), float(amount)) for name, amount in zip(names[valid], amounts[valid])]
    liquid_fraction = sum(amount for name, amount in stable_phases if name == "LIQUID")
    return liquid_fraction, stable_phases


def find_bracket(scan_rows, target):
    for lower, upper in zip(scan_rows[:-1], scan_rows[1:]):
        if lower[1] < target <= upper[1]:
            return lower[0], upper[0]
    return None


def refine_crossing(target, lower_c, upper_c, evaluate):
    lower = float(lower_c)
    upper = float(upper_c)
    if evaluate(lower)[0] >= target or evaluate(upper)[0] < target:
        raise RuntimeError("Invalid transition bracket.")
    while upper - lower > REFINEMENT_TOLERANCE_C:
        midpoint = 0.5 * (lower + upper)
        if evaluate(midpoint)[0] >= target:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def phase_label(name):
    return PHASE_LABELS.get(name, f"{name} [formula or phase description not mapped]")


def phase_text(stable_phases):
    entries = [
        f"{phase_label(name)}: {amount:.5f}"
        for name, amount in stable_phases
        if amount >= REPORTING_PHASE_FRACTION_MIN
    ]
    return "; ".join(entries) if entries else "No phase above reporting threshold"


def summary_for_composition(composition_id, oxide_composition, dbf, components, phases):
    component_composition = oxide_to_components(oxide_composition)
    cache = {}

    def evaluate(temperature_c):
        key = round(float(temperature_c), 8)
        if key not in cache:
            cache[key] = equilibrium_state(dbf, components, phases, component_composition, temperature_c)
        return cache[key]

    temperatures = np.arange(T_MIN_C, T_MAX_C + 0.5 * COARSE_STEP_C, COARSE_STEP_C)
    scan_rows = []
    for temperature_c in temperatures:
        liquid_fraction, stable_phases = evaluate(temperature_c)
        scan_rows.append((float(temperature_c), liquid_fraction, stable_phases))
        if PRINT_EACH_GRID_SCAN:
            print(f" {temperature_c:8.1f} degC NP(LIQUID) = {liquid_fraction:.8f} {phase_text(stable_phases)}")

    def crossing(target):
        bracket = find_bracket(scan_rows, target)
        if bracket is None:
            return None, None, None
        temperature_c = refine_crossing(target, *bracket, evaluate)
        liquid_fraction, stable_phases = evaluate(temperature_c)
        return temperature_c, liquid_fraction, stable_phases

    first_temperature, _, first_phases = crossing(PHASE_FRACTION_TOLERANCE)
    milestones = {}
    for milestone in LIQUID_FRACTION_MILESTONES:
        temperature_c, _, _ = crossing(milestone)
        milestones[f"T{int(round(milestone * 100)):02d}_degC"] = temperature_c

    true_liquidus, _, true_liquidus_phases = crossing(1.0 - PHASE_FRACTION_TOLERANCE)
    maximum = max(scan_rows, key=lambda row: row[1])

    summary = {
        "composition_id": composition_id,
        **{oxide: oxide_composition.get(oxide, 0.0) for oxide in COMPOSITION_RANGES},
        "raw_oxide_total_wt_pct": sum(oxide_composition.values()),
        "first_liquid_boundary_degC": first_temperature,
        "scan_min_liquid_fraction_NP": scan_rows[0][1],
        **milestones,
        "practical_liquidus_degC": milestones["T99_degC"],
        "true_equilibrium_liquidus_degC": true_liquidus,
        "scan_max_liquid_fraction_NP": maximum[1],
        "first_liquid_phase_assemblage": phase_text(first_phases) if first_phases else "Not bracketed below scan range",
        "true_liquidus_phase_assemblage": phase_text(true_liquidus_phases) if true_liquidus_phases else "Not bracketed above scan range",
    }

    scan_records = []
    for temperature_c, liquid_fraction, stable_phases in scan_rows:
        scan_records.append({
            "composition_id": composition_id,
            **{oxide: oxide_composition.get(oxide, 0.0) for oxide in COMPOSITION_RANGES},
            "temperature_degC": temperature_c,
            "pressure_Pa": PRESSURE_PA,
            "liquid_fraction_NP": liquid_fraction,
            "nonliquid_fraction_NP": 1.0 - liquid_fraction,
            "stable_phases": phase_text(stable_phases),
        })
    return summary, scan_records


def failed_summary(composition_id, composition, error):
    milestones = {f"T{int(round(value * 100)):02d}_degC": None for value in LIQUID_FRACTION_MILESTONES}
    return {
        "composition_id": composition_id,
        **{oxide: composition.get(oxide, 0.0) for oxide in COMPOSITION_RANGES},
        "raw_oxide_total_wt_pct": sum(composition.values()),
        "first_liquid_boundary_degC": None,
        "scan_min_liquid_fraction_NP": None,
        **milestones,
        "practical_liquidus_degC": None,
        "true_equilibrium_liquidus_degC": None,
        "scan_max_liquid_fraction_NP": None,
        "first_liquid_phase_assemblage": f"FAILED: {type(error).__name__}: {error}",
        "true_liquidus_phase_assemblage": "Calculation failed",
    }


def print_summary_table(summaries):
    fields = [
        "composition_id", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "first_liquid_boundary_degC", "scan_min_liquid_fraction_NP", "T01_degC",
        "T10_degC", "T25_degC", "T50_degC", "T75_degC", "T90_degC",
        "T95_degC", "T99_degC", "true_equilibrium_liquidus_degC",
    ]
    labels = [
        "Point", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO", "First liquid",
        "Min liquid", "T01", "T10", "T25", "T50", "T75", "T90", "T95",
        "T99", "True liquidus",
    ]
    rows = []
    for summary in summaries:
        row = []
        for field in fields:
            value = summary[field]
            if field == "composition_id":
                row.append(str(value))
            elif value is None:
                row.append("Not found")
            elif "fraction" in field:
                row.append(f"{value:.5f}")
            else:
                row.append(f"{value:.2f}")
        rows.append(row)

    widths = [max(len(label), *(len(row[index]) for row in rows)) for index, label in enumerate(labels)]
    separator = "+" + "+".join("=" * (width + 2) for width in widths) + "+"
    print("\nPYCALPHAD MELTING GRID SUMMARY")
    print(separator)
    print("|" + "|".join(f" {label:^{width}} " for label, width in zip(labels, widths)) + "|")
    print(separator)
    for row in rows:
        print("|" + "|".join(f" {value:>{width}} " for value, width in zip(row, widths)) + "|")
    print(separator)
    print("Composition values are raw input oxide wt%.")
    print("Liquid fractions are pycalphad NP phase amounts, not automatically mass fractions.")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def write_composition_audit_csv(compositions, components):
    """Log raw oxide input, normalized oxide equivalent, and solver component fractions."""
    oxide_fields = list(COMPOSITION_RANGES)
    real_components = [component for component in components if component != "VA"]
    fieldnames = [
        "composition_id",
        "raw_oxide_total_wt_pct",
        "solver_equivalent_oxide_total_wt_pct",
    ]
    fieldnames.extend(f"raw_{oxide}_wt_pct" for oxide in oxide_fields)
    fieldnames.extend(f"solver_{oxide}_wt_pct" for oxide in oxide_fields)
    fieldnames.extend(f"solver_X_{component}" for component in real_components)

    with COMPOSITION_AUDIT_FILE.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for composition_id, raw_composition in enumerate(compositions, start=1):
            normalized_oxides = normalized_solver_oxide_composition(raw_composition)
            component_fractions = oxide_to_components(raw_composition)
            row = {
                "composition_id": composition_id,
                "raw_oxide_total_wt_pct": sum(raw_composition.values()),
                "solver_equivalent_oxide_total_wt_pct": sum(normalized_oxides.values()),
            }
            for oxide in oxide_fields:
                row[f"raw_{oxide}_wt_pct"] = raw_composition.get(oxide, 0.0)
                row[f"solver_{oxide}_wt_pct"] = normalized_oxides.get(oxide, 0.0)
            for component in real_components:
                row[f"solver_X_{component}"] = component_fractions.get(component, 0.0)
            writer.writerow(row)


def write_settings_csv(compositions, components, phases):
    rows = [
        ("Thermodynamic database", "TDB_FILE", str(TDB_FILE.resolve()), "TDB source defining phase Gibbs energies, phases, and solution models"),
        ("Pressure", "PRESSURE_PA", PRESSURE_PA, "Constant equilibrium pressure in Pa"),
        ("Temperature scan", "T_MIN_C", T_MIN_C, "Lower coarse scan temperature in degC"),
        ("Temperature scan", "T_MAX_C", T_MAX_C, "Upper coarse scan temperature in degC"),
        ("Temperature scan", "COARSE_STEP_C", COARSE_STEP_C, "Initial temperature spacing in degC"),
        ("Refinement", "REFINEMENT_TOLERANCE_C", REFINEMENT_TOLERANCE_C, "Bisection endpoint tolerance in degC"),
        ("Liquid criterion", "PHASE_FRACTION_TOLERANCE", PHASE_FRACTION_TOLERANCE, "Threshold for first liquid and strict liquidus definitions"),
        ("Liquid criterion", "PRACTICAL_LIQUIDUS_FRACTION", PRACTICAL_LIQUIDUS_FRACTION, "Engineering near complete melting fraction"),
        ("Liquid criterion", "LIQUID_FRACTION_MILESTONES", json.dumps(LIQUID_FRACTION_MILESTONES), "Reported NP liquid fraction milestones"),
        ("Reporting", "REPORTING_PHASE_FRACTION_MIN", REPORTING_PHASE_FRACTION_MIN, "Minimum NP phase amount displayed in phase text"),
        ("Reporting", "PRINT_EACH_GRID_SCAN", PRINT_EACH_GRID_SCAN, "Print every coarse grid temperature state"),
        ("Grid safety", "MAX_CALCULATIONS", MAX_CALCULATIONS, "Maximum number of generated bulk compositions"),
        ("Phases", "EXCLUDED_PHASES", json.dumps(sorted(EXCLUDED_PHASES)), "Phases explicitly omitted from equilibrium calculation"),
        ("Numerics", "CALCULATION_OPTIONS", json.dumps(CALCULATION_OPTIONS), "pycalphad equilibrium calculation options"),
        ("Grid", "COMPOSITION_RANGES", json.dumps(COMPOSITION_RANGES), "Raw oxide wt% composition ranges"),
        ("Grid", "COMPOSITION_POINT_COUNT", len(compositions), "Number of generated bulk compositions"),
        ("TDB system", "ACTIVE_COMPONENTS", json.dumps(components), "Union of active TDB components plus vacancy"),
        ("TDB system", "ALLOWED_PHASE_COUNT", len(phases), "Number of phases passed to pycalphad equilibrium"),
        ("TDB system", "ALLOWED_PHASES", json.dumps(phases), "Phase names available after component filtering and exclusions"),
        ("Output", "SUMMARY_FILE", str(SUMMARY_FILE), "One summary row per grid composition"),
        ("Output", "SCAN_FILE", str(SCAN_FILE), "Coarse temperature scan records for every grid composition"),
        ("Output", "SETTINGS_FILE", str(SETTINGS_FILE), "This settings record"),
        ("Output", "COMPOSITION_AUDIT_FILE", str(COMPOSITION_AUDIT_FILE), "Raw and normalized solver equivalent compositions"),
    ]
    with SETTINGS_FILE.open("w", newline="", encoding="utf_8") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "setting", "value", "meaning"])
        writer.writerows(rows)


def main():
    if not 0.0 < PRACTICAL_LIQUIDUS_FRACTION <= 1.0:
        raise ValueError("PRACTICAL_LIQUIDUS_FRACTION must be greater than zero and at most one.")
    if not TDB_FILE.exists():
        raise FileNotFoundError(f"TDB file not found: {TDB_FILE}")

    compositions = list(composition_grid())
    if len(compositions) > MAX_CALCULATIONS:
        raise ValueError(f"Grid contains {len(compositions)} points, above MAX_CALCULATIONS.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    dbf = Database(str(TDB_FILE))
    components = active_tdb_components(compositions)
    phases = [phase for phase in filter_phases(dbf, components) if phase not in EXCLUDED_PHASES]
    write_settings_csv(compositions, components, phases)
    write_composition_audit_csv(compositions, components)

    print("\nPYCALPHAD EQUILIBRIUM MELTING COMPOSITION GRID")
    print(f"TDB file: {TDB_FILE.name}")
    print(f"Pressure: {PRESSURE_PA:.0f} Pa")
    print(f"Composition points: {len(compositions)}")
    print(f"Temperature range: {T_MIN_C:.2f} to {T_MAX_C:.2f} degC")
    print(f"Coarse temperature step: {COARSE_STEP_C:.2f} degC")
    print(f"Refinement tolerance: {REFINEMENT_TOLERANCE_C:.3f} degC")
    print(f"Practical liquidus criterion: NP(LIQUID) >= {PRACTICAL_LIQUIDUS_FRACTION:.4f}")

    summaries = []
    all_scan_records = []
    for composition_id, composition in enumerate(compositions, start=1):
        print("\n" + "=" * 76)
        print(f"COMPOSITION POINT {composition_id} OF {len(compositions)}")
        print("=" * 76)
        print(" ".join(f"{oxide}={value:.3f}" for oxide, value in composition.items() if value != 0.0))
        try:
            summary, scan_records = summary_for_composition(composition_id, composition, dbf, components, phases)
            summaries.append(summary)
            all_scan_records.extend(scan_records)
        except Exception as error:
            summaries.append(failed_summary(composition_id, composition, error))
            print(f"Calculation failed: {type(error).__name__}: {error}")

    print_summary_table(summaries)
    write_csv(SUMMARY_FILE, summaries)
    write_csv(SCAN_FILE, all_scan_records)

    print("\nCSV files written")
    print(f" {SUMMARY_FILE}")
    print(f" {SCAN_FILE}")
    print(f" {SETTINGS_FILE}")
    print(f" {COMPOSITION_AUDIT_FILE}")


if __name__ == "__main__":
    main()
