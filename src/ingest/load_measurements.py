"""Generate and load a synthetic dimensional inspection dataset.

WHY SYNTHETIC RATHER THAN A PUBLIC DATASET
------------------------------------------
The obvious public candidates (NASA C-MAPSS, UCI SECOM, Bosch Production Line)
are anonymised process-sensor streams, not dimensional metrology. Relabelling
"sensor_147" as "bore diameter" would be dressing up data as something it is
not, and any quality engineer interviewing you would spot it.

Generating the data instead is both more honest and a better demonstration:
this module models real process behaviour - tool wear drift within a batch,
per-machine bias, measurement uncertainty, unilateral form tolerances, and
occasional process excursions - at a target capability around Cpk 1.33.

Say exactly that in the README and in interviews. "I modelled the process"
beats "I downloaded a CSV".

Deterministic: SEED is fixed so the golden set's reference answers stay stable.
Change the seed and every reference_sql result in the eval set changes with it.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.db import connect

SEED = 20260920
START = datetime(2026, 1, 5, tzinfo=timezone.utc)
END = datetime(2026, 9, 15, tzinfo=timezone.utc)

MACHINES = ["CMM-01", "CMM-02", "CMM-03", "ARM-01"]
OPERATORS = ["R. Sharma", "A. Menon", "T. Iqbal", "S. Patel", "K. Rao"]

# Per-machine systematic bias, in mm. ARM-01 is an articulated arm: less
# accurate than the bridge CMMs, which is realistic and gives the data a
# genuine pattern worth discovering in the analysis.
MACHINE_BIAS = {"CMM-01": 0.000, "CMM-02": 0.002, "CMM-03": -0.001, "ARM-01": 0.004}
MACHINE_NOISE = {"CMM-01": 1.0, "CMM-02": 1.0, "CMM-03": 1.1, "ARM-01": 2.2}


@dataclass(frozen=True)
class Feature:
    name: str
    characteristic: str
    nominal: float
    lower_tol: float
    upper_tol: float
    cpk: float = 1.33

    @property
    def unilateral(self) -> bool:
        """Form and position tolerances are one-sided: nominal 0, upper only."""
        return self.lower_tol == 0.0 and self.nominal == 0.0


@dataclass(frozen=True)
class Part:
    part_id: str
    part_number: str
    description: str
    material: str
    drawing_rev: str
    features: tuple[Feature, ...]


PARTS: tuple[Part, ...] = (
    Part("P-1001", "1001-A", "Bearing Housing", "Aluminium 6061-T6", "C", (
        Feature("bore_dia_A", "diameter", 25.000, -0.021, 0.000, 1.33),
        Feature("bore_dia_B", "diameter", 32.000, -0.025, 0.000, 1.20),
        Feature("face_flatness", "flatness", 0.0, 0.0, 0.030, 1.10),
        Feature("bolt_hole_pos", "position", 0.0, 0.0, 0.100, 1.50),
    )),
    Part("P-1002", "1002-B", "Drive Shaft", "EN8 Steel", "B", (
        Feature("journal_dia", "diameter", 19.990, -0.013, 0.000, 1.45),
        Feature("shaft_runout", "roundness", 0.0, 0.0, 0.015, 1.05),
        Feature("keyway_pos", "position", 0.0, 0.0, 0.080, 1.40),
    )),
    Part("P-2010", "2010-D", "Valve Body", "Cast Iron GG25", "D", (
        Feature("seat_dia", "diameter", 48.000, -0.039, 0.000, 1.25),
        Feature("seat_flatness", "flatness", 0.0, 0.0, 0.025, 0.95),
        Feature("port_perp", "perpendicularity", 0.0, 0.0, 0.050, 1.30),
    )),
    Part("P-2011", "2011-A", "Valve Cover", "Aluminium 6082", "A", (
        Feature("gasket_flatness", "flatness", 0.0, 0.0, 0.040, 1.60),
        Feature("pilot_dia", "diameter", 60.000, -0.030, 0.000, 1.35),
    )),
    Part("P-3001", "3001-F", "Gear Blank", "20MnCr5", "F", (
        Feature("bore_dia", "diameter", 40.000, -0.025, 0.000, 1.38),
        Feature("face_runout", "roundness", 0.0, 0.0, 0.020, 1.15),
        Feature("od_dia", "diameter", 120.000, -0.054, 0.000, 1.28),
    )),
    Part("P-3002", "3002-B", "Pinion", "16MnCr5", "B", (
        Feature("pitch_dia", "diameter", 28.000, -0.021, 0.000, 1.42),
        Feature("tooth_pos", "position", 0.0, 0.0, 0.060, 1.20),
    )),
    Part("P-4417", "4417-C", "Mounting Bracket", "Mild Steel S275", "C", (
        Feature("mount_flatness", "flatness", 0.0, 0.0, 0.030, 0.90),
        Feature("hole_dia_1", "diameter", 10.500, -0.015, 0.015, 1.50),
        Feature("hole_pos_1", "position", 0.0, 0.0, 0.100, 1.35),
        Feature("edge_perp", "perpendicularity", 0.0, 0.0, 0.045, 1.25),
    )),
    Part("P-4418", "4418-A", "Flange Plate", "SS304", "A", (
        Feature("face_flatness", "flatness", 0.0, 0.0, 0.035, 1.20),
        Feature("pcd_pos", "position", 0.0, 0.0, 0.090, 1.30),
        Feature("thickness", "diameter", 12.000, -0.050, 0.050, 1.55),
    )),
)


def _sigma(feature: Feature) -> float:
    """Process sigma implied by the target Cpk and the tolerance band."""
    if feature.unilateral:
        # One-sided: Cpk = USL / (3 sigma), measuring from zero.
        return feature.upper_tol / (3.0 * feature.cpk)
    band = feature.upper_tol - feature.lower_tol
    return band / (6.0 * feature.cpk)


def _measure(rng: random.Random, feature: Feature, machine: str, wear: float) -> float:
    """One measured value, including machine bias, tool wear and noise."""
    sigma = _sigma(feature) * MACHINE_NOISE[machine]

    if feature.unilateral:
        # Form error is a magnitude: always >= 0, skewed toward small values.
        value = abs(rng.gauss(0.0, sigma)) + wear * feature.upper_tol * 0.35
        return round(max(0.0, value), 4)

    centre = feature.nominal + (feature.lower_tol + feature.upper_tol) / 2.0
    drift = wear * (feature.upper_tol - feature.lower_tol) * 0.30
    return round(rng.gauss(centre + MACHINE_BIAS[machine] + drift, sigma), 4)


def generate() -> tuple[list, list, list]:
    """Build parts, inspection_runs and measurements rows."""
    rng = random.Random(SEED)

    part_rows = [
        (p.part_id, p.part_number, p.description, p.material, p.drawing_rev) for p in PARTS
    ]

    run_rows: list[tuple] = []
    measurement_rows: list[tuple] = []
    run_id = 0
    total_days = (END - START).days

    for part in PARTS:
        # 6-10 batches per part across the period.
        for batch_index in range(rng.randint(6, 10)):
            batch_id = f"{part.part_id}-B{batch_index + 1:03d}"
            machine = rng.choice(MACHINES)
            operator = rng.choice(OPERATORS)
            batch_date = START + timedelta(
                days=rng.randint(0, total_days), hours=rng.randint(6, 20)
            )
            # ~8% of batches run degraded: worn tooling or a setup problem.
            excursion = rng.random() < 0.08
            batch_size = rng.randint(12, 30)

            for unit in range(batch_size):
                run_id += 1
                # Tool wear accumulates across the batch and resets at the next.
                wear = unit / max(1, batch_size - 1)
                if excursion:
                    wear += 0.8

                run_rows.append((
                    run_id, part.part_id, machine, operator, batch_id,
                    batch_date + timedelta(minutes=unit * rng.randint(3, 9)),
                ))

                for feature in part.features:
                    measurement_rows.append((
                        run_id, feature.name, feature.characteristic,
                        feature.nominal, feature.lower_tol, feature.upper_tol,
                        _measure(rng, feature, machine, wear),
                    ))

    return part_rows, run_rows, measurement_rows


def load() -> None:
    """Generate and load. Destructive: replaces existing inspection data."""
    part_rows, run_rows, measurement_rows = generate()

    with connect() as conn:
        # measurements -> inspection_runs -> parts, because of the FKs.
        conn.execute("TRUNCATE measurements, inspection_runs, parts RESTART IDENTITY CASCADE")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO parts (part_id, part_number, description, material, drawing_rev)"
                " VALUES (%s,%s,%s,%s,%s)", part_rows)
            cur.executemany(
                "INSERT INTO inspection_runs (run_id, part_id, machine_id, operator, batch_id,"
                " inspected_at) VALUES (%s,%s,%s,%s,%s,%s)", run_rows)
            cur.executemany(
                "INSERT INTO measurements (run_id, feature_name, characteristic, nominal,"
                " lower_tol, upper_tol, actual) VALUES (%s,%s,%s,%s,%s,%s,%s)", measurement_rows)
            # BIGSERIAL was bypassed by explicit run_id values; resync it.
            cur.execute("SELECT setval(pg_get_serial_sequence('inspection_runs','run_id'),"
                        " (SELECT MAX(run_id) FROM inspection_runs))")
        conn.commit()

    print(f"Loaded {len(part_rows)} parts, {len(run_rows)} runs, "
          f"{len(measurement_rows)} measurements.")


def summary() -> None:
    """Sanity check. A 2-8% fail rate per characteristic is realistic."""
    with connect() as conn:
        print("\ncharacteristic      n      fail_rate")
        for row in conn.execute(
            "SELECT characteristic, COUNT(*),"
            " ROUND(100.0 * COUNT(*) FILTER (WHERE NOT in_spec) / COUNT(*), 2)"
            " FROM measurements GROUP BY characteristic ORDER BY characteristic"
        ).fetchall():
            print(f"{row[0]:<18} {row[1]:>6}   {row[2]:>6}%")

        span = conn.execute(
            "SELECT MIN(inspected_at)::date, MAX(inspected_at)::date FROM inspection_runs"
        ).fetchone()
        print(f"\ndate range: {span[0]} to {span[1]}")


if __name__ == "__main__":
    load()
    summary()
