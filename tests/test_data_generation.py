"""Tests for the synthetic inspection dataset.

These matter more than they look. The golden set's reference_sql answers are
computed against this data, so if generation stops being deterministic or the
fail rates drift, every data-side eval number silently changes meaning.
"""

from src.ingest.load_measurements import PARTS, generate


def _in_spec(nominal, lower, upper, actual) -> bool:
    """Mirror of the generated column in sql/001_schema.sql."""
    return nominal + lower <= actual <= nominal + upper


def test_generation_is_deterministic():
    """The seed is fixed so reference_sql results stay stable. If this fails,
    every data-side expected answer in golden_set.yaml is now wrong."""
    assert generate() == generate()


def test_volumes_are_sane():
    parts, runs, measurements = generate()
    assert len(parts) == len(PARTS)
    assert 300 < len(runs) < 3000
    assert len(measurements) > len(runs)


def test_every_run_points_at_a_real_part():
    parts, runs, _ = generate()
    part_ids = {p[0] for p in parts}
    assert all(run[1] in part_ids for run in runs)


def test_every_measurement_points_at_a_real_run():
    _, runs, measurements = generate()
    run_ids = {r[0] for r in runs}
    assert all(m[0] in run_ids for m in measurements)


def test_form_tolerances_are_unilateral_and_non_negative():
    """Flatness, position, perpendicularity and roundness are magnitudes: a
    negative flatness value is physically meaningless and would make any
    tolerance query nonsense."""
    _, _, measurements = generate()
    form = {"flatness", "position", "perpendicularity", "roundness"}
    for _, _, characteristic, nominal, lower, _, actual in measurements:
        if characteristic in form:
            assert nominal == 0.0 and lower == 0.0
            assert actual >= 0.0


def test_fail_rate_is_realistic():
    """A dataset where nothing fails makes the non-conformance questions
    unanswerable; one where everything fails is not a real process."""
    _, _, measurements = generate()
    failures = sum(
        0 if _in_spec(nominal, lower, upper, actual) else 1
        for _, _, _, nominal, lower, upper, actual in measurements
    )
    rate = failures / len(measurements)
    assert 0.005 < rate < 0.20, f"fail rate {rate:.3%} is not a plausible process"


def test_date_range_covers_the_questions_in_the_golden_set():
    """Golden-set questions ask about specific months; the data must span them."""
    _, runs, _ = generate()
    dates = [run[5] for run in runs]
    assert min(dates).year == 2026
    months = {d.month for d in dates}
    assert 3 in months, "no March 2026 data, but the golden set asks about it"


def test_p4417_exists_because_the_golden_set_names_it():
    parts, runs, _ = generate()
    assert any(p[0] == "P-4417" for p in parts)
    assert any(r[1] == "P-4417" for r in runs)
