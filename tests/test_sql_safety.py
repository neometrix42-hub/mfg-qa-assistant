"""Tests for the SQL validator. Run: pytest

These are free wins - the validator is pure and security-critical, so bugs here
are both likely and expensive. Extend this file as you add checks.
"""

import pytest

from src.tools.query_measurements import UnsafeSQL, validate


def test_plain_select_gets_a_limit():
    assert "LIMIT" in validate("SELECT * FROM parts")


def test_existing_limit_is_preserved():
    assert validate("SELECT * FROM parts LIMIT 10").endswith("LIMIT 10")


def test_cte_is_allowed():
    assert validate("WITH x AS (SELECT 1) SELECT * FROM x")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE parts",
        "SELECT 1; DROP TABLE parts",
        "INSERT INTO parts VALUES ('x')",
        "UPDATE parts SET material = 'x'",
        "SELECT 1 -- harmless\n; DELETE FROM parts",
        "",
    ],
)
def test_dangerous_statements_are_rejected(sql):
    with pytest.raises(UnsafeSQL):
        validate(sql)
