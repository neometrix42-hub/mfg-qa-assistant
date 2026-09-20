"""Load a public manufacturing dataset into parts / inspection_runs / measurements.

YOU WRITE THIS ONE. It is ordinary ETL, and it is genuinely worth describing in
interviews - "I reshaped a public sensor dataset into a normalised inspection
schema" is real data engineering, which is the role you are actually targeting.

Pick ONE source and put it in data/raw/:
  - NASA C-MAPSS turbofan degradation  (clean, well documented, time series)
  - UCI SECOM                          (591 sensors, pass/fail labels)
  - Bosch Production Line Performance  (Kaggle, large, genuinely industrial)

DO NOT use Neometrix client data. Not in this repo, not in a branch, not locally.
"""

from src.db import connect


def load() -> None:
    """Read the raw dataset, reshape it, insert it.

    TODO:
      1. Read the raw file from data/raw/.
      2. Invent a part catalogue - map each unit/lot to a part_id and part_number.
      3. Map each observation to an inspection_run (machine_id, operator,
         batch_id, inspected_at). Spread inspected_at over a realistic date range;
         your golden set asks month-based questions, so the dates must be usable.
      4. Map each sensor reading to a measurement row with a plausible
         characteristic ('diameter' / 'flatness' / 'position'), nominal, and
         signed lower_tol / upper_tol.
      5. Bulk insert with psycopg's copy() - row-by-row INSERT will be painfully
         slow at this volume.

    Do NOT insert deviation or in_spec. They are generated columns; the database
    computes them. Attempting to insert them raises an error.

    Sanity check when done - roughly 2-8% out of spec is realistic:
      SELECT characteristic,
             COUNT(*) FILTER (WHERE NOT in_spec)::float / COUNT(*) AS fail_rate
      FROM measurements GROUP BY characteristic;
    """
    raise NotImplementedError("Week 2: implement the loader")


if __name__ == "__main__":
    load()
    with connect() as conn:
        for table in ("parts", "inspection_runs", "measurements"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count} rows")
