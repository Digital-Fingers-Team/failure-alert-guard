"""Build drive-level snapshots from the compacted Backblaze quarters.

Periods (by snapshot day d): P1 = 2024-01..06 (fit), P2 = 2024-07..09 (calibrate),
P3 = 2024-10..12 (test). Labels look ahead up to W days (Q1 2025 is label-only).
For every period and W:
  risk    : one snapshot per drive whose first failure falls in the period, at a day drawn
            uniformly from the W days before failure (snapped to the latest observed day).
  uniform : one snapshot per drive seen in the period, at a day drawn uniformly from the
            days whose label is known (failed later, or still observed W days later).
at_risk = fails within W days of the snapshot. Features: SMART values at d and their
change since d-7 and d-30 (NULL when that day was not observed).
"""
import duckdb

W_LIST = [7, 30]
PERIODS = {"P1": ("2024-01-01", "2024-06-30"), "P2": ("2024-07-01", "2024-09-30"),
           "P3": ("2024-10-01", "2024-12-31")}
LEVEL = [1, 3, 4, 7, 9, 12, 190, 194, 241, 242]
COUNTER = [5, 10, 183, 184, 187, 188, 189, 192, 193, 197, 198, 199, 240]
SM = [f"smart_{i}_raw" for i in LEVEL + COUNTER]
SEED = 0.20241020

con = duckdb.connect("data/bb/work.duckdb")
con.execute("SET memory_limit='3GB'; SET threads=2; SET temp_directory='data/bb/tmp';"
            "SET preserve_insertion_order=false;")
con.execute(f"SELECT setseed({SEED})")
con.execute("""CREATE OR REPLACE VIEW d AS SELECT CAST(date AS DATE) AS date, serial_number AS serial,
               model, capacity_bytes, failure, * EXCLUDE (date, serial_number, model, capacity_bytes, failure)
               FROM read_parquet('data/bb/data_Q*.parquet')""")
# first failure wins; a few drives keep reporting after a failure record
con.execute("""CREATE OR REPLACE TABLE drives AS SELECT serial, any_value(model) AS model,
               any_value(capacity_bytes) AS capacity_bytes,
               min(date) AS first_date, max(date) AS last_date,
               min(CASE WHEN failure = 1 THEN date END) AS fail_date FROM d GROUP BY serial""")
print(con.execute("SELECT count(*), count(fail_date) FROM drives").fetchall(), flush=True)

parts = []
for W in W_LIST:
    for p, (lo, hi) in PERIODS.items():
        parts.append(f"""SELECT serial, '{p}' AS period, {W} AS W, 'risk' AS kind,
            CAST(fail_date - CAST(floor(random() * {W}) AS INTEGER) AS DATE) AS target
            FROM drives WHERE fail_date BETWEEN '{lo}' AND '{hi}'""")
        # last day whose label is known
        parts.append(f"""SELECT serial, '{p}' AS period, {W} AS W, 'uniform' AS kind,
            CAST(a + CAST(floor(random() * (datediff('day', a, b) + 1)) AS INTEGER) AS DATE) AS target
            FROM (SELECT serial, greatest(first_date, DATE '{lo}') AS a,
                         least(DATE '{hi}', coalesce(fail_date, last_date - {W})) AS b FROM drives)
            WHERE b >= a""")
con.execute("CREATE OR REPLACE TABLE snap0 AS " + " UNION ALL ".join(parts))
# snap each target to the latest observed day within 10 days before it (reporting gaps are rare)
con.execute("""CREATE OR REPLACE TABLE snap AS SELECT s.serial, s.period, s.W, s.kind, max(d.date) AS date
               FROM snap0 s JOIN d ON d.serial = s.serial
               AND d.date BETWEEN s.target - INTERVAL 10 DAY AND s.target
               GROUP BY s.serial, s.period, s.W, s.kind""")
print(con.execute("SELECT (SELECT count(*) FROM snap0), count(*) FROM snap").fetchall(), flush=True)
con.execute("""CREATE OR REPLACE TABLE keys AS
               SELECT DISTINCT serial, date AS d0, date - CAST(k AS INTEGER) AS date, k
               FROM snap, (VALUES (0), (7), (30)) t(k)""")
cols = ", ".join(f"d.{c}" for c in SM)
con.execute(f"""CREATE OR REPLACE TABLE rows AS SELECT k.serial, k.d0, k.k, {cols}
                FROM keys k JOIN d ON d.serial = k.serial AND d.date = k.date""")
cnt = [f"smart_{i}_raw" for i in COUNTER]
feat = ", ".join([f"r0.{c}" for c in SM] +
                 [f"r0.{c} - r{k}.{c} AS {c}_d{k}" for k in (7, 30) for c in cnt])
con.execute(f"""COPY (
    SELECT s.period, s.W, s.kind, s.serial, s.date, v.model, v.capacity_bytes,
           datediff('day', s.date, v.fail_date) AS ttf,
           CASE WHEN v.fail_date IS NOT NULL AND datediff('day', s.date, v.fail_date) < s.W
                THEN 1 ELSE 0 END AS at_risk, {feat}
    FROM snap s JOIN drives v USING (serial)
    JOIN rows r0 ON r0.serial = s.serial AND r0.d0 = s.date AND r0.k = 0
    LEFT JOIN rows r7 ON r7.serial = s.serial AND r7.d0 = s.date AND r7.k = 7
    LEFT JOIN rows r30 ON r30.serial = s.serial AND r30.d0 = s.date AND r30.k = 30
) TO 'data/bb/snapshots.parquet' (FORMAT parquet)""")
for r in con.execute("""SELECT period, W, kind, count(*), sum(at_risk) FROM 'data/bb/snapshots.parquet'
                        GROUP BY ALL ORDER BY ALL""").fetchall():
    print(r)
