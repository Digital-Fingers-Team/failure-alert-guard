"""Download Backblaze drive-stats quarters and compact them to one parquet per quarter.

Keeps every drive-day but only the SMART attributes used in the study, as float32.
The raw zip is deleted after conversion to save disk.
usage: python fetch_backblaze.py data_Q1_2024 [data_Q2_2024 ...]
"""
import os, sys, zipfile, subprocess
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/{}.zip"
OUT = "data/bb"
SMART = [1, 3, 4, 5, 7, 9, 10, 12, 183, 184, 187, 188, 189, 190, 192, 193, 194, 197, 198, 199, 240, 241, 242]
NUM = [f"smart_{i}_raw" for i in SMART]
META = ["date", "serial_number", "model", "capacity_bytes", "failure"]
SCHEMA = pa.schema([("date", pa.string()), ("serial_number", pa.string()), ("model", pa.string()),
                    ("capacity_bytes", pa.int64()), ("failure", pa.int8())] + [(c, pa.float32()) for c in NUM])


def convert(q):
    dst = f"{OUT}/{q}.parquet"
    if os.path.exists(dst):
        print(q, "exists"); return
    z = f"{OUT}/raw/{q}.zip"
    if not os.path.exists(z):
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", z + ".part", URL.format(q)], check=True)
        os.rename(z + ".part", z)
    names = sorted(n for n in zipfile.ZipFile(z).namelist()
                   if n.endswith(".csv") and "__MACOSX" not in n)
    w = pq.ParquetWriter(dst + ".part", SCHEMA, compression="zstd")
    with zipfile.ZipFile(z) as zf:
        for n in names:
            df = pd.read_csv(zf.open(n), usecols=lambda c: c in META or c in NUM, low_memory=False)
            df = df.reindex(columns=META + NUM)
            df["capacity_bytes"] = pd.to_numeric(df["capacity_bytes"], errors="coerce").fillna(-1).astype("int64")
            df["failure"] = df["failure"].fillna(0).astype("int8")
            df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce").astype("float32")
            df["date"] = df["date"].astype(str)
            w.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
            print(q, n, len(df), flush=True)
    w.close()
    os.rename(dst + ".part", dst)
    os.remove(z)


if __name__ == "__main__":
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    for q in sys.argv[1:]:
        convert(q)
