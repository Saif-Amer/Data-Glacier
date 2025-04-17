import os
import time
import pandas as pd
import dask.dataframe as dd
import modin.pandas as mpd
import yaml

# --------- CONFIGURATION ---------
INPUT_FILE = "yellow-tripdata-2023-01.csv"
OUTPUT_FILE = "nyc_output.txt.gz"
SCHEMA_FILE = "schema.yaml"
SEPARATOR = "|"

# --------- CLEANING FUNCTION ---------
def clean_columns(columns):
    return [col.strip().replace(" ", "_").replace("#", "")
            .replace("/", "_").replace("$", "") for col in columns]

# --------- BENCHMARKING ---------
def benchmark_read(method, file_path):
    start = time.time()
    if method == 'pandas':
        df = pd.read_csv(file_path)
    elif method == 'dask':
        df = dd.read_csv(file_path).compute()
    elif method == 'modin':
        df = mpd.read_csv(file_path)
    duration = time.time() - start
    print(f"[{method.upper()}] Loaded in {duration:.2f} seconds")
    return df

# --------- STEP 1: Read with Pandas ---------
print("Reading file with pandas...")
df = benchmark_read('pandas', INPUT_FILE)

# --------- STEP 2: Clean column names ---------
print("Cleaning column names...")
df.columns = clean_columns(df.columns)

# --------- STEP 3: Write YAML schema ---------
print("Writing schema.yaml...")
schema = {
    'separator': SEPARATOR,
    'columns': df.columns.tolist()
}
with open(SCHEMA_FILE, 'w') as file:
    yaml.dump(schema, file)

# --------- STEP 4: Validate schema ---------
print("Validating schema...")
with open(SCHEMA_FILE) as file:
    loaded_schema = yaml.safe_load(file)

assert df.columns.tolist() == loaded_schema['columns'], "Schema validation failed!"
print("Schema validation passed.")

# --------- STEP 5: Write pipe-separated GZ file ---------
print(f"Writing to compressed output file: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, sep=SEPARATOR, index=False, compression='gzip')

# --------- STEP 6: Summary ---------
print("Generating summary...")
summary = {
    'Total Rows': len(df),
    'Total Columns': len(df.columns),
    'File Size (MB)': round(os.path.getsize(OUTPUT_FILE) / (1024 * 1024), 2)
}
print("Summary:")
for key, value in summary.items():
    print(f"- {key}: {value}")

# --------- STEP 7: (Optional) Benchmark Others ---------
print("\nBenchmarking other libraries for reading:")
benchmark_read('dask', INPUT_FILE)
benchmark_read('modin', INPUT_FILE)
