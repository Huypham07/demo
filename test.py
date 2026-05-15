import pandas as pd

file_path = "outputs/experiments/evidence/evidence_no_nli.parquet"

# Read parquet
df = pd.read_parquet(file_path)

# Drop column
df = df.drop(columns=["text"])

# Save back with same filename
df.to_parquet(file_path, index=False)

print("Done")