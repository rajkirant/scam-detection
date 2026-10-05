from datasets import load_dataset
import os

print("=" * 70)
print("DOWNLOADING DATASET FROM HUGGING FACE HUB...")
print("=" * 70)

dataset = load_dataset("yevgeniy03/home-telecom-callcenter-transcripts-viewer")
print(f"✓ Dataset downloaded: {len(dataset['train'])} examples")

print("\n" + "=" * 70)
print("CONVERTING TO PANDAS DATAFRAME...")
print("=" * 70)

df = dataset['train'].to_pandas()
print(f"✓ Shape: {df.shape}")

print("\n" + "=" * 70)
print("SAVING AS CSV...")
print("=" * 70)

output_file = "call_center_transcripts.csv"
df.to_csv(output_file, index=False, encoding='utf-8')

file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"✓ Saved: {output_file}")
print(f"  Size: {file_size_mb:.2f} MB")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")

print("\n✓ COMPLETE!")
EOF
