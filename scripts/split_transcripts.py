"""
Split the combined transcript markdown file into individual .txt files.

Reads:  ./call_transcripts_starter_sample.md
Writes: ./transcripts/call_001_bank1_legitimate.txt  ... etc

Run from the project root:
    cd ~/scam-detection
    python scripts/split_transcripts.py
"""

import re
from pathlib import Path

SOURCE_FILE = Path("./call_transcripts_starter_sample.md")
TRANSCRIPTS_DIR = Path("./transcripts")


def main():
    if not SOURCE_FILE.exists():
        print(f"ERROR: {SOURCE_FILE} not found.")
        print("Run this from ~/scam-detection (the folder containing the .md file).")
        return

    text = SOURCE_FILE.read_text(encoding="utf-8")
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        r"^##\s+(call_\d+_\w+\.txt)\s*\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    matches = pattern.findall(text)
    if not matches:
        print("ERROR: no transcript sections found.")
        print("Expected headings like:  ## call_001_bank1_legitimate.txt")
        return

    written = 0
    for filename, body in matches:
        body = re.sub(r"^GROUND TRUTH LABEL:.*$", "", body, flags=re.MULTILINE)
        body = re.sub(r"^---+\s*$", "", body, flags=re.MULTILINE)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()

        if not body:
            print(f"  SKIP {filename} (empty after cleaning)")
            continue

        out_path = TRANSCRIPTS_DIR / filename
        out_path.write_text(body + "\n", encoding="utf-8")
        n_lines = len(body.splitlines())
        print(f"  created {out_path}  ({n_lines} lines)")
        written += 1

    print(f"\nDone. {written} transcript files written to {TRANSCRIPTS_DIR}/")

    labels_file = Path("./labels.csv")
    if labels_file.exists():
        import csv
        with open(labels_file, encoding="utf-8") as f:
            label_names = {row["filename"] for row in csv.DictReader(f)}
        written_names = {p.name for p in TRANSCRIPTS_DIR.glob("*.txt")}

        missing = label_names - written_names
        extra = written_names - label_names

        print("\nCross-check against labels.csv:")
        print(f"  labels.csv lists:      {len(label_names)} files")
        print(f"  transcripts/ contains: {len(written_names)} files")
        if missing:
            print(f"  WARNING - in labels.csv but no transcript: {sorted(missing)}")
        if extra:
            print(f"  WARNING - transcript exists but not in labels.csv: {sorted(extra)}")
        if not missing and not extra:
            print("  OK - filenames match exactly.")


if __name__ == "__main__":
    main()
