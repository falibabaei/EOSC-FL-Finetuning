"""
Inspect the prepared EOSC FL dataset — distribution, stats, and readiness for FL.
"""
from pathlib import Path
import json
import csv

BASE = Path(__file__).parent

def inspect():
    print(f"{'='*60}")
    print(f"EOSC FL Dataset Inspection")
    print(f"{'='*60}\n")

    # 1. Directory structure
    clients = sorted([d for d in BASE.iterdir() if d.is_dir() and d.name.startswith("Client_")])
    print(f"Number of clients: {len(clients)}")
    for c in clients:
        subdirs = [d.name for d in c.iterdir() if d.is_dir() and not d.name.startswith("_")]
        n_chunks = len(list((c / "_chunks").glob("*.json"))) if (c / "_chunks").exists() else 0
        print(f"  {c.name}: subdirs={subdirs}, chunks={n_chunks}")

    # 2. Metadata
    csv_path = BASE / "metadata.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"\nMetadata: {len(rows)} entries")
        by_client = {}
        for r in rows:
            by_client.setdefault(r["client_id"], 0)
            by_client[r["client_id"]] += 1
        for cid, cnt in by_client.items():
            print(f"  {cid}: {cnt} entries")
    else:
        print("\nNo metadata.csv found.")

    # 3. Sample a chunk
    for c in clients:
        chunks_dir = c / "_chunks"
        if chunks_dir.exists():
            samples = list(chunks_dir.glob("*.json"))[:1]
            for s in samples:
                data = json.loads(s.read_text())
                print(f"\nSample chunk from {c.name}:")
                print(f"  instruction: {data.get('instruction','')[:80]}...")
                print(f"  output: {data.get('output','')[:120]}...")
            break


if __name__ == "__main__":
    inspect()
