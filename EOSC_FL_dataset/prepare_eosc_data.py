"""
EOSC Federated Learning Dataset Preparation Script

Downloads and prepares EOSC documents for each FL client:
  Client_1: EOSC Association (policies, governance, task_force_reports)
  Client_2: EU Commission (strategy, implementation, roadmaps)
  Client_3: EOSC Projects (EOSC_Focus, EOSC_Future, AI4EOSC)
  Client_4: Academic (surveys, research_papers)

Usage:
  python prepare_eosc_data.py                   # process all clients
  python prepare_eosc_data.py --client 1         # only client 1
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
CHUNK_SIZE = 512

CLIENT_SOURCES = {
    "Client_1_EOSC_Association": {
        "policies": [
            ("https://eosc.eu/wp-content/uploads/2024/12/20241031_SRIA_1.3_final_Annex.pdf",
             "EOSC_SRIA_1.3.pdf"),
            ("https://eosc.eu/wp-content/uploads/2023/12/20231114_SRIA_1.2_final2.pdf",
             "EOSC_SRIA_1.2.pdf"),
        ],
        "governance": [
            ("https://eosc.eu/wp-content/uploads/2023/08/SRIA-1.1-final.pdf",
             "EOSC_SRIA_1.1.pdf"),
            ("https://eosc.eu/eosc-about/sria-mar",
             "eosc_sria_overview.html"),
        ],
        "task_force_reports": [
            ("https://eosc.eu/wp-content/uploads/2024/11/20241031_SRIA_1.3_final_Annex_to-be-approved.pdf",
             "EOSC_SRIA_1.3_draft.pdf"),
        ],
    },
    "Client_2_EU_Commission": {
        "strategy": [
            ("https://research-and-innovation.ec.europa.eu/strategy/strategy-research-and-innovation/our-digital-future/open-science/european-open-science-cloud-eosc_en",
             "eosc_strategy.html"),
            ("https://digital-strategy.ec.europa.eu/en/policies/open-science-cloud",
             "eosc_digital_strategy.html"),
        ],
        "implementation": [
            ("https://eosc.eu/", "eosc_eu_home.html"),
        ],
        "roadmaps": [
            ("https://eur-lex.europa.eu/resource.html?uri=cellar:f097c300-e758-11ee-9ea8-01aa75ed71a1.0002.02/DOC_2",
             "horizon_europe_strategic_plan_2025_2027.pdf"),
        ],
    },
    "Client_3_EOSC_Projects": {
        "EOSC_Focus": [
            ("https://cordis.europa.eu/project/id/101058432", "EOSC_Focus.html"),
        ],
        "EOSC_Future": [
            ("https://cordis.europa.eu/project/id/101017536", "EOSC_Future.html"),
        ],
        "AI4EOSC": [
            ("https://cordis.europa.eu/project/id/101058593", "AI4EOSC.html"),
        ],
    },
    "Client_4_Academic": {
        "surveys": [
            ("https://api.openalex.org/works?filter=title.search:European+Open+Science+Cloud+survey&sort=cited_by_count:desc&per_page=25",
             "survey_results.json"),
        ],
        "research_papers": [
            ("https://api.openalex.org/works?filter=title.search:European+Open+Science+Cloud&sort=cited_by_count:desc&per_page=50",
             "eosc_papers.json"),
        ],
    },
}


def download_file(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, timeout=60,
                          headers={"User-Agent": "EOSC-FL/1.0"},
                          stream=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        dest.unlink(missing_ok=True)
        tqdm.write(f"  Failed: {url} — {e}")
        return False


def extract_text_pdf(pdf_path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages if p.extract_text())
    except Exception as e:
        return f"[PDF extraction failed for {pdf_path.name}: {e}]"


def extract_text_html(html_path: Path) -> str:
    try:
        import trafilatura
        text = trafilatura.extract(html_path.read_text(encoding="utf-8", errors="replace"))
        return text or html_path.read_text(encoding="utf-8", errors="replace")
    except ImportError:
        return html_path.read_text(encoding="utf-8", errors="replace")


def extract_text_json(json_path: Path) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    texts = []
    for result in data.get("results", []):
        title = result.get("title", "")
        abstract = result.get("abstract_inverted_index", "")
        if isinstance(abstract, dict):
            words = {}
            for word, positions in abstract.items():
                for pos in positions:
                    words[pos] = word
            abstract = " ".join(words.get(i, "") for i in range(max(words) + 1)) if words else ""
        if title:
            texts.append(f"Title: {title}\n\nAbstract: {abstract}")
    return "\n\n---\n\n".join(texts) if texts else json.dumps(data, indent=2)


def chunk_text(text: str, target_size: int = CHUNK_SIZE) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), target_size):
        chunk = " ".join(words[i:i + target_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks


def format_as_instruction(chunk: str, category: str) -> dict:
    return {
        "instruction": f"Explain the following {category} aspect of EOSC.",
        "input": "",
        "output": chunk,
    }


def prepare_client(client_name: str, sources: dict, metadata_rows: list):
    client_dir = BASE_DIR / client_name
    chunks_dir = client_dir / "_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for category, urls in sources.items():
        cat_dir = client_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for url, fname in urls:
            dest = cat_dir / fname
            ok = download_file(url, dest)
            if not ok:
                continue

            if fname.endswith(".pdf"):
                text = extract_text_pdf(dest)
            elif fname.endswith(".json"):
                text = extract_text_json(dest)
            elif fname.endswith(".html"):
                text = extract_text_html(dest)
            else:
                text = dest.read_text(encoding="utf-8", errors="replace")

            chunks = chunk_text(text)

            for idx, chunk in enumerate(chunks):
                record = format_as_instruction(chunk, category)
                chunk_file = chunks_dir / f"{dest.stem}_chunk_{idx:04d}.json"
                chunk_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                all_chunks.append(record)

            metadata_rows.append({
                "client_id": client_name,
                "filename": dest.name,
                "category": category,
                "subcategory": "",
                "source_url": url,
                "token_count": len(text.split()),
                "num_chunks": len(chunks),
                "date": time.strftime("%Y-%m-%d"),
            })

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Prepare EOSC FL dataset")
    parser.add_argument("--client", type=int, choices=[1, 2, 3, 4],
                        help="Only process a specific client (1-4)")
    args = parser.parse_args()

    client_names = list(CLIENT_SOURCES.keys())
    if args.client:
        client_names = [client_names[args.client - 1]]

    metadata_rows = []
    total_chunks = 0

    for client_name in client_names:
        src = CLIENT_SOURCES[client_name]
        print(f"\n{'=' * 60}")
        print(f"Processing {client_name}")
        print(f"{'=' * 60}")
        chunks = prepare_client(client_name, src, metadata_rows)
        n = len(chunks)
        total_chunks += n
        print(f"  -> {n} chunks created")

    print(f"\nTotal chunks: {total_chunks}")

    csv_path = BASE_DIR / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "client_id", "filename", "category", "subcategory",
            "source_url", "token_count", "num_chunks", "date"
        ])
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"\nMetadata written to {csv_path}")


if __name__ == "__main__":
    main()
