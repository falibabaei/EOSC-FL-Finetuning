# EOSC Federated Learning - LLM Fine-tuning

Federated fine-tuning of a small LLM (distilgpt2) on EOSC (European Open Science Cloud) documents using NVFlare.

## Project Structure

```
FL_llm_finetuning/
├── run_fl_simulation.py    # Sets up and launches the NVFlare FL job
├── client.py               # Client-side training script (LoRA fine-tuning)
├── run.sh                  # Slurm job submission script
└── requirements.txt        # Python dependencies

EOSC_FL_dataset/
├── prepare_eosc_data.py    # Downloads and chunks EOSC documents
├── inspect_dataset.py      # Verifies dataset readiness
└── metadata.csv            # Tracks all document provenance
```

## Dataset

4 clients, each with a distinct EOSC document category:

| Client | Topic | Sources |
|---|---|---|
| Client_1 | EOSC Association | SRIA v1.1-v1.3 PDFs, governance docs |
| Client_2 | EU Commission | Strategy pages, Horizon Europe plan |
| Client_3 | EOSC Projects | CORDIS project pages (EOSC-Focus, EOSC-Future, AI4EOSC) |
| Client_4 | Academic | OpenAlex API (surveys + papers) |

## Data Preparation

Download source documents and generate chunked datasets:

```bash
source venv/bin/activate
pip install pdfplumber trafilatura requests tqdm

# Prepare all clients
python EOSC_FL_dataset/prepare_eosc_data.py

# Or prepare a single client (1-4)
python EOSC_FL_dataset/prepare_eosc_data.py --client 1

# Verify the prepared dataset
python EOSC_FL_dataset/inspect_dataset.py
```

Source documents are downloaded to each client's subdirectory. Text is extracted, split into ~512-token chunks, and saved as instruction-format JSON files in `_chunks/`. Metadata is recorded in `metadata.csv`.

## Setup

```bash
# Create and activate venv (one-time)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r FL_llm_finetuning/requirements.txt
```

## Usage

### Local simulation (testing)

```bash
source venv/bin/activate
python FL_llm_finetuning/run_fl_simulation.py --rounds 2 --clients 2 --threads 1
```

### HPC with Slurm

```bash
sbatch FL_llm_finetuning/run.sh
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--rounds` | 3 | Number of FL rounds |
| `--clients` | 4 | Number of clients (2-4) |
| `--threads` | 2 | Parallel client threads |
| `--workspace` | `/tmp/eosc_fl_workspace` | Simulator working directory |

## How it works

1. **Server** (NVFlare FedAvg) coordinates rounds, no model architecture stored server-side
2. **Each client** loads distilgpt2 locally, applies LoRA adapters (`r=8`, target: `c_attn`), fine-tunes on its own EOSC documents
3. **Weights** are merged, converted to numpy, and exchanged with the server
4. **Server** averages weights (FedAvg) and distributes to clients for next round

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- NVFlare 2.8+
- transformers, peft, datasets, accelerate
