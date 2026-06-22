"""
EOSC Federated Learning Simulation Runner

Sets up and runs an NVFlare simulation with 4 clients,
each fine-tuning a small LLM with LoRA on EOSC documents.

Uses FedJob API directly (not FedAvgRecipe) to avoid server-side
model serialization issues with HuggingFace models.

Usage:
  python run_fl_simulation.py
  python run_fl_simulation.py --rounds 3 --clients 2
"""
import argparse
from pathlib import Path

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

SCRIPT_DIR = Path(__file__).parent
CLIENT_SCRIPT = str(SCRIPT_DIR / "client.py")


def main():
    parser = argparse.ArgumentParser(description="Run EOSC FL LLM fine-tuning simulation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=4, choices=[2, 3, 4],
                        help="Number of clients")
    parser.add_argument("--workspace", type=str, default="/tmp/eosc_fl_workspace",
                        help="Simulator workspace directory")
    parser.add_argument("--threads", type=int, default=2,
                        help="Parallel client threads")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"EOSC Federated Learning Simulation")
    print(f"{'='*60}")
    print(f"  Model:       distilgpt2 (loaded per-client)")
    print(f"  Clients:     {args.clients}")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Workspace:   {args.workspace}")
    print(f"  Threads:     {args.threads}")
    print(f"{'='*60}\n")

    job = FedJob(name="eosc_llm_fl", min_clients=args.clients)

    controller = FedAvg(
        num_clients=args.clients,
        num_rounds=args.rounds,
        persistor_id="",
        model=None,
        task_name="train",
    )
    job.to_server(controller)

    runner = ScriptRunner(
        script=CLIENT_SCRIPT,
        framework=FrameworkType.NUMPY,
        server_expected_format=ExchangeFormat.NUMPY,
        params_transfer_type=TransferType.FULL,
    )
    job.to_clients(runner)

    job.simulator_run(
        workspace=args.workspace,
        n_clients=args.clients,
        threads=args.threads,
    )

    print(f"\n{'='*60}")
    print(f"Simulation complete. Workspace: {args.workspace}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
