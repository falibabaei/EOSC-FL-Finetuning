"""
EOSC FL Client - LLM fine-tuning with LoRA using NVFlare Client API.

Weights are exchanged as numpy arrays (no model on server).
Each client:
  1. Loads distilgpt2 from HuggingFace locally
  2. Receives global weights as numpy dict from server
  3. Applies LoRA, fine-tunes on local EOSC data
  4. Merges LoRA, sends merged weights back as numpy dict
"""
import json
import os
from pathlib import Path

import nvflare.client as flare
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator

MODEL_NAME = "distilgpt2"
BLOCK_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 1
LR = 5e-5

EOSC_DATA_DIR = Path(__file__).resolve().parent.parent / "EOSC_FL_dataset"

CLIENT_IDS = {
    "site-1": "Client_1_EOSC_Association",
    "site-2": "Client_2_EU_Commission",
    "site-3": "Client_3_EOSC_Projects",
    "site-4": "Client_4_Academic",
}


def get_site_name() -> str:
    name = flare.get_site_name()
    if name:
        return name
    return os.environ.get("NVFLARE_SITE_NAME", "site-1")


def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


class EOSCChunkDataset(Dataset):
    def __init__(self, chunk_dir: Path, tokenizer, block_size: int = BLOCK_SIZE):
        self.examples = []
        chunk_files = sorted(chunk_dir.glob("*.json"))
        if not chunk_files:
            return
        for fpath in chunk_files:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            text = f"Instruction: {data.get('instruction', '')}\nOutput: {data.get('output', '')}"
            enc = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_local_data(site_name: str, tokenizer) -> DataLoader:
    dir_name = CLIENT_IDS.get(site_name, site_name)
    chunk_dir = EOSC_DATA_DIR / dir_name / "_chunks"
    if not chunk_dir.exists():
        for cid in CLIENT_IDS.values():
            cd = EOSC_DATA_DIR / cid / "_chunks"
            if cd.exists():
                chunk_dir = cd
                break
        else:
            raise FileNotFoundError(f"No EOSC data found in {EOSC_DATA_DIR}")
    dataset = EOSCChunkDataset(chunk_dir, tokenizer)
    print(f"  Loaded {len(dataset)} samples for {site_name}")
    if len(dataset) == 0:
        raise ValueError(f"No data loaded for {site_name}")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)


def numpy_to_torch(numpy_params: dict, model: torch.nn.Module):
    own_state = model.state_dict()
    for name, param in numpy_params.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))


def torch_to_numpy(state_dict: dict) -> dict:
    return {k: v.cpu().numpy() for k, v in state_dict.items()}


def train():
    flare.init()
    site_name = get_site_name()
    print(f"\n{'='*50}")
    print(f"Client: {site_name}")
    print(f"{'='*50}")

    tokenizer, base_model = load_tokenizer_and_model()
    train_loader = get_local_data(site_name, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        input_model = flare.receive()
        if input_model is None:
            break

        print(f"  --- Round {input_model.current_round} ---")

        # Load global weights into base model
        if input_model.params:
            numpy_to_torch(input_model.params, base_model)

        # Apply LoRA and train
        model = apply_lora(base_model)
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            total_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"    Epoch {epoch + 1}/{EPOCHS} loss: {avg_loss:.4f}")

        # Merge LoRA and send back merged weights as numpy
        merged_model = model.merge_and_unload()
        numpy_params = torch_to_numpy(merged_model.state_dict())

        print(f"  Sending {len(numpy_params)} weight tensors, loss={avg_loss:.4f}")
        flare.send(flare.FLModel(params=numpy_params, metrics={"loss": avg_loss}))


if __name__ == "__main__":
    train()
