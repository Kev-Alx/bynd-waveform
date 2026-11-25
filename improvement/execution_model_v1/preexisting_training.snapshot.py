"""Independent Mandarin surrogate fine-tuning; not an author-recovered recipe."""
from __future__ import annotations
import json
from pathlib import Path
import random
import torch
from tqdm import tqdm
from .data import audit_disjoint
from .io import atomic_json, digest, load_audio, read_jsonl, seed_everything, environment
from .models.whisper import load_whisper


def finetune(config, train_manifest, validation_manifest, output, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.json").exists():
        raise FileExistsError("Choose a fresh training output directory; automatic checkpoint overwrite is disabled")
    split_audit = audit_disjoint([train_manifest, validation_manifest])
    train, validation = read_jsonl(train_manifest), read_jsonl(validation_manifest)
    if any(r["corpus"] != "aishell" or r["split"] != "dev" or r["role"] != "finetune" for r in train):
        raise ValueError("Paper Mandarin surrogate must fine-tune on AISHELL dev, role=finetune")
    if any(r["role"] != "validation" or r["split"] == "test" for r in validation):
        raise ValueError("Use disjoint non-test validation records")
    if config["model"].get("language") != "zh":
        raise ValueError("Mandarin fine-tuning requires language=zh")
    seed_everything(config.get("seed", 2026))
    print("Loading trainable Whisper checkpoint...", flush=True)
    model, provenance = load_whisper(config["model"], device, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-5), weight_decay=config.get("weight_decay", 0.01))
    accumulation = int(config.get("gradient_accumulation", 8))
    epochs = int(config.get("epochs", 3))
    if accumulation < 1 or epochs < 1:
        raise ValueError("Positive accumulation and epochs required")
    history, best = [], float("inf")
    epoch_checkpoints = []
    atomic_json(output / "training.json", {"status": "running", "config": config, "split_audit": split_audit,
                "model": provenance, "environment": environment(), "status_of_recipe": "independent engineering choice"})
    for epoch in range(epochs):
        model.train()
        order = list(train)
        random.Random(config.get("seed", 2026) + epoch).shuffle(order)
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(tqdm(order, desc=f"Fine-tune epoch {epoch + 1}")):
            loss = model.supervised_loss(load_audio(row["audio"]).to(device), row["text"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(order) - group_start)
            (loss / group_size).backward()
            total += loss.item()
            if (index + 1) % accumulation == 0 or index + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            val = sum(model.supervised_loss(load_audio(r["audio"]).to(device), r["text"]).item()
                      for r in tqdm(validation, desc=f"Validating epoch {epoch + 1}", unit="utterances")) / len(validation)
        history.append({"epoch": epoch + 1, "training_loss": total / len(order), "validation_loss": val})
        state = {"dims": vars(model.model.dims), "model_state_dict": model.model.state_dict()}
        torch.save(state, output / "last.pt")
        torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch + 1, "config": config}, output / "optimizer.pt")
        if config.get("save_epoch_checkpoints", False):
            checkpoint = output / f"epoch-{epoch + 1:03d}.pt"
            torch.save(state, checkpoint)
            epoch_checkpoints.append({"epoch": epoch + 1, "path": str(checkpoint.resolve()),
                                      "sha256": digest(checkpoint)})
            atomic_json(output / "epoch-checkpoints.json", epoch_checkpoints)
        if val < best:
            best = val
            torch.save(state, output / "best.pt")
        atomic_json(output / "history.json", history)
    metadata = {"status": "complete", "config": config, "split_audit": split_audit,
                "model": provenance, "environment": environment(), "history": history,
                "best_sha256": digest(output / "best.pt"), "status_of_recipe": "independent engineering choice"}
    if config.get("save_epoch_checkpoints", False):
        metadata["epoch_checkpoints"] = epoch_checkpoints
    atomic_json(output / "training.json", metadata)
