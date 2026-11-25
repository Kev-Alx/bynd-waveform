"""Optimizer-aware continuation with a wall-clock budget and durable checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from tqdm import tqdm

from .data import audit_disjoint
from .io import atomic_json, digest, environment, load_audio, read_jsonl, seed_everything
from .models.whisper import load_whisper


def atomic_torch_save(path, state):
    path = Path(path)
    temporary = path.with_suffix(".saving.pt")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def resume_finetune(resume_path, output, max_seconds=18000, save_reserve_seconds=600,
                    device="cuda", progress=None, clock=time.monotonic):
    """Continue through the configured final epoch; checkpoint before the runtime limit.

    Stop checks run at optimizer boundaries, so partial gradient accumulation is
    never discarded or serialized as if it had updated the model. A stopped
    continuation can itself be resumed through its new resume-state.json.
    """
    started = clock()
    if not 0 < save_reserve_seconds < max_seconds:
        raise ValueError("Saving reserve must be positive and shorter than the runtime limit")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.json").exists():
        raise FileExistsError("Use a fresh continuation directory")
    source = json.loads(Path(resume_path).read_text())
    for field in ["training_config", "train_manifest", "validation_manifest"]:
        if digest(source[field]) != source[field + "_sha256"]:
            raise ValueError(f"Changed continuation input: {field}")
    if digest(source["checkpoint"]["path"]) != source["checkpoint"]["sha256"]:
        raise ValueError("Changed model checkpoint")
    optimizer_path = source.get("bundle", source.get("optimizer"))
    optimizer_sha = source.get("bundle_sha256", source.get("optimizer_sha256"))
    if digest(optimizer_path) != optimizer_sha:
        raise ValueError("Changed optimizer checkpoint")
    config = json.loads(Path(source["training_config"]).read_text())
    train = read_jsonl(source["train_manifest"])
    validation = read_jsonl(source["validation_manifest"])
    if any(r["corpus"] != "aishell" or r["split"] != "dev" or r["role"] != "finetune" for r in train):
        raise ValueError("Fine-tuning must retain the AISHELL development manifest")
    if any(r["role"] != "validation" or r["split"] == "test" for r in validation):
        raise ValueError("Checkpoint validation cannot use official test data")
    split_audit = audit_disjoint([source["train_manifest"], source["validation_manifest"]])
    accumulation = int(config.get("gradient_accumulation", 8))
    epochs = int(config["epochs"])
    if accumulation < 1 or config["model"].get("language") != "zh":
        raise ValueError("Invalid Mandarin continuation configuration")
    history = list(source["history"])
    cursor = dict(source.get("cursor", {"epoch": source["next_epoch"], "train_position": 0,
                                         "training_total": 0., "validation_position": 0, "validation_total": 0.}))
    if cursor["epoch"] > epochs or (history and history[-1]["epoch"] >= epochs):
        raise ValueError("The configured final epoch has already completed")
    if cursor["train_position"] % accumulation and cursor["train_position"] != len(train):
        raise ValueError("Training cursor is not an optimizer boundary")
    seed_everything(config.get("seed", 2026))
    stop = {"reason": None}
    previous_handlers = {}

    def request_stop(signum, frame):
        stop["reason"] = f"signal-{signum}"

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        previous_handlers[sig] = signal.signal(sig, request_stop)

    def should_stop():
        if clock() - started >= max_seconds - save_reserve_seconds:
            stop["reason"] = stop["reason"] or "runtime-limit-saving-reserve"
        return stop["reason"] is not None

    def emit(phase, **extra):
        state = {"status": "running", "phase": phase, "epoch": cursor["epoch"], "target_epochs": epochs,
                 "train_position": cursor["train_position"], "train_count": len(train),
                 "validation_position": cursor["validation_position"], "validation_count": len(validation),
                 "elapsed_seconds": clock() - started, "max_runtime_seconds": max_seconds,
                 "save_reserve_seconds": save_reserve_seconds, **extra}
        if progress:
            progress(state)

    metadata = {"status": "running", "config": config, "resumed_from": str(Path(resume_path).resolve()),
                "source_resume_sha256": digest(resume_path), "split_audit": split_audit,
                "max_runtime_seconds": max_seconds, "save_reserve_seconds": save_reserve_seconds,
                "rng_continuation": "restored" if source.get("bundle") else "original epoch 2 RNG was not saved; seeded continuation",
                "history": history}
    atomic_json(output / "training.json", metadata)
    try:
        emit("restoring-epoch-checkpoint-and-optimizer")
        model_config = {**config["model"], "checkpoint": source["checkpoint"]["path"]}
        model, provenance = load_whisper(model_config, device, trainable=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-5),
                                     weight_decay=config.get("weight_decay", .01))
        saved = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        if not source.get("bundle") and saved["epoch"] != source["completed_epochs"]:
            raise ValueError("Model and optimizer epochs disagree")
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            if group["lr"] != config.get("learning_rate", 1e-5) or group["weight_decay"] != config.get("weight_decay", .01):
                raise ValueError("Saved optimizer settings differ from the frozen recipe")
        if source.get("bundle"):
            random.setstate(saved["rng_state"]["python"])
            np.random.set_state(saved["rng_state"]["numpy"])
            torch.set_rng_state(saved["rng_state"]["torch"])
            if str(device).startswith("cuda"):
                torch.cuda.set_rng_state_all(saved["rng_state"]["cuda"])
        del saved
        metadata.update(model=provenance, environment=environment(), optimizer_restored=True)
        atomic_json(output / "training.json", metadata)
        print(f"Restored model and AdamW optimizer; continuing epoch {cursor['epoch']} of {epochs} at recording {cursor['train_position']}.", flush=True)
        last_emit = 0.
        while cursor["epoch"] <= epochs and not should_stop():
            epoch = cursor["epoch"]
            order = list(train)
            random.Random(config.get("seed", 2026) + epoch - 1).shuffle(order)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pending_loss = 0.
            for index in tqdm(range(cursor["train_position"], len(order)), desc=f"Fine-tune epoch {epoch}"):
                row = order[index]
                loss = model.supervised_loss(load_audio(row["audio"]).to(device), row["text"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite continuation loss")
                group_start = (index // accumulation) * accumulation
                group_size = min(accumulation, len(order) - group_start)
                (loss / group_size).backward()
                pending_loss += loss.item()
                if (index + 1) % accumulation == 0 or index + 1 == len(order):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    cursor["train_position"] = index + 1
                    cursor["training_total"] += pending_loss
                    pending_loss = 0.
                    if clock() - last_emit >= 5:
                        emit(f"training-epoch-{epoch}")
                        last_emit = clock()
                    if should_stop():
                        break
            if should_stop():
                break
            model.eval()
            emit(f"loss-validation-epoch-{epoch}")
            with torch.no_grad():
                for index in tqdm(range(cursor["validation_position"], len(validation)), desc=f"Validating epoch {epoch}"):
                    if should_stop():
                        break
                    row = validation[index]
                    loss = model.supervised_loss(load_audio(row["audio"]).to(device), row["text"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite validation loss")
                    cursor["validation_total"] += loss.item()
                    cursor["validation_position"] = index + 1
                    if clock() - last_emit >= 5:
                        emit(f"loss-validation-epoch-{epoch}")
                        last_emit = clock()
            if cursor["validation_position"] != len(validation):
                break
            history.append({"epoch": epoch, "training_loss": cursor["training_total"] / len(train),
                            "validation_loss": cursor["validation_total"] / len(validation)})
            if epoch == epochs:
                break
            cursor = {"epoch": epoch + 1, "train_position": 0, "training_total": 0.,
                      "validation_position": 0, "validation_total": 0.}
        complete = bool(history and history[-1]["epoch"] == epochs)
        emit("saving-final-checkpoint" if complete else "saving-before-runtime-stop", stop_reason=stop["reason"])
        state = {"dims": vars(model.model.dims), "model_state_dict": model.model.state_dict()}
        bundle = {**state, "optimizer": optimizer.state_dict(), "config": config, "cursor": cursor,
                  "history": history, "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                  "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []}}
        atomic_torch_save(output / "continuation.pt", bundle)
        atomic_torch_save(output / "model.pt", state)
        checkpoint = {"epoch": cursor["epoch"], "path": str((output / "model.pt").resolve()),
                      "sha256": digest(output / "model.pt")}
        resume = {key: source[key] for field in ["training_config", "train_manifest", "validation_manifest"]
                  for key in [field, field + "_sha256"]}
        resume.update(status="complete" if complete else "saved; not scheduled", completed_epochs=history[-1]["epoch"] if history else 0,
                      next_epoch=epochs + 1 if complete else cursor["epoch"], checkpoint=checkpoint, cursor=cursor,
                      bundle=str((output / "continuation.pt").resolve()), bundle_sha256=digest(output / "continuation.pt"),
                      history=history, automatic_followup_allowed=False, rng_state_saved=True)
        atomic_json(output / "resume-state.json", resume)
        atomic_json(output / "history.json", history)
        metadata.update(status="complete" if complete else "stopped", completed_epochs=resume["completed_epochs"],
                        history=history, cursor=cursor, checkpoint=checkpoint, resume_state=str((output / "resume-state.json").resolve()),
                        stop_reason=stop["reason"], elapsed_seconds=clock() - started)
        atomic_json(output / "training.json", metadata)
        emit("complete" if complete else "stopped-with-resumable-checkpoint", status=metadata["status"],
             checkpoint=checkpoint, resume_state=metadata["resume_state"], completed_epochs=resume["completed_epochs"], stop_reason=stop["reason"])
        return metadata
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
