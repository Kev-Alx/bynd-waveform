"""Resolve and download a CTC model, recording its immutable HF revision."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm
from beyondwaveform.io import atomic_json, digest

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", required=True)
parser.add_argument("--revision", default="main")
parser.add_argument("--output", required=True)
parser.add_argument("--cache-dir", help="Keep downloaded snapshots in a chosen writable artifact directory")
args = parser.parse_args()
print(f"Resolving immutable revision for {args.model_id}...", flush=True)
info = HfApi().model_info(args.model_id, revision=args.revision, files_metadata=True)
revision = info.sha
files = {f.rfilename: {"bytes": f.size, "git_blob_sha1": f.blob_id,
                      "lfs_sha256": f.lfs.sha256 if f.lfs else None}
         for f in info.siblings if f.rfilename.endswith((".json", ".txt", ".safetensors"))
         or f.rfilename == "pytorch_model.bin"}
print(f"Downloading {args.model_id}@{revision} (Hugging Face shows file progress)...", flush=True)
path = Path(snapshot_download(args.model_id, revision=revision,
                             cache_dir=args.cache_dir, allow_patterns=sorted(files)))
hashes = {}
for filename, expected in tqdm(files.items(), desc="Verifying pinned CTC files", unit="files"):
    file = path / filename
    if file.stat().st_size != expected["bytes"]:
        raise ValueError(f"Downloaded size mismatch: {filename}")
    actual = digest(file)
    if expected["lfs_sha256"]:
        if actual != expected["lfs_sha256"]:
            raise ValueError(f"Downloaded LFS SHA256 mismatch: {filename}")
    else:
        contents = file.read_bytes()
        blob = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
        if blob != expected["git_blob_sha1"]:
            raise ValueError(f"Downloaded Git blob mismatch: {filename}")
    hashes[filename] = actual
config = {"name": args.model_id, "kind": "ctc", "model_id": str(path.resolve()), "revision": revision,
          "repo_id": args.model_id, "language": "en", "sample_rate": 16000,
          "decoding": {"kind": "greedy", "external_language_model": False}, "artifact_hashes": hashes}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
receipt = {"repo": args.model_id, "revision": revision, "snapshot": str(path.resolve()),
           "files": hashes, "upstream_files": files,
           "source": f"https://huggingface.co/{args.model_id}/tree/{revision}"}
receipt_path = Path(str(output) + ".receipt.json")
if receipt_path.exists():
    if json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Existing CTC receipt differs; use another output")
else:
    atomic_json(receipt_path, receipt)
if output.exists():
    if yaml.safe_load(output.read_text()) != config:
        raise ValueError("Existing CTC config differs; use another output")
else:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(output)
print(f"Saved pinned target config to {output}")
