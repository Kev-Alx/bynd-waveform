"""Explicit downloads, immutable source revisions, and content receipts."""
from __future__ import annotations
import json
import hashlib
import concurrent.futures
import shutil
import tarfile
import time
from pathlib import Path
import requests
from tqdm import tqdm
from .io import atomic_json, digest

KNN_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"
UTMOS_REVISION = "ed25eacbfa42b99156c36ebec67a733b5dbb9b79"


def download(url, destination, expected=None, algorithm="sha256"):
    """Resume partial responses, verify published checksums when provided.

    A local SHA256 receipt detects later corruption even for upstream assets
    without a published checksum. It is not an upstream authenticity claim.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists():
        if expected and digest(destination, algorithm) != expected:
            raise ValueError(f"Checksum mismatch: {destination}")
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior["url"] != url or digest(destination) != prior["sha256"]:
                raise ValueError(f"Artifact receipt mismatch: {destination}")
            return destination
        if not expected:
            raise ValueError(f"Existing unverified artifact: {destination}")
    else:
        # Never mix byte ranges from different mirrors/URLs.
        url_key = hashlib.sha256(url.encode()).hexdigest()[:12]
        partial = destination.with_suffix(destination.suffix + f".{url_key}.partial")
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with requests.get(url, headers=headers,
                                  stream=True, timeout=(30, 120)) as response:
                    if response.status_code == 416 and offset:
                        # A completed or stale partial cannot be resumed. Restart
                        # this downloader-owned temporary file cleanly.
                        partial.unlink()
                        raise IOError("Stale partial download; restarting")
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server resumed at an unexpected byte offset")
                    # requests transparently decompresses HTTP content encoding;
                    # Content-Length then describes wire bytes, not yielded bytes.
                    length = int(response.headers.get("Content-Length", 0)) if not response.headers.get("Content-Encoding") else 0
                    received = 0
                    with partial.open("ab" if append else "wb") as stream, tqdm(
                        total=length or None, unit="B", unit_scale=True, desc=destination.name
                    ) as progress:
                        for chunk in response.iter_content(2**20):
                            stream.write(chunk)
                            received += len(chunk)
                            progress.update(len(chunk))
                    if length and received != length:
                        raise IOError("Incomplete download")
                break
            except (requests.RequestException, IOError):
                if attempt == 3:
                    raise
                time.sleep(min(2**attempt, 8))
        if expected and digest(partial, algorithm) != expected:
            raise ValueError(f"Downloaded checksum mismatch: {partial}")
        partial.replace(destination)
    atomic_json(receipt, {"url": url, "sha256": digest(destination), "bytes": destination.stat().st_size,
                          "upstream_checksum": expected, "upstream_checksum_algorithm": algorithm if expected else None})
    return destination


def download_ranged(url, destination, workers=8, part_bytes=16 * 2**20):
    """Download large public checkpoints by validated byte ranges, resumably."""
    destination = Path(destination)
    receipt = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists():
        if not receipt.exists() or json.loads(receipt.read_text()).get("sha256") != digest(destination):
            raise ValueError(f"Existing checkpoint has no matching receipt: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                      stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Encoding"):
            raise ValueError("Server must support uncompressed HTTP byte ranges")
        value = response.headers.get("Content-Range", "")
        if not value.startswith("bytes 0-0/"):
            raise ValueError("Server returned an unexpected content range")
        total = int(value.split("/")[-1])
    key = hashlib.sha256(url.encode()).hexdigest()[:12]
    workdir = destination.parent / f".{destination.name}.{key}.parts"
    workdir.mkdir(exist_ok=True)
    jobs = [(i, start, min(total - 1, start + part_bytes - 1))
            for i, start in enumerate(range(0, total, part_bytes))]

    def get_part(job):
        index, start, end = job
        path = workdir / f"{index:05d}.part"
        expected_bytes = end - start + 1
        if path.exists() and path.stat().st_size == expected_bytes:
            return path
        for attempt in range(4):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                                  stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}" or response.headers.get("Content-Encoding"):
                        raise ValueError("Checkpoint server returned wrong byte range")
                    tmp = path.with_suffix(".tmp")
                    with tmp.open("wb") as output:
                        for chunk in response.iter_content(2**20):
                            output.write(chunk)
                if tmp.stat().st_size != expected_bytes:
                    raise IOError("Incomplete checkpoint chunk")
                tmp.replace(path)
                return path
            except (requests.RequestException, IOError):
                if attempt == 3:
                    raise
                time.sleep(min(2**attempt, 8))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(tqdm(pool.map(get_part, jobs), total=len(jobs), desc=destination.name + " parts"))
    combined = destination.with_suffix(destination.suffix + ".assembling")
    with combined.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)
    if combined.stat().st_size != total:
        raise IOError("Assembled checkpoint has wrong length")
    combined.replace(destination)
    atomic_json(receipt, {"url": url, "sha256": digest(destination), "bytes": total,
                          "upstream_checksum": None, "download_method": "validated HTTP ranges"})
    for part in parts:
        part.unlink()
    workdir.rmdir()
    return destination


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = []
        for member in tqdm(tar, desc=f"Checking {Path(archive).name}", unit="files"):
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive entry: {member.name}")
            members.append(member)
        for member in tqdm(members, desc=f"Extracting {Path(archive).name}", unit="files"):
            tar.extract(member, destination)


def download_repository(repo, revision, root):
    root = Path(root)
    name = repo.split("/")[-1]
    folder = root / "sources" / f"{name}-{revision}"
    archive = download(f"https://codeload.github.com/{repo}/tar.gz/{revision}", root / "downloads" / f"{name}-{revision}.tar.gz")
    if not (folder / ".complete.json").exists():
        safe_extract(archive, root / "sources")
        atomic_json(folder / ".complete.json", {"repo": repo, "revision": revision, "archive_sha256": digest(archive)})
    return folder


def download_models(root, components, whisper_name="small", vocoder_variant="regular"):
    root = Path(root)
    result = {}
    if "feature" in components or "sources" in components:
        result["knn_source"] = str(download_repository("bshall/knn-vc", KNN_REVISION, root).resolve())
    if "feature" in components:
        release = "https://github.com/bshall/knn-vc/releases/download/v0.1"
        for filename in ["WavLM-Large.pt", "g_02500000.pt" if vocoder_variant == "regular" else "prematch_g_02500000.pt"]:
            url = f"{release}/{filename}?download=1"
            path = root / "checkpoints" / filename
            result[filename] = str((download_ranged(url, path) if filename == "WavLM-Large.pt" else download(url, path)).resolve())
    if "whisper" in components:
        import whisper
        if whisper_name not in whisper.available_models():
            raise ValueError(f"Unknown Whisper model: {whisper_name}")
        url = whisper._MODELS[whisper_name]
        result["whisper"] = str(download(url, root / "whisper" / f"{whisper_name}.pt", expected=url.split("/")[-2]).resolve())
    if "utmos" in components:
        source = download_repository("tarepan/SpeechMOS", UTMOS_REVISION, root)
        import torch
        torch.hub.set_dir(str((root / "torchhub").resolve()))
        torch.hub.load(str(source), "utmos22_strong", source="local", pretrained=True)
        result["utmos_source"] = str(source.resolve())
    return result


def download_corpus(corpus, root, splits=None, base_url="https://www.openslr.org/resources"):
    root = Path(root)
    if corpus == "librispeech":
        splits = splits or ["dev-clean", "test-clean"]
        allowed = {"train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"}
        if not set(splits) <= allowed:
            raise ValueError("Unknown LibriSpeech split")
        checksums_file = download(f"{base_url}/12/md5sum.txt", root / "downloads" / "librispeech-md5sum.txt")
        checksums = {line.split()[1].lstrip("*"): line.split()[0] for line in checksums_file.read_text().splitlines() if line.strip()}
        files = [(f"{split}.tar.gz", checksums[f"{split}.tar.gz"], 12) for split in splits]
    elif corpus == "aishell":
        files = [("data_aishell.tgz", None, 33)]
    else:
        raise ValueError(corpus)
    for name, checksum, resource in files:
        archive = download(f"{base_url}/{resource}/{name}", root / "downloads" / name, checksum, "md5")
        marker = root / "downloads" / f"{name}.extracted.json"
        if not marker.exists():
            safe_extract(archive, root)
            atomic_json(marker, {"sha256": digest(archive)})
    if corpus == "aishell":
        nested_archives = sorted((root / "data_aishell" / "wav").glob("*.tar.gz"))
        for archive in tqdm(nested_archives, desc="AISHELL speaker archives", unit="archives"):
            marker = archive.with_suffix(".extracted.json")
            if not marker.exists():
                safe_extract(archive, archive.parent)
                atomic_json(marker, {"sha256": digest(archive)})
