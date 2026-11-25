"""Background evaluations must export transcripts even under an ASCII locale."""
import os
from pathlib import Path
import subprocess
import sys


def test_export_evaluation_and_report_under_ascii_locale(tmp_path):
    source = r'''
import codecs
import json
import locale
from pathlib import Path
import sys
import torch
from beyondwaveform.io import atomic_json, digest, read_jsonl, save_audio, write_jsonl
import beyondwaveform.pipeline as pipeline

assert codecs.lookup(locale.getpreferredencoding(False)).name == "ascii"
root = Path(sys.argv[1])
text = "Hello \u2014 caf\u00e9 \u4f60\u597d"
write_jsonl(root / "unicode.jsonl", [{"id": "one", "text": text}])
assert read_jsonl(root / "unicode.jsonl")[0]["text"] == text
atomic_json(root / "unicode.json", {"text": text})
assert json.loads((root / "unicode.json").read_bytes().decode("utf-8"))["text"] == text

asset = save_audio(root / "clip.wav", torch.linspace(-0.1, 0.1, 1600))
run = root / "run"
write_jsonl(run / "generated.jsonl", [{"id": "one", "text": "hello world", "language": "en",
                                       "audio": {"adversarial": asset}}])
atomic_json(run / "run.json", {"status": "complete", "generated_sha256": digest(run / "generated.jsonl"),
    "fingerprint": "test", "manifest_sha256": "test", "roles": ["test"], "surrogate": {}})
class Target:
    def transcribe(self, audio):
        return text
pipeline.load_target = lambda *args: (Target(), {})
pipeline.evaluate(run, {"kind": "extension"}, root / "eval", device="cpu")
rows = [json.loads(line) for line in (root / "eval/utterances.jsonl").read_bytes().decode("utf-8").splitlines()]
assert rows[0]["hypothesis"] == text

sys.path.insert(0, str(Path.cwd() / "scripts"))
from run_defense_suite import read_json, report
assert read_json(root / "unicode.json")["text"] == text
atomic_json(root / "plan.json", {})
report(root, {"settings": [], "unavailable": {}}, {}, "test")
assert "Attack \u2212 clean" in (root / "report.md").read_bytes().decode("utf-8")
'''
    env = {**os.environ, "LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C",
           "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "PYTHONIOENCODING": "ascii"}
    result = subprocess.run([sys.executable, "-c", source, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
