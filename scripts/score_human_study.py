"""Summarize independently collected human-study responses.

JSONL rows include volunteer_id, sample_id, task. For task=paired give
indistinguishable:bool; task=artifact give suspicious:bool; task=transcription
give reference and hypothesis. These tasks cannot recreate absent volunteers.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--responses", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rows = [json.loads(line) for line in Path(args.responses).read_text().splitlines() if line.strip()]
if not rows or len({(r["volunteer_id"], r["sample_id"], r["task"]) for r in rows}) != len(rows):
    raise ValueError("Responses must be unique per volunteer/sample/task")
volunteers = {r["volunteer_id"] for r in rows}
paired = [r for r in rows if r["task"] == "paired"]
artifact = [r for r in rows if r["task"] == "artifact"]
transcription = [r for r in rows if r["task"] == "transcription"]
if not paired or not artifact or not transcription:
    raise ValueError("All three paper assessments require recorded responses")
errors = [error_counts(r["reference"], r["hypothesis"], r.get("language", "en"))
          for r in tqdm(transcription, desc="Scoring human transcripts", unit="responses")]
summary = {"volunteers": len(volunteers), "paired_responses": len(paired),
           "paired_indistinguishable_percent": 100*sum(r["indistinguishable"] for r in paired)/len(paired),
           "artifact_responses": len(artifact),
           "no_noticeable_artifact_percent": 100*sum(not r["suspicious"] for r in artifact)/len(artifact),
           "transcriptions": len(transcription), "human_corpus_error": aggregate_errors(errors),
           "status": "new human study; original participant audio/responses unavailable"}
atomic_json(args.output, summary)
print(json.dumps(summary, indent=2))
