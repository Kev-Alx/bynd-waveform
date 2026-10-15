"""Each subcommand also has a small standalone script in scripts/."""
import argparse
import json
from pathlib import Path
from .config import load_config
from .io import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="BeyondWaveform independent reproduction pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download-data")
    p.add_argument("--corpus", choices=["librispeech", "aishell"], required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--splits", nargs="+")
    p.add_argument("--base-url", default="https://www.openslr.org/resources")
    p = commands.add_parser("prepare-data")
    p.add_argument("--corpus", choices=["librispeech", "aishell"], required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--split", required=True)
    p.add_argument("--role", choices=["train", "finetune", "validation", "test", "pilot"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-seconds", type=float, default=30)
    p.add_argument("--ids-file")
    p = commands.add_parser("audit-splits")
    p.add_argument("manifests", nargs="+")
    p = commands.add_parser("download-models")
    p.add_argument("--root", default="artifacts")
    p.add_argument("--components", nargs="+", choices=["sources", "feature", "whisper", "utmos"], default=["feature", "whisper"])
    p.add_argument("--whisper", default="small")
    p.add_argument("--vocoder-variant", choices=["regular", "prematched"], default="regular")
    p = commands.add_parser("generate")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    p = commands.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--target", required=True, help="YAML target, or 'surrogate' to use run's exact surrogate")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--variants", nargs="+", choices=["clean", "reconstruction", "adversarial"])
    p.add_argument("--quality", action="store_true")
    p.add_argument("--artifact-root", default="artifacts")
    p = commands.add_parser("select")
    p.add_argument("summaries", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--inclusive", action="store_true", help="Use >=2.5 instead of Table 2's strict >2.5")
    p = commands.add_parser("finetune")
    p.add_argument("--config", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--validation-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p = commands.add_parser("experiments")
    p.add_argument("--config", required=True)
    p.add_argument("--suite", choices=["feature-grid", "ablations", "waveform-grid"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--manifest")
    p.add_argument("--quality", action="store_true")
    p.add_argument("--device", default="cuda")
    p = commands.add_parser("report")
    p.add_argument("summaries", nargs="+")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "download-data":
        from .assets import download_corpus
        download_corpus(args.corpus, args.root, args.splits, args.base_url)
    elif args.command == "prepare-data":
        from .data import prepare_manifest
        rows = prepare_manifest(args.corpus, args.root, args.split, args.role, args.output,
                                args.count, args.seed, args.max_seconds, args.ids_file)
        print(f"Wrote {len(rows)} utterances to {args.output}")
    elif args.command == "audit-splits":
        from .data import audit_disjoint
        print(json.dumps(audit_disjoint(args.manifests), indent=2))
    elif args.command == "download-models":
        from .assets import download_models
        print(json.dumps(download_models(args.root, args.components, args.whisper, args.vocoder_variant), indent=2))
    elif args.command == "generate":
        from .pipeline import generate
        result = generate(load_config(args.config), args.manifest, args.output, args.device, args.limit)
        print(json.dumps({k: result[k] for k in ["status", "count", "generation_seconds"]}, indent=2))
    elif args.command == "evaluate":
        from .pipeline import evaluate
        if args.target == "surrogate":
            config = json.loads((Path(args.run) / "run.json").read_text())["config"]["surrogate"]
            config = {"kind": "whisper", **config}
        else:
            config = load_config(args.target)
        result = evaluate(args.run, config, args.output, args.device, args.variants, args.quality, args.artifact_root)
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "select":
        from .selection import select_configuration
        print(json.dumps(select_configuration(args.summaries, args.output, strict=not args.inclusive)["selected"], indent=2))
    elif args.command == "finetune":
        from .training import finetune
        finetune(load_config(args.config), args.train_manifest, args.validation_manifest, args.output, args.device)
    elif args.command == "experiments":
        from .experiments import plan_experiments
        plan = plan_experiments(load_config(args.config), args.suite, args.output)
        if args.execute:
            if not args.manifest:
                parser.error("--execute requires --manifest")
            from .pipeline import generate, evaluate
            from tqdm import tqdm
            for entry in tqdm(plan, desc="Experiment grid", unit="runs"):
                config = load_config(entry["config"])
                run = Path(args.output) / entry["name"]
                generate(config, args.manifest, run, args.device)
                evaluate(run, {"kind": "whisper", **config["surrogate"]}, run / "evaluation", args.device,
                         variants=["adversarial"], quality=args.quality,
                         artifact_root=config["features"]["artifact_root"])
        print(f"{'Completed' if args.execute else 'Prepared'} {len(plan)} experiments")
    elif args.command == "report":
        from .experiments import report
        report(args.summaries, args.output)


if __name__ == "__main__":
    main()
