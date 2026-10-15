"""Paper grids, term-removal ablations, and results without invented plot values."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import yaml
from .io import atomic_json


def experiment_configs(base, suite):
    variants = []
    if suite == "feature-grid":
        for rho, weight in itertools.product([0.05, 0.1, 0.5, 1.0], [0.1, 0.5, 1.0, 2.0]):
            config = deepcopy(base)
            config["method"] = "feature"
            config.setdefault("attack", {}).update(rho=rho, lambda_perc=weight)
            variants.append((f"rho-{rho}_lambda-{weight}", config))
    elif suite == "ablations":
        for name, settings in [("full", {}), ("without-attack", {"attack": False}),
                               ("without-perceptual", {"lambda_perc": 0.0}),
                               ("without-tv", {"temporal": False}),
                               ("without-hf", {"high_frequency": False})]:
            config = deepcopy(base)
            config["method"] = "feature"
            config.setdefault("attack", {}).update(settings)
            variants.append((name, config))
    elif suite == "waveform-grid":
        for method in ["pgd", "mi-fgsm", "vmi-fgsm"]:
            for epsilon, scale in itertools.product([0.001, 0.002, 0.005, 0.01, 0.02], [1/50, 2/50, 1/10]):
                momentum_grid = [0.5, 1.0] if method == "mi-fgsm" else [1.0]
                neighbors = itertools.product([3, 5], [1.0, 1.5, 2.0]) if method == "vmi-fgsm" else [(5, 1.5)]
                for momentum, (samples, beta) in itertools.product(momentum_grid, list(neighbors)):
                    config = deepcopy(base)
                    config["method"] = method
                    config["waveform"] = {"steps": 50, "epsilon": epsilon, "step_size": epsilon*scale,
                                           "momentum": momentum, "variance_samples": samples, "beta": beta}
                    name = f"{method}_eps-{epsilon}_alpha-{epsilon*scale:g}_mu-{momentum}_N-{samples}_beta-{beta}"
                    variants.append((name, config))
    else:
        raise ValueError(suite)
    return variants


def plan_experiments(base, suite, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, config in experiment_configs(base, suite):
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        records.append({"name": name, "config": str(path.resolve())})
    atomic_json(output / "plan.json", {"suite": suite, "experiments": records})
    return records


def report(summaries, output):
    lines = ["# BeyondWaveform experiment report", "", "Computed values below are observations, not paper results.", "",
             "| Run | Target | Variant | Count | Error (%) | DNSMOS | NISQA | UTMOS |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for path in summaries:
        result = json.loads(Path(path).read_text())
        for variant, values in result["summary"].items():
            q = values.get("quality", {})
            cells = [Path(result["generation_run"]).name, result["target"].get("name", result["target"]["kind"]),
                     variant, str(values["count"]), f"{values['error_percent']:.3f}",
                     *[f"{q[m]:.3f}" if m in q else "not measured" for m in ["dnsmos", "nisqa", "utmos"]]]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Paper anchors: English raw clean 4.75% WER, feature 75.43%; Mandarin raw clean 6.29% CER, feature 72.25%.",
              "Table 4 feature DNSMOS/NISQA/UTMOS: 2.954 / 3.526 / 3.590.",
              "Comparisons require matched corpus, manifest, model, decoder, scorer and configuration; see run provenance."]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n")

