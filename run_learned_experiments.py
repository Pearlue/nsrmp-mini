"""Run the opt-in learned-parser experiment for the miniature 2D task."""
import json
import math
import os
import platform
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nsrmp.batch import collate
from nsrmp.data import FEATURE_NOISE, make_dataset
from nsrmp.learned_metrics import build_annotations, evaluate_learned
from nsrmp.model import ParseNSModel
from nsrmp.train import train_parser


_MODES = ("surface_context", "keywords_only")
_TRAIN_OPTIONS = {"curriculum": True, "lr": 3e-3, "batch_size": 64, "beta": 1.0,
                  "entropy_coef": 0.01, "ema_decay": 0.9}
_METRICS = ("iou", "iou_m", "subject_acc", "placement_acc", "program_acc", "step_program_acc")


def _environment():
    root = Path(__file__).resolve().parent

    def read(*command):
        try:
            return subprocess.check_output(command, cwd=root, stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    if platform.system() == "Darwin":
        memory = read("sysctl", "-n", "hw.memsize")
        memory = int(memory) if memory else None
        cpu = read("sysctl", "-n", "machdep.cpu.brand_string")
    else:
        try:
            memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            memory = None
        cpu = platform.processor()
    status = read("git", "status", "--porcelain", "--untracked-files=all")
    return {"git_commit": read("git", "rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "platform": platform.platform(), "cpu": cpu or platform.machine(),
            "memory_bytes": memory, "logical_cpus": os.cpu_count(), "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def _splits(config, phase, seed):
    normal = {"n_objects": (3, 5)}
    if phase == "2":
        normal["n_steps"] = (1, 2)
    splits = {
        "train": (config["n_train"], seed, normal),
        "validation": (config["n_test"], 5000 + seed, normal),
        "test": (config["n_test"], 1000 + seed, normal),
    }
    for value in (range(3, 11) if phase == "1" else range(1, 6)):
        params = ({"n_objects": (value, value)} if phase == "1"
                  else {"n_objects": (3, 5), "n_steps": (value, value)})
        splits[str(value)] = (config["n_gen"], 2000 + 100 * seed + value, params)
    return splits


def _describe_splits(splits):
    return {name: {"count": size, "seed": seed, **{"n_steps": (1, 1), **settings},
                   "p_complex": 0.5, "feature_noise": FEATURE_NOISE}
            for name, (size, seed, settings) in splits.items()}


def _evaluate(model, examples, config, context, seed):
    data = collate(examples, mode="learned", surface_context=context)
    result = evaluate_learned(model, data, build_annotations(examples, data),
                              sampled_count=config["sampled_count"],
                              uniform_count=config["uniform_count"], eval_seed=seed)
    json.dumps(result, allow_nan=False)
    return result


def _failure(record, error):
    record.update(status="failed", error=f"{type(error).__name__}: {error}",
                  traceback=traceback.format_exc())
    print(f"[learned phase {record['phase']}] {record['mode']} seed {record['seed']}: "
          f"{record['error']}", flush=True)


def _run_seed(config, phase, seed, runs):
    splits = _splits(config, phase, seed)
    records = []
    for mode, rows in runs.items():
        record = {"phase": phase, "mode": mode, "seed": seed, "status": "pending",
                  "model_seed": seed, "training_seed": seed, "splits": _describe_splits(splits),
                  "seconds": {"data_generation": None, "train": None, "eval": None,
                              "end_to_end": None}}
        rows.append(record)
        records.append(record)

    start = time.perf_counter()
    try:
        examples = {name: make_dataset(count, seed=split_seed, **params)
                    for name, (count, split_seed, params) in splits.items()}
    except Exception as error:
        elapsed = time.perf_counter() - start
        for record in records:
            record["seconds"].update(data_generation=elapsed, end_to_end=elapsed)
            _failure(record, error)
        return

    data_seconds = time.perf_counter() - start
    for record in records:
        seconds = record["seconds"]
        seconds["data_generation"] = data_seconds
        started = time.perf_counter()
        print(f"[learned phase {phase}] {record['mode']} seed {seed}: training", flush=True)
        try:
            context = record["mode"] == "surface_context"
            training = collate(examples["train"], mode="learned", surface_context=context)
            torch.default_generator.manual_seed(seed)
            model = ParseNSModel()
            tick = time.perf_counter()
            try:
                history = train_parser(model, training, config["epochs"], seed=seed,
                                       num_samples=config["num_samples"], **config["training"])
            finally:
                seconds["train"] = time.perf_counter() - tick
            record["history"] = [{k: v if math.isfinite(v) else None for k, v in row.items()}
                                 for row in history]
            record["train_program_executions"] = sum(row["n_samples"] for row in history)
            if record["history"] != history:
                raise ValueError("non-finite training history (stored as null)")

            tick = time.perf_counter()
            try:
                record["generalisation"] = {}
                for name, (_, split_seed, _) in splits.items():
                    if name == "train":
                        continue
                    eval_config = config if name == "test" else {**config, "uniform_count": 0}
                    summary = _evaluate(model, examples[name], eval_config, context, split_seed)
                    if name in ("validation", "test"):
                        record[name] = summary
                    else:
                        record["generalisation"][name] = summary
            finally:
                seconds["eval"] = time.perf_counter() - tick
            record["status"] = "ok"
            print(f"[learned phase {phase}] {record['mode']} seed {seed}: "
                  f"IoU-M {record['test']['overall']['iou_m']:.3f}, "
                  f"program accuracy {record['test']['overall']['program_acc']:.3f} "
                  f"({seconds['train']:.1f}s training)", flush=True)
        except Exception as error:
            _failure(record, error)
        finally:
            seconds["end_to_end"] = data_seconds + time.perf_counter() - started


def _stat(values):
    total = len(values)
    used = [value for value in values if value is not None]
    if not used:
        return f"N/A (0/{total})"
    std = f"{statistics.stdev(used):.3f}" if len(used) > 1 else "N/A"
    return f"{statistics.mean(used):.3f} \u00b1 {std} ({len(used)}/{total})"


def _markdown(result):
    lines = ["# Our results - mini 2D learned parser", "",
             "Greedy test metrics from the last epoch. Cells show mean ± sample std "
             "and contributing / configured seeds.", "",
             "| Mode | Split | " + " | ".join(_METRICS) + " |",
             "| --- | --- |" + " --- |" * len(_METRICS)]
    for mode, records in result["runs"].items():
        for group in ("overall", "simple", "complex"):
            cells = [_stat([r["test"][group][metric] if r["status"] == "ok" else None
                            for r in records]) for metric in _METRICS]
            lines.append(f"| {mode} | {group} | " + " | ".join(cells) + " |")
    lines += ["", "| Mode | Seed | Status | Train seconds | Eval seconds | Program executions |",
              "| --- | --- | --- | --- | --- | --- |"]
    for mode, records in result["runs"].items():
        for r in records:
            seconds = r["seconds"]
            times = [f"{seconds[key]:.3f}" if seconds[key] is not None else "N/A"
                     for key in ("train", "eval")]
            lines.append(f"| {mode} | {r['seed']} | {r['status']} | "
                         f"{' | '.join(times)} | {r.get('train_program_executions', 'N/A')} |")
    lines += ["", "Training execution counts include all K draws per visited instruction. "
              "See JSON for runtime boundaries, validation, generalisation, diagnostics and failures."]
    return "\n".join(lines) + "\n"


_LEGACY_LABELS = {
    "ours": "Rule-parser NS",
    "parsed_baseline": "Parse-matched neural",
    "baseline": "Raw-sentence GRU",
    "ours_no_curriculum": "Rule-parser NS (no curriculum)",
}


def _load_legacy(path):
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or not isinstance(result.get("config"), dict)
                or not isinstance(result.get("runs"), dict)):
            raise ValueError("expected config and runs objects")
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid legacy comparison JSON at {path}: {error}") from error
    return result


def _comparison_markdown(result, legacy, legacy_path):
    def scores(records, seeds, group, metric):
        indexed = {}
        for record in records:
            seed = record["seed"]
            if seed in indexed:
                raise ValueError(f"duplicate comparison seed {seed}")
            if seed not in seeds:
                raise ValueError(f"unexpected comparison seed {seed}")
            indexed[seed] = record

        values = []
        for seed in seeds:
            record = indexed.get(seed)
            if record is None or record.get("status", "ok") != "ok":
                values.append(None)
                continue
            try:
                value = record["test"][group][metric]
            except (KeyError, TypeError) as error:
                raise ValueError(f"missing comparison metric {group}.{metric}") from error
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"invalid comparison metric {group}.{metric}: {value!r}")
            values.append(value)
        return _stat(values)

    config = result["config"]
    legacy_seeds = legacy["config"].get("seeds") if legacy is not None else config["seeds"]
    if not isinstance(legacy_seeds, list):
        raise ValueError("invalid legacy comparison config.seeds")
    missing = f"N/A (0/{len(legacy_seeds)})"
    lines = ["## Comparison with existing models", "",
             "Mini 2D task, final-epoch greedy test results: mean ± sample std "
             "(contributing/configured seeds). Program accuracy is whole-instruction "
             "AST exact-match for learned parsers.", "",
             "| Model | IoU-M overall | IoU-M simple | IoU-M complex | Program accuracy | Source |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, label in _LEGACY_LABELS.items():
        if name == "ours_no_curriculum" and (legacy is None or name not in legacy["runs"]):
            continue
        records = legacy["runs"].get(name) if legacy is not None else None
        if records is not None and not isinstance(records, list):
            raise ValueError(f"invalid legacy comparison records for {name}")
        values = ([scores(records, legacy_seeds, group, "iou_m")
                   for group in ("overall", "simple", "complex")]
                  if records is not None else [missing] * 3)
        if records is None:
            program = "N/A (no source)"
        else:
            program = ("By construction" if name in ("ours", "ours_no_curriculum")
                       else "N/A (not measured)")
        source = f"Legacy {legacy_path.name}" if legacy is not None else "Legacy missing"
        lines.append(f"| {label} | {' | '.join(values)} | {program} | {source} |")

    for mode, records in result["runs"].items():
        values = [scores(records, config["seeds"], group, "iou_m")
                  for group in ("overall", "simple", "complex")]
        program = scores(records, config["seeds"], "overall", "program_acc")
        lines.append(f"| Learned Parser ({mode}) | {' | '.join(values)} | {program} "
                     f"| Learned learned_phase{result['phase']}.json |")

    keys = ("n_train", "n_test", "n_gen", "epochs", "seeds")

    def describe(source):
        return ", ".join(f"{key}={source.get(key)}" for key in keys)

    lines += ["", f"Learned source: `learned_phase{result['phase']}.json`; "
              f"config: {describe(config)}, num_samples={config['num_samples']}."]
    if legacy is None:
        lines.append(f"Legacy source missing: `{legacy_path}`; legacy rows are unavailable.")
    else:
        old_config = legacy["config"]
        lines.append(f"Legacy source (read-only): `{legacy_path}`; config: {describe(old_config)}.")
        mismatches = [f"{key}: legacy={old_config.get(key)}, learned={config.get(key)}"
                      for key in keys if old_config.get(key) != config.get(key)]
        if mismatches:
            lines.append("**NOT DIRECTLY COMPARABLE:** " + "; ".join(mismatches) + ".")
        else:
            lines.append("Recorded dataset sizes, epochs and seeds match; this does not prove "
                         "identical data-generator revisions or compute-matched training.")
        lines.append("Legacy JSON has no recorded Git commit; values are existing results, not a rerun.")

    lines.append("Learned training samples K programs per instruction; legacy models do not "
                 "use the same sampling protocol or compute budget.")
    return "\n".join(lines) + "\n"


def _plot(result, path):
    fig = Figure(figsize=(10, 3.5), layout="constrained")
    FigureCanvasAgg(fig)
    loss, reward, generalisation = fig.subplots(1, 3)
    for mode, records in result["runs"].items():
        for r in records:
            history = r.get("history", [])
            label = f"{mode} seed {r['seed']}"
            for ax, metric in ((loss, "task_loss"), (reward, "reward")):
                if history:
                    ax.plot([e["epoch"] + 1 for e in history], [e[metric] for e in history],
                            marker=".", label=label)
            if r["status"] == "ok":
                by_axis = r["generalisation"]
                generalisation.plot([int(k) for k in by_axis],
                                    [v["overall"]["iou_m"] for v in by_axis.values()],
                                    marker=".", label=label)
    loss.set(xlabel="Epoch", ylabel="Scene loss")
    reward.set(xlabel="Epoch", ylabel="Reward")
    axis = "Objects" if result["phase"] == "1" else "Steps"
    generalisation.set(xlabel=axis, ylabel="IoU-M")
    if loss.lines:
        loss.legend(fontsize="xx-small")
    fig.savefig(path)


def run(cfg, outdir, *, phase="1", modes=("surface_context",), num_samples=4,
        sampled_count=0, uniform_count=None):
    """Run one phase (or both), writing a JSON record and reports for each phase."""
    if phase not in ("1", "2", "all"):
        raise ValueError(f"unsupported learned phase: {phase!r}")
    uniform_count = num_samples if uniform_count is None else uniform_count
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes).difference(_MODES):
        raise ValueError(f"expected distinct learned modes from {_MODES}: {modes!r}")
    config = {key: getattr(cfg, key) for key in ("n_train", "n_test", "n_gen", "epochs")}
    seeds = list(cfg.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("expected non-empty distinct non-negative seeds")
    if min(*config.values(), num_samples) < 1 or min(sampled_count, uniform_count) < 0:
        raise ValueError("expected positive experiment sizes and non-negative diagnostic counts")
    config.update(seeds=seeds, modes=list(modes), num_samples=num_samples,
                  sampled_count=sampled_count, uniform_count=uniform_count,
                  training=_TRAIN_OPTIONS, checkpoint="last_epoch",
                  validation_count=config["n_test"], eval_seed="split_seed")
    outdir = Path(outdir)
    if outdir.resolve() == Path(__file__).resolve().parent / "results":
        raise ValueError("use a learned output directory separate from legacy results")
    outdir.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    phases = ("1", "2") if phase == "all" else (phase,)
    results = {}
    failures = []
    for current in phases:
        start = time.perf_counter()
        runs = {mode: [] for mode in modes}
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                _run_seed(config, current, seed, runs)
        result = {"config": config, "phase": current, "runs": runs, "environment": environment,
                  "total_seconds": time.perf_counter() - start,
                  "timing_scope": "Includes data generation, training and evaluation; excludes "
                                  "environment setup and report rendering. Paired modes share generation."}
        stem = outdir / f"learned_phase{current}"
        raw = json.dumps(result, indent=2, allow_nan=False)
        stem.with_suffix(".json").write_text(raw + "\n", encoding="utf-8")
        result = json.loads(raw)
        legacy_path = outdir.parent / f"phase{current}.json"
        report = _markdown(result) + "\n" + _comparison_markdown(
            result, _load_legacy(legacy_path), legacy_path)
        stem.with_suffix(".md").write_text(report, encoding="utf-8")
        _plot(result, stem.with_suffix(".png"))
        results[current] = result
        failures += [f"phase {current}, {mode}, seed {r['seed']}" for mode, rows in runs.items()
                     for r in rows if r["status"] != "ok"]
    if failures:
        raise RuntimeError(f"learned runs failed ({'; '.join(failures)}); see records in {outdir}")
    return results if phase == "all" else results[phase]
