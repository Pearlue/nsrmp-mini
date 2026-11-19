import os
import time
from functools import lru_cache
from pathlib import Path

import torch

from nsrmp.baseline import ParsedNeuralBaseline
from nsrmp.batch import collate
from nsrmp.data import keyword_kind, make_dataset
from nsrmp.learned_parser import Grammar, ROLES, role_to_program
from nsrmp.learned_lr import RolePolicy
from nsrmp.model import ActionSimulator, NSModel
from nsrmp.parser import parse
from nsrmp.train import policy_objective, scene_loss


def build_model(*, parser_source, backend):
    simulator = ActionSimulator()
    with torch.random.fork_rng(devices=[]):
        parser = {"learned": RolePolicy, "rule": lambda: None}[parser_source]()
    model = {"symbolic": NSModel, "neural": ParsedNeuralBaseline}[backend](sim=simulator)
    model.parser = parser
    return model


@lru_cache(maxsize=None)
def _role_mask(pattern, prefix):
    prototypes = {"colour": "red", "noun": "cube", "relation": "left"}
    grammar = Grammar(tuple(prototypes[kind] for kind in pattern))
    for i in range(len(prefix)):
        grammar.valid_roles(prefix[:i])
    return grammar.valid_roles(prefix)


def _draw(parser, scene, keywords, patterns, count, greedy):
    logits = parser(scene)[:, 0].repeat(count, 1, 1)
    words = keywords * count
    kinds = patterns * count
    traces = [() for _ in words]
    log_prob = logits.new_zeros(len(words))
    entropy = torch.zeros_like(log_prob)
    for pos in range(max(map(len, words))):
        active = [i for i, row in enumerate(words) if pos < len(row)]
        available = [_role_mask(kinds[i], traces[i]) for i in active]
        allowed = torch.tensor(available)
        scores = logits[active, pos].masked_fill(~allowed, -torch.inf)
        logs = scores.log_softmax(-1)
        probs = logs.exp()
        role = scores.argmax(-1, keepdim=True) if greedy else torch.multinomial(probs, 1)
        log_prob = log_prob.index_add(0, torch.tensor(active), logs.gather(1, role).flatten())
        local_entropy = -(probs * logs.masked_fill(~allowed, 0)).sum(-1)
        entropy = entropy.index_add(0, torch.tensor(active), local_entropy)
        for i, chosen in zip(active, role.flatten().tolist()):
            traces[i] += (ROLES[chosen],)
    programs = [(role_to_program(row, trace),) for row, trace in zip(words, traces)]
    lengths = scene.keyword_mask.sum((1, 2))
    return programs, log_prob.reshape(count, -1), entropy.reshape(count, -1) / lengths


def _set_phase(model, phase):
    model.requires_grad_(phase in ("warm", "mixed"))
    if phase == "sim":
        model.sim.requires_grad_(True)
    if phase == "relation":
        relation = model.concepts.rel_mlp if isinstance(model, NSModel) else model.find.relation
        relation.requires_grad_(True)
        if model.parser is not None:
            model.parser.requires_grad_(True)


def _save(path, payload):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_examples(config, seed):
    examples = make_dataset(config["train_examples"], seed)
    extra = config.get("extra_complex_examples", 0)
    if extra:
        examples += make_dataset(extra, config["extra_train_seed"] + seed, p_complex=1.0)
    return examples


def _epoch_indices(batch, config, phase):
    simple = (~batch.is_complex).nonzero().flatten()
    mixed = phase in ("mixed", "relation")
    if not mixed:
        return ((simple, len(simple)),)
    if not config.get("extra_complex_examples", 0):
        return ((torch.arange(len(batch)), len(batch)),)
    complex_count = config["train_examples"] - len(simple)
    return ((simple, len(simple)), (batch.is_complex.nonzero().flatten(), complex_count))


def _shuffle_epoch(blocks, generator):
    picks = [pool[torch.randperm(len(pool), generator=generator)[:count]] for pool, count in blocks]
    if len(picks) == 1:
        return picks[0]
    merged = torch.cat(picks)
    return merged[torch.randperm(len(merged), generator=generator)]


def _fit(model, optimizer, scheduler, shuffle, batch, rules, keywords, patterns, config, phase, mode):
    _set_phase(model, phase)
    blocks = _epoch_indices(batch, config, phase)
    examples_per_epoch = sum(count for _, count in blocks)
    batch_size = config["batch_size"]
    frozen_programs = {}
    learned = model.parser is not None
    if learned and phase == "sim":
        with torch.no_grad():
            for ids in blocks[0][0].split(batch_size):
                rows = ids.tolist()
                programs, _, _ = _draw(model.parser, batch.inputs.select(ids),
                                       [keywords[i] for i in rows], [patterns[i] for i in rows], 1, True)
                frozen_programs.update(zip(rows, programs))
    started = time.monotonic()
    history = []
    for _ in range(config[f"{phase}_epochs"]):
        total = 0.0
        for ids in _shuffle_epoch(blocks, shuffle).split(batch_size):
            rows = ids.tolist()
            current = batch.select(ids)
            scene = current.inputs
            if learned and phase != "sim":
                k = config["num_samples"]
                programs, log_prob, entropy = _draw(model.parser, scene, [keywords[i] for i in rows],
                                                    [patterns[i] for i in rows], k, False)
                scene = scene.select(torch.arange(len(ids)).repeat(k))
            else:
                k = 1
                source = frozen_programs if learned else rules
                programs = [source[i] for i in rows]
            pred = model.execute(scene, programs, move_mode=mode).final_boxes.reshape(k, len(ids), -1, 4)
            losses = scene_loss(pred, current.final_boxes, current.inputs.object_mask,
                                beta=config["beta"], objective=config["loss"])
            loss = losses.mean()
            if learned and phase != "sim":
                policy, _ = policy_objective(log_prob, -losses)
                loss = loss + policy - config["entropy_coef"] * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += losses.detach().mean().item() * len(ids)
        scheduler.step()
        history.append(total / examples_per_epoch)
    return {"seconds": time.monotonic() - started, "scene_loss": history,
            "examples_per_epoch": examples_per_epoch,
            "updates": len(history) * ((examples_per_epoch + batch_size - 1) // batch_size)}


def train_group(group, seed, config):
    spec = config["groups"][str(group)]
    mode = spec["move_mode"]
    directory = Path(config["output"]) / f"group-{group}" / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    examples = training_examples(config, seed)
    batch = collate(examples, surface_context=True, mode="learned")
    model = build_model(parser_source=spec["parser_source"], backend=spec["backend"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    shuffle = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    learned = model.parser is not None
    rules = [] if learned else [parse(e.text) for e in examples]
    keywords = [batch.inputs.step_keywords(i, 0) for i in range(len(batch))] if learned else []
    patterns = [tuple(map(keyword_kind, words)) for words in keywords]
    phases = {"A": ("warm", "mixed"), "B": ("warm", "sim", "relation")}[spec["schedule"]]
    epochs = sum(config[f"{phase}_epochs"] for phase in phases)
    minimum_lr = config["learning_rate"] * config.get("min_lr_ratio", 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=minimum_lr)
    settings = {key: value for key, value in config.items() if key != "groups"}
    settings.update(spec)
    history = {}
    for phase in phases:
        current_mode = "soft" if phase == "warm" else mode
        history[phase] = _fit(model, optimizer, scheduler, shuffle, batch, rules, keywords,
                              patterns, settings, phase, current_mode)
        path = directory / (phase + ".pt")
        _save(path, dict(group=group, seed=seed, phase=phase, spec=spec, move_mode=current_mode,
                         model=model.state_dict(), config=settings, history=history))
        print(f"group {group} seed {seed} {phase} loss {history[phase]['scene_loss'][-1]:.5f}", flush=True)
    return path
