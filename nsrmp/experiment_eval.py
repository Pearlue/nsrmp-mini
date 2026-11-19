import torch

from nsrmp.batch import collate
from nsrmp.curriculum import build_model
from nsrmp.metrics import iou
from nsrmp.parser import parse


@torch.inference_mode()
def evaluate(checkpoint, examples, *, parser_source=None, move_mode=None):
    state = torch.load(checkpoint, map_location="cpu")
    spec = state["spec"]
    model = build_model(parser_source=spec["parser_source"], backend=spec["backend"]).eval()
    model.load_state_dict(state["model"])
    source = spec["parser_source"] if parser_source is None else parser_source
    mode = state["move_mode"] if move_mode is None else move_mode
    if source not in ("learned", "rule") or (source == "learned" and model.parser is None):
        raise ValueError(f"checkpoint cannot evaluate parser {source!r}")
    moved_count = 0
    overlap_sum = 0.0
    size = state["config"]["batch_size"]
    for start in range(0, len(examples), size):
        items = examples[start:start + size]
        batch = collate(items, surface_context=True, mode="learned")
        scene = batch.inputs
        if source == "rule":
            programs = [parse(e.text) for e in items]
        else:
            programs = model.parser.sample(scene, greedy=True).programs
        result = model.execute(scene, programs, move_mode=mode)
        moved = (scene.boxes - batch.final_boxes).abs().sum(-1).gt(1e-6) & scene.object_mask
        overlap_sum += iou(result.final_boxes, batch.final_boxes)[moved].double().sum().item()
        moved_count += int(moved.sum())
    return {"iou_m": overlap_sum / moved_count if moved_count else None,
            "moved_objects": moved_count, "examples": len(examples)}
