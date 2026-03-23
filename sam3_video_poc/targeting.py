from __future__ import annotations

def normalize_click_point(
    click_x: int,
    click_y: int,
    width: int,
    height: int,
) -> list[list[float]]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return [[float(click_x) / float(width), float(click_y) / float(height)]]


def pick_target_obj_id(outputs: dict[str, object]) -> tuple[int | None, float | None]:
    raw_obj_ids = outputs.get("out_obj_ids", [])
    obj_ids = [int(obj_id) for obj_id in raw_obj_ids]
    if not obj_ids:
        return None, None
    raw_scores = outputs.get("out_probs", [])
    scores = [float(score) for score in raw_scores]
    best_idx = max(range(len(obj_ids)), key=scores.__getitem__) if len(scores) == len(obj_ids) and scores else 0
    return obj_ids[best_idx], scores[best_idx] if scores else None


def resolve_text_target_obj_id(
    prompt_outputs: dict[str, object],
    outputs_per_frame: dict[int, dict[str, object]],
) -> tuple[int, int, float | None]:
    target_obj_id, target_score = pick_target_obj_id(prompt_outputs)
    if target_obj_id is not None:
        return target_obj_id, 0, target_score

    for frame_idx in sorted(outputs_per_frame):
        target_obj_id, target_score = pick_target_obj_id(outputs_per_frame[frame_idx])
        if target_obj_id is not None:
            return target_obj_id, frame_idx, target_score

    raise RuntimeError("SAM3 Video found no objects for the provided text prompt")
