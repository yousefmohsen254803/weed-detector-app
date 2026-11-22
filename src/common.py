"""Shared helpers for the weed-seedling detector pipeline."""
from pathlib import Path
import json

CLASSES = ["weed_seedling", "fallen_leaf", "moss"]
COLORS_BGR = {0: (60, 220, 60), 1: (0, 190, 255), 2: (230, 60, 230)}  # seedling green, leaf orange, moss magenta
ROOT = Path(__file__).resolve().parent.parent


def load_annotations(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iou(a, b):
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(img, boxes, thickness=3, font_scale=0.8):
    """boxes: list of dicts with class_id, xyxy and optional conf. Draws in place."""
    import cv2
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b["xyxy"]]
        c = COLORS_BGR[b["class_id"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness)
        label = CLASSES[b["class_id"]] + (f" {b['conf']:.2f}" if "conf" in b else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), c, -1)
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    return img
