from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .geometry import Box, iou


def greedy_match(
    truth: Iterable[Box], predictions: Iterable[Box], iou_threshold: float = 0.5
) -> tuple[list[dict], list[dict]]:
    truth_list = list(truth)
    prediction_list = sorted(predictions, key=lambda box: box.confidence, reverse=True)
    matched_truth: set[int] = set()
    prediction_rows: list[dict] = []

    for prediction in prediction_list:
        candidates = [
            (index, iou(prediction, target))
            for index, target in enumerate(truth_list)
            if index not in matched_truth and target.class_id == prediction.class_id
        ]
        best_index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        is_true_positive = best_index >= 0 and best_iou >= iou_threshold
        if is_true_positive:
            matched_truth.add(best_index)
        prediction_rows.append(
            {
                **asdict(prediction),
                "iou": best_iou,
                "outcome": "tp" if is_true_positive else "fp",
            }
        )

    missed_rows = [
        {**asdict(target), "iou": 0.0, "outcome": "fn"}
        for index, target in enumerate(truth_list)
        if index not in matched_truth
    ]
    return prediction_rows, missed_rows


def _average_precision(rows: list[dict], positives: int) -> float:
    if positives == 0:
        return float("nan")
    ordered = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    tp = np.cumsum([row["outcome"] == "tp" for row in ordered])
    fp = np.cumsum([row["outcome"] == "fp" for row in ordered])
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    change = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[change + 1] - recall[change]) * precision[change + 1]))


def evaluate(
    ground_truth: Iterable[Box],
    predictions: Iterable[Box],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    truth_by_image: defaultdict[str, list[Box]] = defaultdict(list)
    prediction_by_image: defaultdict[str, list[Box]] = defaultdict(list)
    for box in ground_truth:
        truth_by_image[box.image_id].append(box)
    for box in predictions:
        prediction_by_image[box.image_id].append(box)

    prediction_rows: list[dict] = []
    missed_rows: list[dict] = []
    image_ids = sorted(set(truth_by_image) | set(prediction_by_image))
    for image_id in image_ids:
        matched, missed = greedy_match(
            truth_by_image[image_id], prediction_by_image[image_id], iou_threshold
        )
        prediction_rows.extend(matched)
        missed_rows.extend(missed)

    all_rows = prediction_rows + missed_rows
    errors = pd.DataFrame(all_rows)
    per_class_rows = []
    for class_id, name in class_names.items():
        class_predictions = [row for row in prediction_rows if row["class_id"] == class_id]
        positives = sum(box.class_id == class_id for boxes in truth_by_image.values() for box in boxes)
        tp = sum(row["outcome"] == "tp" for row in class_predictions)
        fp = sum(row["outcome"] == "fp" for row in class_predictions)
        fn = sum(row["class_id"] == class_id for row in missed_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": _average_precision(class_predictions, positives),
            }
        )
    per_class = pd.DataFrame(per_class_rows)
    page_successes = []
    for image_id in image_ids:
        wally_truth = [box for box in truth_by_image[image_id] if class_names.get(box.class_id, "").lower() in {"wally", "waldo"}]
        if not wally_truth:
            continue
        matches, _ = greedy_match(wally_truth, prediction_by_image[image_id], iou_threshold)
        page_successes.append(any(row["outcome"] == "tp" for row in matches))
    summary = {
        "iou_threshold": iou_threshold,
        "images": len(image_ids),
        "page_success_rate": float(np.mean(page_successes)) if page_successes else float("nan"),
        "map50": float(per_class["ap50"].mean()) if not per_class.empty else float("nan"),
    }
    return per_class, errors, summary

