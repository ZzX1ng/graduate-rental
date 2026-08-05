"""Build gold label rows for edge-probing test eval from test.jsonl (aligned by guid)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np


def _normalize_label_field(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]


def load_test_lines(test_jsonl: Path) -> list:
    lines = []
    with test_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def parse_guid(guid: str) -> tuple[int, int]:
    parts = guid.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected guid format: {guid!r}")
    return int(parts[1]), int(parts[2])


def gold_row_for_guid(
    guid: str,
    test_lines: Sequence[dict],
    label_to_id: dict,
    label_num: int,
) -> np.ndarray:
    line_num, target_num = parse_guid(guid)
    if line_num < 0 or line_num >= len(test_lines):
        raise IndexError(f"guid {guid}: line_num {line_num} out of range [0, {len(test_lines)})")
    targets = test_lines[line_num].get("targets") or []
    if target_num < 0 or target_num >= len(targets):
        raise IndexError(
            f"guid {guid}: target_num {target_num} out of range [0, {len(targets)}) for line {line_num}"
        )
    raw_labels = targets[target_num].get("label")
    names = _normalize_label_field(raw_labels)
    row = np.zeros((label_num,), dtype=int)
    for name in names:
        if name not in label_to_id:
            raise KeyError(f"Unknown label {name!r} for guid {guid} (task vocab mismatch)")
        row[label_to_id[name]] = 1
    return row


def build_gold_matrix_from_test_cache(task, test_cache, test_lines: Sequence[dict]) -> np.ndarray:
    """Same row order as ``test_cache.iter_all()``."""
    rows: List[np.ndarray] = []
    label_to_id = task.LABEL_TO_ID
    label_num = len(task.LABELS)
    for datum in test_cache.iter_all():
        guid = datum["data_row"].guid
        rows.append(gold_row_for_guid(guid, test_lines, label_to_id, label_num))
    if len(rows) != len(test_cache):
        raise RuntimeError(f"Row count mismatch: gold {len(rows)} vs cache {len(test_cache)}")
    return np.stack(rows, axis=0)
