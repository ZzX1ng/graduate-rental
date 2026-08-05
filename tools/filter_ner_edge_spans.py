#!/usr/bin/env python3
"""Filter NER edge-probing targets that fall beyond BERT's truncated input."""

import argparse
import json
import shutil
from pathlib import Path

from transformers import BertTokenizer

from jiant.utils import retokenize
from jiant.utils.tokenization_normalization import normalize_tokenizations


def build_wordpiece_aligner(text, tokenizer):
    space_tokenization = text.split()
    target_tokenization = tokenizer.tokenize(text)
    normed_space_tokenization, normed_target_tokenization = normalize_tokenizations(
        space_tokenization, target_tokenization, tokenizer
    )
    return retokenize.TokenAligner(normed_space_tokenization, normed_target_tokenization)


def target_survives_truncation(aligner, span, max_wordpiece_tokens):
    """Return whether a word-level target span remains valid after wordpiece truncation."""
    target_span = aligner.project_token_span(span[0], span[1])
    return target_span[1] <= max_wordpiece_tokens


def filter_file(path, tokenizer, max_seq_length, dry_run=False):
    max_wordpiece_tokens = max_seq_length - 2  # [CLS], [SEP]
    rows = []
    removed = 0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            kept_targets = []
            aligner = build_wordpiece_aligner(row["text"], tokenizer)
            for target in row.get("targets", []):
                total += 1
                if target_survives_truncation(
                    aligner=aligner,
                    span=target["span1"],
                    max_wordpiece_tokens=max_wordpiece_tokens,
                ):
                    kept_targets.append(target)
                else:
                    removed += 1
            row["targets"] = kept_targets
            rows.append(row)

    if not dry_run:
        backup_path = path.with_suffix(path.suffix + ".pre_span_filter")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default="bert-large-uncased")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model)
    grand_total = 0
    grand_removed = 0
    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        total, removed = filter_file(
            path=path,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            dry_run=args.dry_run,
        )
        grand_total += total
        grand_removed += removed
        print(f"{split}: removed {removed} / {total} targets")
    print(f"total: removed {grand_removed} / {grand_total} targets")


if __name__ == "__main__":
    main()
