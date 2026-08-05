#!/usr/bin/env python3
import argparse
import itertools
import json
import os
from collections import defaultdict
from glob import glob


SPLIT_MAP = {
    "train": "train",
    "development": "val",
    "test": "test",
    "conll-2012-test": "test",
}

ALLOWED_POS = {
    "$",
    "''",
    ",",
    "-LRB-",
    "-RRB-",
    ".",
    ":",
    "ADD",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NFP",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "``",
}


def parse_constituents(parse_bits):
    stack = []
    spans = []
    for i, bit in enumerate(parse_bits):
        for part in bit.split("*")[0].split("("):
            label = part.strip()
            if label:
                stack.append((label, i))
        for _ in range(bit.count(")")):
            if stack:
                label, start = stack.pop()
                spans.append((label, start, i + 1))
    return spans


def parse_srl_column(col_values):
    stack = []
    spans = []
    for i, tag in enumerate(col_values):
        open_labels = []
        cur = ""
        in_open = False
        for ch in tag:
            if ch == "(":
                in_open = True
                cur = ""
            elif in_open and ch in ("*", ")"):
                if cur:
                    open_labels.append(cur)
                in_open = False
            elif in_open:
                cur += ch
        for label in open_labels:
            stack.append((label, i))
        for _ in range(tag.count(")")):
            if stack:
                label, start = stack.pop()
                spans.append((label, start, i + 1))
    return spans


def parse_coref_column(values):
    open_clusters = defaultdict(list)
    mentions = []
    for i, raw in enumerate(values):
        if raw == "-":
            continue
        for part in raw.split("|"):
            part = part.strip()
            if not part:
                continue
            if part.startswith("(") and part.endswith(")"):
                cid = part[1:-1]
                mentions.append((cid, i, i + 1))
            elif part.startswith("("):
                cid = part[1:]
                open_clusters[cid].append(i)
            elif part.endswith(")"):
                cid = part[:-1]
                if open_clusters[cid]:
                    start = open_clusters[cid].pop()
                    mentions.append((cid, start, i + 1))
    return mentions


def read_sentences(conll_file):
    sentences = []
    sent = []
    with open(conll_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    sentences.append(sent)
                    sent = []
                continue
            if line.startswith("#begin") or line.startswith("#end"):
                continue
            sent.append(line.split())
    if sent:
        sentences.append(sent)
    return sentences


def convert_split(split_dir, split_name, out_root):
    conll_files = glob(os.path.join(split_dir, "data", "english", "annotations", "*", "*", "*", "*.gold_conll"))
    pos_path = os.path.join(out_root, "pos", f"{split_name}.jsonl")
    nonterm_path = os.path.join(out_root, "nonterminal", f"{split_name}.jsonl")
    srl_path = os.path.join(out_root, "srl", f"{split_name}.jsonl")
    coref_path = os.path.join(out_root, "coref", f"{split_name}.jsonl")
    for p in [pos_path, nonterm_path, srl_path, coref_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(pos_path, "w", encoding="utf-8") as f_pos, open(
        nonterm_path, "w", encoding="utf-8"
    ) as f_non, open(srl_path, "w", encoding="utf-8") as f_srl, open(coref_path, "w", encoding="utf-8") as f_coref:
        for conll_file in conll_files:
            rel = conll_file.split("/data/", 1)[1]
            for sent_id, rows in enumerate(read_sentences(conll_file)):
                words = [r[3] for r in rows]
                pos_tags = [r[4] for r in rows]
                parse_bits = [r[5] for r in rows]
                predicates = [i for i, r in enumerate(rows) if r[6] != "-"]
                coref_col = [r[-1] for r in rows]
                srl_cols = [list(col) for col in zip(*[r[11:-1] for r in rows])] if rows and len(rows[0]) > 12 else []

                info = {"source": rel, "split": split_name, "sentence_id": sent_id}
                pos_targets = [{"span1": [i, i + 1], "label": t} for i, t in enumerate(pos_tags) if t in ALLOWED_POS]
                f_pos.write(
                    json.dumps(
                        {
                            "text": " ".join(words),
                            "info": info,
                            "targets": pos_targets,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                const_spans = parse_constituents(parse_bits)
                f_non.write(
                    json.dumps(
                        {
                            "text": " ".join(words),
                            "info": info,
                            "targets": [{"span1": [s, e], "label": l} for l, s, e in const_spans if l != "TOP"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                for pred_idx, col in zip(predicates, srl_cols):
                    spans = parse_srl_column(col)
                    targets = []
                    head = [pred_idx, pred_idx + 1]
                    for label, s, e in spans:
                        if label == "V":
                            continue
                        targets.append({"span1": [s, e], "span2": head, "label": label})
                    f_srl.write(
                        json.dumps({"text": " ".join(words), "info": info, "targets": targets}, ensure_ascii=False)
                        + "\n"
                    )

                mentions = parse_coref_column(coref_col)
                pairs = []
                for (cid1, s1, e1), (cid2, s2, e2) in itertools.combinations(mentions, 2):
                    pairs.append({"span1": [s1, e1], "span2": [s2, e2], "label": str(int(cid1 == cid2))})
                f_coref.write(
                    json.dumps({"text": " ".join(words), "info": info, "targets": pairs}, ensure_ascii=False) + "\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontonotes_conll_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    for raw_split, out_split in SPLIT_MAP.items():
        split_dir = os.path.join(args.ontonotes_conll_root, "data", raw_split)
        convert_split(split_dir, out_split, args.output_root)


if __name__ == "__main__":
    main()
