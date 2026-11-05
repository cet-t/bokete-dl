from collections import Counter, defaultdict
from datetime import datetime, timezone
import random
import re

from .io import line, open_jsonl, read_jsonl, write_json, metadata_path
from .similarity import normalize, similarity

DATASET = "YANS-official/ogiri-bokete"
TYPES = {"text_to_text", "image_to_text", "image_text_to_text"}


def prepare(revision, out, threshold=0.7):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full 40-character dataset commit hash")
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", revision=revision)
    counts, missing, ids, rows = Counter(), Counter(), set(), []
    duplicates = []
    for row in ds:
        typ, ident, prompt = row.get("type"), row.get("odai_id"), row.get("odai")
        counts[str(typ)] += 1
        for key in ("type", "odai_id", "odai"):
            if row.get(key) is None or row.get(key) == "":
                missing[key] += 1
        if not isinstance(ident, str) or not ident:
            raise ValueError("odai_id must be a nonempty string")
        if ident in ids:
            duplicates.append(ident)
        ids.add(ident)
        if typ not in TYPES:
            raise ValueError(f"unknown type: {typ!r}; counts so far: {dict(counts)}")
        if typ == "text_to_text":
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"missing text prompt for ID {ident}")
            rows.append({"id": ident, "prompt": prompt, "normalized": normalize(prompt), "decision": "pending", "reason": "", "group_id": "", "example": False})
    if duplicates or missing:
        raise ValueError(f"dataset invalid: missing={dict(missing)}, duplicate_ids={duplicates}")
    pairs = [{"id1": a["id"], "id2": b["id"], "similarity": round(similarity(a["prompt"], b["prompt"]), 4)} for i, a in enumerate(rows) for b in rows[i+1:] if similarity(a["prompt"], b["prompt"]) >= threshold]
    # Create sidecars first only after verifying all destinations are available.
    from pathlib import Path
    paths = [Path(out), metadata_path(out), Path(str(out) + ".duplicates.json")]
    if any(p.exists() for p in paths):
        raise FileExistsError("output or sidecar already exists")
    with open_jsonl(out) as f:
        for row in rows:
            line(f, row)
    write_json(paths[2], {"threshold": threshold, "pairs": pairs, "instructions": "Review pairs and assign matching group_id values in the JSONL."})
    write_json(paths[1], {"dataset": DATASET, "revision": revision, "timestamp": datetime.now(timezone.utc).isoformat(), "total": len(ds), "types": dict(counts), "missing": dict(missing), "duplicate_ids": duplicates, "text_count": len(rows)})


def split(input_path, seed, out):
    rows = read_jsonl(input_path)
    if not rows or any(r.get("decision") not in ("include", "exclude") for r in rows):
        raise ValueError("all records must be reviewed as include or exclude")
    if any(r["decision"] == "exclude" and not r.get("reason", "").strip() for r in rows):
        raise ValueError("excluded records require a reason")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs")
    included = [r for r in rows if r["decision"] == "include"]
    # Exact matches must stay together even if reviewers omitted a group.
    groups = defaultdict(list)
    for r in included:
        if normalize(r["prompt"]) != r.get("normalized"):
            raise ValueError("normalized text does not match prompt")
        groups[r.get("group_id") or "id:" + r["id"]].append(r)
    seen = {}
    for gid, members in groups.items():
        for r in members:
            n = normalize(r["prompt"])
            if n in seen and seen[n] != gid:
                raise ValueError("exact duplicates must share group_id")
            seen[n] = gid
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    parts = {"examples": [], "development": [], "evaluation": []}
    # Allocate by groups, keeping the random order stable.
    n = len(keys)
    for i, gid in enumerate(keys):
        name = "examples" if i < round(n * .6) else "development" if i < round(n * .8) else "evaluation"
        parts[name].extend({"id": r["id"], "group_id": gid, "prompt": r["prompt"], "example": bool(r.get("example", False))} for r in groups[gid])
    from pathlib import Path
    import json
    meta_file = metadata_path(input_path)
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    write_json(out, {"dataset_revision": meta.get("revision"), "seed": seed, "parts": parts, "counts": {k: len(v) for k, v in parts.items()}, "excluded_reasons": dict(Counter(r["reason"] for r in rows if r["decision"] == "exclude")), "timestamp": datetime.now(timezone.utc).isoformat()})
