import json
import time
import uuid
from collections import Counter
from pathlib import Path

from . import llm, prompts
from .io import line, open_jsonl, read_jsonl, write_json, metadata_path
from .similarity import normalize, similarity


def validate(value, count, lengths=True):
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list) or len(items) != count or any(not isinstance(x, str) or not x.strip() or (lengths and not 10 <= len(x.strip()) <= 100) for x in items):
        raise ValueError("invalid items structure, count, or length")
    return [x.strip() for x in items]


def call(messages, model, seed, count, lengths=True):
    errors = []
    for retry in range(3):
        try:
            return validate(llm.generate_json(messages, model, {"seed": seed + retry * 10000000}), count, lengths), errors
        except (ValueError, llm.LLMError) as e:
            errors.append(str(e))
    return None, errors


def generate(model, mode, theme, count, seed, splits_path, out, threshold=0.7):
    if count <= 0:
        raise ValueError("count must be positive")
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    parts = splits["parts"]
    examples = [r["prompt"] for r in parts["examples"] if r.get("example")]
    if mode == "examples" and not 3 <= len(examples) <= 5:
        raise ValueError("select 3 to 5 examples by setting example=true in the examples split")
    if mode != "examples":
        examples = []
    existing = [r["prompt"] for name in ("examples", "development") for r in parts[name]]
    if Path(out).exists() or metadata_path(out).exists():
        raise FileExistsError("output or metadata already exists")
    info = llm.model_info(model)
    start = time.monotonic()
    counts = Counter()
    accepted = []
    run_id = uuid.uuid4().hex
    max_batches = max(6, ((count + 4) // 5) * 4)
    errors = []
    try:
        with open_jsonl(out) as f:
            for batch in range(max_batches):
                if len(accepted) >= count:
                    break
                needed = min(5, count - len(accepted))
                items, failures = call(prompts.generation(theme, needed, examples), model, seed + batch, needed)
                counts["format_errors"] += len(failures)
                errors.extend(failures)
                if items is None:
                    counts["failed_batches"] += 1
                    continue
                counts["valid_batches"] += 1
                for item in items:
                    if normalize(item) in {normalize(x) for x in existing + accepted}:
                        counts["rejected_exact"] += 1
                        continue
                    near = max((similarity(item, x) for x in existing + accepted), default=0)
                    accepted.append(item)
                    line(f, {"id": f"{run_id}-{len(accepted):04d}", "type": "text_to_text", "prompt": item, "theme": theme, "status": "pending", "similarity_suspected": near >= threshold, "max_similarity": round(near, 4)})
                    counts["similarity_suspected"] += near >= threshold
    finally:
        write_json(metadata_path(out), {"mode": mode, "model": info, "settings": {"temperature": .8, "top_p": .9, "num_predict": 512, "seed": seed, "batch_size": 5, "similarity_threshold": threshold}, "dataset_revision": splits.get("dataset_revision"), "elapsed_seconds": round(time.monotonic() - start, 2), "requested": count, "success": len(accepted), "counts": dict(counts), "errors": errors})
    if len(accepted) < count:
        raise ValueError(f"generation incomplete: {count - len(accepted)} missing")


def answer(model, input_path, seed, out):
    candidates = read_jsonl(input_path)
    if len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate candidate IDs")
    if Path(out).exists() or metadata_path(out).exists():
        raise FileExistsError("output or metadata already exists")
    info = llm.model_info(model)
    errors = 0
    start = time.monotonic()
    with open_jsonl(out) as f:
        for candidate in candidates:
            seen = set()
            for attempt in range(1, 4):
                actual_seed = seed + attempt - 1
                items, failures = call(prompts.answer(candidate["prompt"]), model, actual_seed, 3, lengths=False)
                errors += bool(items is None)
                unique = []
                for item in items or []:
                    n = normalize(item)
                    if n and n not in seen:
                        unique.append(item)
                        seen.add(n)
                line(f, {"candidate_id": candidate["id"], "attempt": attempt, "seed": actual_seed, "answers": unique if items is not None else None, "unique_count": len(unique) if items is not None else None, "errors": failures, "status": "ok" if items is not None else "error"})
    write_json(metadata_path(out), {"model": info, "seed": seed, "elapsed_seconds": round(time.monotonic() - start, 2), "failed_attempts": errors})
    if errors:
        raise ValueError(f"{errors} answer attempts failed")
