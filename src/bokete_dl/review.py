from collections import Counter
import json
import random
import secrets
from pathlib import Path

from .io import read_jsonl, write_json

DIMENSIONS = ("clarity", "answer_space", "surprise", "independent_understanding", "safety")


def export(input_paths, seed, out, reviewers=2, answers_path=None, show_answers=False):
    if reviewers < 2:
        raise ValueError("at least two reviewers are required")
    candidates = []
    generation_stats = []
    for path in input_paths:
        sidecar = Path(str(path) + ".meta.json")
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        if "mode" in meta:
            generation_stats.append({"mode": meta["mode"], "counts": meta.get("counts", {})})
        for row in read_jsonl(path):
            row.setdefault("mode", meta.get("mode", "human"))
            candidates.append(row)
    if not candidates or len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("candidates must be nonempty and IDs unique")
    answers = {}
    if show_answers:
        if not answers_path:
            raise ValueError("--answers is required with --show-answers")
        for row in read_jsonl(answers_path):
            answers.setdefault(row["candidate_id"], []).extend(row.get("answers") or [])
    dest = Path(out)
    if dest.exists():
        raise FileExistsError(f"review directory already exists: {dest}")
    dest.mkdir(parents=True)
    rng = random.Random(seed)
    # Unpredictable IDs avoid revealing source order even when the shuffle seed is known.
    ids = [secrets.token_hex(8) for _ in candidates]
    mapping = {key: {"candidate_id": c["id"], "mode": c.get("mode", "unknown"), "theme": c.get("theme"), "similarity_suspected": c.get("similarity_suspected")} for key, c in zip(ids, candidates)}
    write_json(dest / "mapping.json", mapping)
    for i in range(1, reviewers + 1):
        entries = []
        for key, c in zip(ids, candidates):
            entry = {"review_id": key, "prompt": c["prompt"], "scores": {k: None for k in DIMENSIONS}, "two_answers_possible": None, "decision": "pending", "reason": "", "rank": None, "rank_status": "pending"}
            if show_answers:
                entry["trial_answers"] = answers.get(c["id"], [])
            entries.append(entry)
        rng.shuffle(entries)
        write_json(dest / f"reviewer_{i}.json", entries)
    write_json(dest / "settings.json", {"seed": seed, "reviewers": reviewers, "show_answers": show_answers, "generation_stats": generation_stats})


def summarize(input_path, out):
    directory = Path(input_path)
    mapping = json.loads((directory / "mapping.json").read_text(encoding="utf-8"))
    files = sorted(directory.glob("reviewer_*.json"))
    if len(files) < 2:
        raise ValueError("at least two reviewer files are required")
    reviews = []
    for file in files:
        rows = json.loads(file.read_text(encoding="utf-8"))
        ids = [r["review_id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(mapping):
            raise ValueError(f"review IDs missing or duplicated: {file.name}")
        for r in rows:
            if r.get("decision") not in ("approve", "reject") or (r["decision"] == "reject" and not r.get("reason", "").strip()):
                raise ValueError(f"unfinished decision or rejection reason: {file.name}")
            if type(r.get("two_answers_possible")) is not bool or any(type(r.get("scores", {}).get(k)) is not int or not 1 <= r["scores"][k] <= 5 for k in DIMENSIONS):
                raise ValueError(f"unfinished scores or answerability: {file.name}")
            if r.get("rank_status") not in ("ranked", "tie", "undecidable") or (r["rank_status"] in ("ranked", "tie") and (type(r.get("rank")) is not int or r["rank"] < 1)):
                raise ValueError(f"unfinished rank: {file.name}")
        reviews.append({r["review_id"]: r for r in rows})
    all_rows = [r for review in reviews for r in review.values()]
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    stats = settings.get("generation_stats", [])
    valid_batches = sum(s["counts"].get("valid_batches", 0) for s in stats)
    format_errors = sum(s["counts"].get("format_errors", 0) for s in stats)
    agreed = sum(len({review[k]["decision"] for review in reviews}) == 1 for k in mapping)
    disagreements = [k for k in mapping if len({review[k]["decision"] for review in reviews}) > 1]
    suspected_known = [m for m in mapping.values() if isinstance(m.get("similarity_suspected"), bool)]
    report = {"candidates": len(mapping), "reviewers": len(reviews), "ratings": len(all_rows), "approval_rate": sum(r["decision"] == "approve" for r in all_rows) / len(all_rows), "answerable_rate": sum(r["two_answers_possible"] for r in all_rows) / len(all_rows), "all_reviewers_answerable_rate": sum(all(review[k]["two_answers_possible"] for review in reviews) for k in mapping) / len(mapping), "decision_agreement_rate": agreed / len(mapping), "disagreements": disagreements, "rank_distribution": dict(Counter(str(r["rank"]) if r["rank_status"] == "ranked" else r["rank_status"] for r in all_rows)), "score_distribution": {k: dict(Counter(str(r["scores"][k]) for r in all_rows)) for k in DIMENSIONS}, "duplicate_suspicion_rate": sum(m["similarity_suspected"] for m in suspected_known) / len(suspected_known) if suspected_known else None, "duplicate_suspicion_count": len(suspected_known), "format_pass_rate": valid_batches / (valid_batches + format_errors) if valid_batches + format_errors else None, "format_pass_count": valid_batches, "format_failure_count": format_errors, "format_pass_rate_note": "Response-level rate, including retries; unavailable for inputs without generation metadata.", "final_decisions": {k: "pending discussion" for k in disagreements}}
    write_json(out, report)
