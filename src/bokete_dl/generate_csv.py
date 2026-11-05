"""Generate theme-specific prompts and save them to one CSV file."""
import csv
import json
import tempfile
from pathlib import Path

from . import generate
from .io import metadata_path, read_jsonl, write_json
from .similarity import normalize


FIELDS = ("id", "type", "prompt", "theme", "status", "similarity_suspected", "max_similarity")


def generate_csv(model, mode, themes, count, seed, splits_path, out, threshold=0.7):
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    themes = [theme.strip() for theme in themes]
    if not themes or any(not theme for theme in themes):
        raise ValueError("at least one nonempty theme is required")
    if len({normalize(theme) for theme in themes}) != len(themes):
        raise ValueError("themes must be unique")
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    if mode == "examples" and not 3 <= sum(bool(r.get("example")) for r in splits["parts"]["examples"]) <= 5:
        raise ValueError("select 3 to 5 examples in the examples split")
    dest = Path(out)
    meta_path = metadata_path(dest)
    if dest.exists() or meta_path.exists():
        raise FileExistsError("output or metadata already exists")
    model_details = generate.llm.model_info(model)
    dest.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="bokete-csv-", dir=dest.parent) as temp:
        with dest.open("x", encoding="utf-8-sig", newline="", buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            f.flush()
            for index, theme in enumerate(themes):
                path = Path(temp) / f"theme-{index:03d}.jsonl"
                result = {"theme": theme, "requested": count, "seed": seed + index * 1000000}
                try:
                    generate.generate(model, mode, theme, count, result["seed"], splits_path, path, threshold)
                    result["status"] = "ok"
                except (ValueError, OSError, KeyError, generate.llm.LLMError) as exc:
                    result.update(status="error", error=str(exc))
                rows = read_jsonl(path) if path.exists() else []
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in FIELDS})
                    f.flush()
                result["success"] = len(rows)
                sidecar = metadata_path(path)
                if sidecar.exists():
                    result["generation"] = json.loads(sidecar.read_text(encoding="utf-8"))
                results.append(result)
                print(f"{theme}: {len(rows)}/{count} ({result['status']})", flush=True)
    write_json(meta_path, {"mode": mode, "model": model_details, "seed": seed,
                           "dataset_revision": splits.get("dataset_revision"), "results": results})
    if any(result["status"] != "ok" for result in results):
        raise ValueError("some themes failed; see CSV metadata for details")
