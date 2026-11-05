"""Generate sets of prompts for several themes in a single run."""
import json
from pathlib import Path

from . import generate
from .io import metadata_path, write_json
from .similarity import normalize


def generate_themes(model, mode, themes, count, seed, splits_path, out, threshold=0.7):
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    themes = [theme.strip() for theme in themes]
    if not themes or any(not theme for theme in themes):
        raise ValueError("at least one nonempty theme is required")
    if len({normalize(theme) for theme in themes}) != len(themes):
        raise ValueError("themes must be unique")
    # Validate input and model before creating output directories.
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    parts = splits["parts"]
    if mode == "examples" and not 3 <= sum(bool(r.get("example")) for r in parts["examples"]) <= 5:
        raise ValueError("select 3 to 5 examples in the examples split")
    generate.llm.model_info(model)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=False)
    results = []
    for index, theme in enumerate(themes, 1):
        path = dest / f"theme-{index:03d}.jsonl"
        result = {"theme": theme, "file": path.name, "requested": count}
        try:
            # Separate seeds for themes without making their batch sequences overlap.
            generate.generate(model, mode, theme, count, seed + (index - 1) * 1000000,
                              splits_path, path, threshold)
            result["status"] = "ok"
        except (ValueError, OSError, KeyError, generate.llm.LLMError) as exc:
            result.update(status="error", error=str(exc))
        sidecar = metadata_path(path)
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            result["success"] = meta.get("success", 0)
        else:
            result["success"] = 0
        results.append(result)
        print(f"{theme}: {result['success']}/{count} ({result['status']})", flush=True)
    write_json(dest / "summary.json", {"mode": mode, "model": model, "seed": seed,
                                       "dataset_revision": splits.get("dataset_revision"),
                                       "results": results})
    if any(r["status"] != "ok" for r in results):
        raise ValueError("some themes failed; see summary.json for details")
