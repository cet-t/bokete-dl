import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def open_jsonl(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8", buffering=1)


def line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def metadata_path(path):
    return Path(str(path) + ".meta.json")
