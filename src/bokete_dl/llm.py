import json
import requests

BASE = "http://127.0.0.1:11434"
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}, "required": ["items"]}


class LLMError(Exception):
    pass


def request(method, path, **kwargs):
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.request(method, BASE + path, allow_redirects=False, timeout=120, **kwargs)
        if not 200 <= response.status_code < 300:
            try:
                detail = response.json().get("error", "")
            except (ValueError, AttributeError):
                detail = ""
            if any(s in detail.lower() for s in ("out of memory", "not enough memory", "requires more system memory")):
                raise LLMError("Ollama memory shortage")
            raise LLMError(f"Ollama HTTP error: {response.status_code}")
        try:
            return response.json()
        except ValueError as e:
            raise LLMError("Ollama response JSON parse error") from e
    except requests.Timeout as e:
        raise LLMError("Ollama timeout") from e
    except requests.ConnectionError as e:
        raise LLMError("Ollama connection unavailable") from e
    except requests.RequestException as e:
        raise LLMError(f"Ollama request failed: {type(e).__name__}") from e
    finally:
        session.close()


def model_info(model):
    if model.lower().endswith(":cloud"):
        raise LLMError("cloud models are not permitted; use a locally installed model")
    tags = request("GET", "/api/tags")
    found = next((m for m in tags.get("models", []) if m.get("name") == model or m.get("model") == model), None)
    if found is None:
        raise LLMError(f"local model not found: {model}")
    if found.get("remote_model") or found.get("remote_host") or (found.get("details") or {}).get("remote_model"):
        raise LLMError("cloud models are not permitted; use a locally installed model")
    version = request("GET", "/api/version").get("version")
    return {"name": model, "digest": found.get("digest"), "details": found.get("details"), "ollama_version": version}


def generate_json(messages, model, options):
    result = request("POST", "/api/chat", json={"model": model, "messages": messages, "format": SCHEMA, "stream": False, "options": {"temperature": 0.8, "top_p": 0.9, "num_predict": 512, **options}})
    try:
        value = json.loads(result["message"]["content"])
    except (KeyError, TypeError, ValueError) as e:
        raise LLMError("generated JSON parse error") from e
    if not isinstance(value, dict):
        raise LLMError("generated JSON must be an object")
    return value
