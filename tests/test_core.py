import json
import pytest
from bokete_dl import data, generate, review, llm
from bokete_dl.similarity import normalize, similarity


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def test_normalize_and_similarity():
    assert normalize(' Ａ　 B\n') == 'A B'
    assert similarity('abc', 'abc') == 1


def test_split_groups_and_pending(tmp_path):
    p = tmp_path / 'review.jsonl'
    rows = [{'id': str(i), 'prompt': 'お題' + str(i // 2), 'normalized': normalize('お題' + str(i // 2)), 'decision': 'include', 'reason': '', 'group_id': str(i // 2)} for i in range(20)]
    write_lines(p, rows)
    data.split(p, 42, tmp_path / 'a.json')
    data.split(p, 42, tmp_path / 'b.json')
    a = json.loads((tmp_path / 'a.json').read_text(encoding='utf-8'))
    b = json.loads((tmp_path / 'b.json').read_text(encoding='utf-8'))
    assert a['parts'] == b['parts']
    groups = [set(r['group_id'] for r in part) for part in a['parts'].values()]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    rows[0]['decision'] = 'pending'
    write_lines(p, rows)
    with pytest.raises(ValueError): data.split(p, 42, tmp_path / 'c.json')


def test_retry_limit(monkeypatch):
    calls = []
    def bad(*args):
        calls.append(1)
        return {'items': []}
    monkeypatch.setattr(llm, 'generate_json', bad)
    items, errors = generate.call([], 'm', 1, 5)
    assert items is None and len(errors) == 3 and len(calls) == 3


def test_generation_duplicates_and_ids(tmp_path, monkeypatch):
    split = tmp_path / 'splits.json'
    existing = 'これは既存のお題のテキストですか？'
    split.write_text(json.dumps({'parts': {'examples': [{'prompt': existing}], 'development': [], 'evaluation': []}}), encoding='utf-8')
    monkeypatch.setattr(llm, 'model_info', lambda model: {'name': model})
    responses = iter([{'items': [existing, 'これは新しいお題のテキストですか？']}, {'items': ['全く別のお題をここに書きましたか？']}])
    monkeypatch.setattr(llm, 'generate_json', lambda *args: next(responses))
    out = tmp_path / 'c.jsonl'
    generate.generate('m', 'no-examples', '', 2, 42, split, out)
    rows = [json.loads(x) for x in out.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 2
    assert existing not in [r['prompt'] for r in rows]


def test_review_blind_and_summary(tmp_path):
    inp = tmp_path / 'c.jsonl'
    write_lines(inp, [{'id': 'secret', 'prompt': 'お題です', 'mode': 'examples', 'similarity_suspected': False}])
    d = tmp_path / 'review'
    review.export([inp], 42, d)
    for p in d.glob('reviewer_*.json'):
        assert 'secret' not in p.read_text(encoding='utf-8') and 'examples' not in p.read_text(encoding='utf-8')
        rows = json.loads(p.read_text(encoding='utf-8'))
        rows[0].update(decision='approve', two_answers_possible=True, rank=1, rank_status='ranked', scores={k: 4 for k in review.DIMENSIONS})
        p.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    report = tmp_path / 'report.json'
    review.summarize(d, report)
    assert json.loads(report.read_text(encoding='utf-8'))['approval_rate'] == 1

def test_ollama_session_ignores_proxy_and_redirects(monkeypatch):
    seen = {}
    class Response:
        status_code = 200
        def json(self): return {'models': []}
    class Session:
        trust_env = True
        def request(self, method, url, **kwargs):
            seen.update(url=url, trust_env=self.trust_env, **kwargs)
            return Response()
        def close(self): pass
    monkeypatch.setattr(llm.requests, 'Session', Session)
    llm.request('GET', '/api/tags')
    assert seen['url'] == 'http://127.0.0.1:11434/api/tags'
    assert seen['trust_env'] is False and seen['allow_redirects'] is False


def test_missing_local_model(monkeypatch):
    monkeypatch.setattr(llm, 'request', lambda *args, **kwargs: {'models': []})
    with pytest.raises(llm.LLMError, match='local model not found'):
        llm.model_info('missing')


def test_cloud_model_rejected(monkeypatch):
    def no_request(*args, **kwargs):
        raise AssertionError("must reject before HTTP request")
    monkeypatch.setattr(llm, "request", no_request)
    with pytest.raises(llm.LLMError, match="cloud models"):
        llm.model_info("deepseek-v4.1-flash:cloud")

def test_generate_themes_outputs_and_partial_failure(tmp_path, monkeypatch):
    from bokete_dl import bulk
    split = tmp_path / 'splits.json'
    split.write_text(json.dumps({'parts': {'examples': [], 'development': [], 'evaluation': []}, 'dataset_revision': 'rev'}), encoding='utf-8')
    monkeypatch.setattr(llm, 'model_info', lambda model: {'name': model})
    calls = []
    def fake_generate(model, mode, theme, count, seed, splits_path, out, threshold):
        calls.append((theme, count, seed))
        if theme == '学校':
            raise ValueError('generation incomplete')
        out.write_text('{"prompt":"example"}\n', encoding='utf-8')
        generate.write_json(generate.metadata_path(out), {'success': count})
    monkeypatch.setattr(generate, 'generate', fake_generate)
    dest = tmp_path / 'bulk'
    with pytest.raises(ValueError, match='some themes failed'):
        bulk.generate_themes('m', 'no-examples', ['未来', '学校'], 100, 42, split, dest)
    summary = json.loads((dest / 'summary.json').read_text(encoding='utf-8'))
    assert calls == [('未来', 100, 42), ('学校', 100, 1000042)]
    assert [r['status'] for r in summary['results']] == ['ok', 'error']
    assert summary['results'][0]['success'] == 100
    with pytest.raises(FileExistsError):
        bulk.generate_themes('m', 'no-examples', ['未来'], 100, 42, split, dest)


def test_generate_themes_rejects_duplicate_themes(tmp_path):
    from bokete_dl import bulk
    with pytest.raises(ValueError, match='unique'):
        bulk.generate_themes('m', 'no-examples', ['未来', '未来'], 100, 1, 'unused', tmp_path / 'out')

def test_generate_csv_writes_rows_and_preserves_partial_failure(tmp_path, monkeypatch):
    import csv
    from bokete_dl import generate_csv
    split = tmp_path / 'splits.json'
    split.write_text(json.dumps({'parts': {'examples': [], 'development': [], 'evaluation': []}, 'dataset_revision': 'rev'}), encoding='utf-8')
    monkeypatch.setattr(llm, 'model_info', lambda model: {'name': model})
    def fake_generate(model, mode, theme, count, seed, splits_path, out, threshold):
        write_lines(out, [{'id': theme, 'type': 'text_to_text', 'prompt': 'カンマ,と\n改行を含むお題', 'theme': theme, 'status': 'pending', 'similarity_suspected': False, 'max_similarity': 0.0}])
        if theme == '学校':
            raise ValueError('generation incomplete')
        generate.write_json(generate.metadata_path(out), {'success': 1})
    monkeypatch.setattr(generate, 'generate', fake_generate)
    dest = tmp_path / 'prompts.csv'
    with pytest.raises(ValueError, match='some themes failed'):
        generate_csv.generate_csv('m', 'no-examples', ['未来', '学校'], 1, 42, split, dest)
    with dest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]['prompt'] == 'カンマ,と\n改行を含むお題'
    assert [r['theme'] for r in rows] == ['未来', '学校']
    meta = json.loads(generate.metadata_path(dest).read_text(encoding='utf-8'))
    assert [r['status'] for r in meta['results']] == ['ok', 'error']
    assert meta['results'][1]['success'] == 1
    with pytest.raises(FileExistsError):
        generate_csv.generate_csv('m', 'no-examples', ['未来'], 1, 42, split, dest)
