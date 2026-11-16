"""TF-IDF retrieval baseline for Day 7. Python 3.11+, standard library only."""
import argparse
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def validate(chunks, questions):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('chunks phải là JSON array không rỗng')
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions phải là JSON array không rỗng')
    ids = set()
    for c in chunks:
        for key in ('id', 'text', 'source_id', 'source', 'page', 'metadata'):
            if key not in c or c[key] is None:
                raise ValueError(f'chunk thiếu {key}')
        if not all(isinstance(c[k], str) and c[k].strip() for k in ('id', 'text', 'source_id', 'source')):
            raise ValueError('chunk id/text/source_id/source phải là chuỗi không rỗng')
        if not isinstance(c['metadata'], dict) or c['id'] in ids:
            raise ValueError('metadata phải là object và ID chunk phải duy nhất')
        ids.add(c['id'])
    ids.clear()
    for q in questions:
        if not isinstance(q.get('id'), str) or not q['id'].strip() or q['id'] in ids:
            raise ValueError('question id phải là chuỗi không rỗng và duy nhất')
        ids.add(q['id'])
        if q.get('split') not in ('dev', 'test') or not isinstance(q.get('question'), str) or not q['question'].strip():
            raise ValueError(f"question {q['id']}: split hoặc question không hợp lệ")
        if not isinstance(q.get('answerable', True), bool):
            raise ValueError(f"question {q['id']}: answerable phải là boolean")
        if q.get('answerable', True):
            if not isinstance(q.get('expected_answer'), str) or not q['expected_answer'].strip():
                raise ValueError(f"question {q['id']}: expected_answer phải là chuỗi không rỗng")
            evidence = required_evidence(q)
            if not evidence:
                raise ValueError(f"question {q['id']}: thiếu gold evidence")
            complete = (
                all(any(hit(c, [item]) for c in chunks) for item in evidence)
                if q.get('required_evidence')
                else any(any(hit(c, [item]) for c in chunks) for item in evidence)
            )
            if not complete:
                raise ValueError(f"question {q['id']}: required evidence không nằm trọn trong chunk đúng nguồn/trang")
        elif not isinstance(q.get('absence_reason'), str) or not q['absence_reason'].strip():
            raise ValueError(f"question {q['id']}: câu thiếu evidence cần absence_reason")


def required_evidence(question):
    """Return all required evidence; legacy `gold` entries are alternatives."""
    if question.get('required_evidence'):
        return question['required_evidence']
    if question.get('gold'):
        return question['gold']
    pages = question.get('gold_pages', [])
    quote = question.get('evidence_quote', '')
    return [{'source_id': question.get('gold_source_id'), 'page': page, 'evidence': quote}
            for page in pages]


def hit(chunk, gold):
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text'] for g in gold)


def question_hit(chunks, question):
    if not question.get('answerable', True):
        return False
    evidence = required_evidence(question)
    if question.get('required_evidence'):
        return all(any(hit(chunk, [item]) for chunk in chunks) for item in evidence)
    return any(hit(chunk, [item]) for chunk in chunks for item in evidence)


class TfidfIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts):
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {term: value / norm for term, value in vector.items()}

    def search(self, question, top_k=3, filters=None):
        query = self._vector(Counter(tokenize(question)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if filters and any(chunk['metadata'].get(key) != value for key, value in filters.items()):
                continue
            score = sum(value * vector.get(term, 0) for term, value in query.items())
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': 'tfidf'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class BM25Index:
    """BM25 lexical retrieval with the same Unicode-aware tokenizer as TF-IDF."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('BM25 k1 phải > 0 và b phải thuộc [0, 1]')
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.counts = [Counter(tokenize(chunk['text'])) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        self.document_frequency = Counter(term for counts in self.counts for term in counts)

    def search(self, question, top_k=3, filters=None):
        query_terms = set(tokenize(question))
        results = []
        total_docs = len(self.chunks)
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths):
            if filters and any(chunk['metadata'].get(key) != value for key, value in filters.items()):
                continue
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.average_length or 1)
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': 'bm25'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class RRFIndex:
    """Fuse independent ranked retrievers using reciprocal rank fusion."""

    def __init__(self, *indexes, rank_constant=60, candidate_k=12):
        if len(indexes) < 2 or rank_constant <= 0 or candidate_k <= 0:
            raise ValueError('RRF cần ít nhất hai index và các tham số dương')
        self.indexes = indexes
        self.rank_constant = rank_constant
        self.candidate_k = candidate_k

    def search(self, question, top_k=3, filters=None):
        fused = {}
        for index in self.indexes:
            candidates = index.search(question, top_k=self.candidate_k, filters=filters)
            for rank, candidate in enumerate(candidates, 1):
                chunk_id = candidate['id']
                if chunk_id not in fused:
                    fused[chunk_id] = {**candidate, 'score': 0.0, 'strategy': 'rrf'}
                fused[chunk_id]['score'] += 1 / (self.rank_constant + rank)
        return sorted(fused.values(), key=lambda c: (-c['score'], c['id']))[:top_k]


def pack_context(results, budget=800):
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']}] [{c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def evaluate(index, questions, budget=800, top_k=3):
    if top_k < 1:
        raise ValueError('top_k phải dương')
    traces = []
    for q in questions:
        start = time.perf_counter()
        results = index.search(q['question'], top_k=max(top_k, 3), filters=q.get('filter'))
        selected_results = results[:top_k]
        context, ids = pack_context(selected_results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': question_hit([c], q)} for c in results]
        answerable = q.get('answerable', True)
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'],
                       'strategy': results[0].get('strategy', 'none') if results else 'none',
                       'top_k': top_k, 'budget_chars': budget, 'answerable': answerable,
                       'results': marked, 'context': context, 'context_ids': ids,
                       'context_chars': len(context), 'elapsed_ms': elapsed_ms,
                       'hit1': question_hit(results[:1], q) if answerable else None,
                       'hit3': question_hit(results[:3], q) if answerable else None,
                       'hitk': question_hit(selected_results, q) if answerable else None,
                       'context_hit': question_hit([c for c in selected_results if c['id'] in ids], q) if answerable else None})
    return traces


def main():
    parser = argparse.ArgumentParser(description='Chạy baseline TF-IDF trên một split rõ ràng')
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--output', required=True, type=Path, help='Thư mục output; file cùng tên sẽ bị ghi đè')
    args = parser.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    # Do not inspect held-out test evidence during dev work.
    selected = [q for q in questions if q.get('split') == args.split]
    validate(chunks, selected)
    started = time.perf_counter()
    index = TfidfIndex(chunks)
    index_seconds = time.perf_counter() - started
    traces = evaluate(index, selected)
    answerable_traces = [trace for trace in traces if trace['answerable']]
    summary = {'strategy': 'tfidf', 'split': args.split, 'n': len(traces),
               'answerable_n': len(answerable_traces),
               'hit1': sum(t['hit1'] is True for t in answerable_traces),
               'hit3': sum(t['hit3'] is True for t in answerable_traces),
               'hitk': sum(t['hitk'] is True for t in answerable_traces),
               'context_hit': sum(t['context_hit'] is True for t in answerable_traces),
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'index_seconds': index_seconds, 'service_cost_usd': 0}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('summary.json', summary)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
