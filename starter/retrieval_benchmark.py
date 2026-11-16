"""Compare lexical, dense Day 6, and hybrid retrieval on one benchmark split."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from baseline import BM25Index, RRFIndex, TfidfIndex, evaluate, validate
from day6_retriever import Day6PgVectorRetriever


def summarize(name, split, traces):
    answerable = [trace for trace in traces if trace['answerable']]
    return {
        'strategy': name,
        'split': split,
        'n': len(traces),
        'answerable_n': len(answerable),
        'hit1': sum(trace['hit1'] is True for trace in answerable),
        'hit3': sum(trace['hit3'] is True for trace in answerable),
        'hitk': sum(trace['hitk'] is True for trace in answerable),
        'context_hit': sum(trace['context_hit'] is True for trace in answerable),
        'avg_context_chars': statistics.mean(trace['context_chars'] for trace in traces),
        'median_ms': statistics.median(trace['elapsed_ms'] for trace in traces),
        'service_cost_usd': 0,
    }


def main():
    parser = argparse.ArgumentParser(description='So sánh retrieval Day 6 và các baseline lexical')
    parser.add_argument('--questions', type=Path, default=Path('data/benchmark.json'))
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', default='B', help='Day 6 chunking strategy (default: B)')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--candidate-k', type=int, default=12)
    parser.add_argument('--budget', type=int, default=800, help='Context budget in characters')
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k or args.budget < 1:
        parser.error('top-k/budget phải dương và candidate-k >= top-k')

    payload = json.loads(args.questions.read_text(encoding='utf-8'))
    questions = payload.get('questions') if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError('questions phải là JSON array hoặc object có trường questions là array')
    selected = [question for question in questions if question.get('split') == args.split]
    if not selected:
        raise ValueError(f'Không có câu hỏi thuộc split {args.split}')

    retriever = Day6PgVectorRetriever(
        database_url=os.getenv('DATABASE_URL'),
        strategy=args.strategy,
    )
    try:
        chunks = retriever.load_chunks()
        if not chunks:
            raise RuntimeError(f'Không có chunk cho strategy Day 6 {args.strategy!r}')
        validate(chunks, selected)
        indexes = {
            'tfidf': TfidfIndex(chunks),
            'bm25': BM25Index(chunks),
            'dense_pgvector': retriever,
            'hybrid_rrf': RRFIndex(BM25Index(chunks), retriever, candidate_k=args.candidate_k),
        }

        args.output.mkdir(parents=True, exist_ok=True)
        summaries = []
        for name, index in indexes.items():
            started = time.perf_counter()
            traces = evaluate(index, selected, budget=args.budget, top_k=args.top_k)
            total_ms = (time.perf_counter() - started) * 1000
            summary = summarize(name, args.split, traces)
            summary.update({'top_k': args.top_k, 'candidate_k': args.candidate_k,
                            'budget_chars': args.budget, 'total_ms': total_ms})
            (args.output / f'{name}-traces.json').write_text(
                json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
            )
            (args.output / f'{name}-summary.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
            )
            summaries.append(summary)
        (args.output / 'comparison.json').write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    finally:
        retriever.close()


if __name__ == '__main__':
    main()
