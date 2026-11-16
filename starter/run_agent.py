"""Run the local-model agent against the Day 6 pgvector index."""
import argparse
import json
import os
import statistics
from pathlib import Path
import time

from agentic import run_agent
from day6_retriever import Day6PgVectorRetriever
from ollama_controller import OllamaController


def main():
    parser = argparse.ArgumentParser(description='Chạy agent với Ollama local và pgvector Day 6')
    parser.add_argument('--questions', type=Path, default=Path('data/benchmark.json'))
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', default='B')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--budget', type=int, default=1800, help='Context budget in characters')
    args = parser.parse_args()
    if args.top_k < 1 or args.max_searches < 1 or args.budget < 1:
        parser.error('top-k, max-searches và budget phải dương')

    payload = json.loads(args.questions.read_text(encoding='utf-8'))
    questions = payload.get('questions') if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError('questions phải là JSON array hoặc object có trường questions là array')
    selected = [question for question in questions if question.get('split') == args.split]
    if not selected:
        raise ValueError(f'Không có câu hỏi thuộc split {args.split}')
    controller = OllamaController()
    retriever = Day6PgVectorRetriever(
        database_url=os.getenv('DATABASE_URL'),
        strategy=args.strategy,
    )
    traces = []
    run_started = time.perf_counter()
    try:
        for question in selected:
            question_started = time.perf_counter()
            try:
                result = run_agent(
                    question['question'],
                    controller,
                    retriever,
                    filters=question.get('filter'),
                    max_searches=args.max_searches,
                    top_k=args.top_k,
                    budget=args.budget,
                )
                traces.append({'query_id': question['id'], **result})
            except Exception as exc:
                traces.append({
                    'query_id': question['id'],
                    'status': 'error',
                    'error': {'type': type(exc).__name__, 'message': str(exc)},
                    'events': [],
                    'citations': [],
                    'elapsed_ms': (time.perf_counter() - question_started) * 1000,
                })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        successful = [trace for trace in traces if trace.get('status') != 'error']
        summary = {
            'split': args.split,
            'model': controller.model,
            'strategy': args.strategy,
            'n': len(traces),
            'errors': sum(trace.get('status') == 'error' for trace in traces),
            'answered': sum(trace.get('status') == 'answered' for trace in traces),
            'abstained': sum(trace.get('status') == 'abstain' for trace in traces),
            'search_count_distribution': {
                str(count): sum(trace.get('search_count') == count for trace in successful)
                for count in sorted({trace.get('search_count', 0) for trace in successful})
            },
            'mean_elapsed_ms': round(statistics.mean(
                trace['elapsed_ms'] for trace in traces if 'elapsed_ms' in trace
            ), 2) if any('elapsed_ms' in trace for trace in traces) else None,
            'total_elapsed_ms': round((time.perf_counter() - run_started) * 1000, 2),
            'service_cost_usd': 0,
            'cost_note': 'Local Ollama and local pgvector; no per-request API charge.',
            'output': str(args.output),
        }
        summary_path = args.output.with_name(args.output.stem + '-summary.json')
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        retriever.close()


if __name__ == '__main__':
    main()
