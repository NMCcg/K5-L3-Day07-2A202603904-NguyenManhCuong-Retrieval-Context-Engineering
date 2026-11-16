"""Behavior checks for the runnable Day 7 baseline and Day 6 JSON adapter."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from baseline import BM25Index, RRFIndex, TfidfIndex, evaluate, hit, pack_context, validate
from convert_day6 import convert_chunk, convert_question


def chunk(id, text, page=1):
    return {'id': id, 'source_id': 'doc', 'source': 'doc.pdf', 'page': page,
            'text': text, 'metadata': {'year': 2025}}


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk('a', 'Hoàn tiền mất bảy ngày.'), chunk('b', 'Giao hàng mất ba ngày.', 2)]
        self.q = {'id': 'q', 'split': 'dev', 'question': 'Hoàn tiền bao lâu?',
                  'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'bảy ngày'}], 'expected_answer': 'bảy ngày'}

    def test_rank_and_evidence(self):
        validate(self.chunks, [self.q])
        trace = evaluate(TfidfIndex(self.chunks), [self.q])[0]
        self.assertEqual(trace['results'][0]['id'], 'a')
        self.assertTrue(trace['hit1'] and trace['hit3'] and trace['context_hit'])
        self.assertFalse(hit(self.chunks[1], self.q['gold']))

    def test_gold_alternative_and_wrong_page(self):
        valid = {'source_id': 'doc', 'page': 1, 'evidence': 'bảy ngày'}
        wrong = {'source_id': 'doc', 'page': 2, 'evidence': 'bảy ngày'}
        self.assertFalse(hit(self.chunks[0], [wrong]))
        self.assertTrue(hit(self.chunks[0], [wrong, valid]))
        with self.assertRaises(ValueError):
            validate(self.chunks, [{**self.q, 'gold': [wrong]}])

    def test_budget_dedup_and_empty_query(self):
        context, ids = pack_context([self.chunks[0], self.chunks[0]], 800)
        self.assertEqual(ids, ['a'])
        self.assertLessEqual(len(context), 800)
        self.assertEqual(pack_context(self.chunks, 1), ('', []))
        self.assertEqual(TfidfIndex(self.chunks).search('khôngkhớp'), [])
        trace = evaluate(TfidfIndex(self.chunks), [self.q], budget=1)[0]
        self.assertTrue(trace['hit3'])
        self.assertFalse(trace['context_hit'])

    def test_bm25_filter_and_rrf_return_ranked_candidates(self):
        chunks = [
            chunk('a', 'hoàn tiền hoàn tiền trong bảy ngày'),
            chunk('b', 'giao hàng hoàn tiền trong bảy ngày', page=2),
        ]
        chunks[0]['metadata']['chapter'] = 'policy'
        chunks[1]['metadata']['chapter'] = 'shipping'
        bm25 = BM25Index(chunks)
        filtered = bm25.search('hoàn tiền', top_k=2, filters={'chapter': 'shipping'})
        self.assertEqual([row['id'] for row in filtered], ['b'])
        fused = RRFIndex(TfidfIndex(chunks), bm25).search('hoàn tiền', top_k=2)
        self.assertEqual(len(fused), 2)
        self.assertTrue(all(row['strategy'] == 'rrf' for row in fused))
        with self.assertRaises(ValueError):
            BM25Index(chunks, b=1.1)

    def test_multihop_requires_all_evidence_and_unanswerable_is_excluded(self):
        chunks = [
            chunk('topology', 'Topology có router và kế hoạch IP/SID.', page=44),
            chunk('recovery', 'Sau link failure, lưu lượng chuyển sang backup path.', page=62),
        ]
        multi_hop = {
            'id': 'multi', 'split': 'dev', 'question': 'Topology IP SID và backup path khi link failure.',
            'expected_answer': 'Topology được chuẩn bị và lưu lượng đổi đường khi lỗi.',
            'required_evidence': [
                {'source_id': 'doc', 'page': 44, 'evidence': 'kế hoạch IP/SID'},
                {'source_id': 'doc', 'page': 62, 'evidence': 'backup path'},
            ],
        }
        missing = {
            'id': 'missing', 'split': 'dev', 'question': 'Giá giấy phép là bao nhiêu?',
            'answerable': False, 'absence_reason': 'Không có giá trong tài liệu.',
        }
        validate(chunks, [multi_hop, missing])
        traces = evaluate(TfidfIndex(chunks), [multi_hop, missing], top_k=2)
        self.assertFalse(traces[0]['hit1'])
        self.assertTrue(traces[0]['hit3'])
        self.assertTrue(traces[0]['context_hit'])
        self.assertIsNone(traces[1]['hit3'])
        with self.assertRaises(ValueError):
            validate(chunks[:1], [multi_hop])

    def test_top_five_keeps_hit3_separate_from_hitk(self):
        rows = [chunk(str(i), f'unrelated chunk {i}') for i in range(4)]
        rows.append(chunk('gold', 'Đáp án nằm ở vị trí thứ năm.', page=5))
        question = {
            'id': 'rank5', 'split': 'dev', 'question': 'Câu hỏi về đáp án',
            'expected_answer': 'Đáp án', 'gold': [
                {'source_id': 'doc', 'page': 5, 'evidence': 'Đáp án nằm ở vị trí thứ năm.'}
            ],
        }

        class FixedIndex:
            def search(self, query, *, top_k, filters=None):
                return rows[:top_k]

        trace = evaluate(FixedIndex(), [question], top_k=5)[0]
        self.assertFalse(trace['hit3'])
        self.assertTrue(trace['hitk'])
        self.assertTrue(trace['context_hit'])

    def test_convert_day6_and_preserve_filter(self):
        raw_chunk = {'chunk_id': 'c1', 'document_id': 'doc', 'source': 'doc.pdf', 'page': 1,
                     'text': 'bảy ngày', 'metadata': {'year': 2025}}
        raw_q = {'id': 'q', 'split': 'dev', 'question': 'bao lâu', 'gold_source_id': 'doc',
                 'gold_page': 1, 'evidence_quote': 'bảy ngày', 'expected_answer': 'bảy ngày', 'filter': {'year': 2025}}
        c, q = convert_chunk(raw_chunk), convert_question(raw_q)
        validate([c], [q])
        self.assertEqual(q['filter'], {'year': 2025})
        self.assertEqual(c['id'], 'c1')

    def test_cli_uses_requested_split_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'chunks.json').write_text(json.dumps(self.chunks), encoding='utf-8')
            # A test gold which fails validation must not be inspected by a dev run.
            test = {**self.q, 'id': 't', 'split': 'test', 'gold': [{'source_id': 'x', 'page': 9, 'evidence': 'missing'}]}
            (root/'q.json').write_text(json.dumps([self.q, test]), encoding='utf-8')
            subprocess.run([sys.executable, str(ROOT/'starter/baseline.py'), '--chunks', str(root/'chunks.json'),
                            '--questions', str(root/'q.json'), '--split', 'dev', '--output', str(root/'out')],
                           check=True, capture_output=True, text=True)
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertEqual((summary['split'], summary['n'], summary['hit1']), ('dev', 1, 1))
            self.assertEqual(len(json.loads((root/'out/traces.json').read_text(encoding='utf-8'))), 1)


if __name__ == '__main__': unittest.main()
