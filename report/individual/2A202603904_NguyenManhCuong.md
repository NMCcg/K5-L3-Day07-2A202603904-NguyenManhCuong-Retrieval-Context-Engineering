# Báo cáo Day 7 — Agentic RAG (trạng thái đã xác minh)

- **Họ và tên:** Nguyễn Mạnh Cường
- **MSSV:** 2A202603904
- **Lớp:** K5-L3B

## Benchmark

`data/benchmark.json` hiện có 35 câu trên 219 chunks:

- Dev: 28 câu, gồm 24 câu answerable và 4 câu unanswerable.
- Holdout test mới: Q29–Q35, gồm 5 câu answerable và 2 câu unanswerable.
- Gold evidence của tất cả câu đã được kiểm tra với chunks; mọi câu đều qua validation.

Test Q29–Q35 được tách sau khi bộ test cũ đã được dùng trong kiểm tra. Vì vậy, các trace Q16–Q28 trong `results/test_traces.json` là lịch sử, không đại diện cho holdout mới. Bản báo cáo cũ được giữ riêng tại [báo cáo lịch sử](./2A202603904_NguyenManhCuong_historical.md).

## Retrieval offline

Đã chạy TF-IDF, BM25 và hybrid lexical RRF (TF-IDF + BM25) trên dev, rồi chạy baseline và cấu hình BM25 đã chọn trên holdout. Metric tính trên câu answerable; Evidence Hit@3 luôn là cutoff 3, còn Hit@k theo top-k cấu hình.

| Phương pháp/cấu hình | Split | Hit@1 | Evidence Hit@3 | Hit@k | Evidence trong context |
|---|---|---:|---:|---:|---:|
| TF-IDF, top-3, budget 800 | dev (24) | 5/24 (20.8%) | 7/24 (29.2%) | 7/24 (29.2%) | 5/24 (20.8%) |
| BM25, top-3, budget 800 | dev (24) | 6/24 (25.0%) | 8/24 (33.3%) | 8/24 (33.3%) | 6/24 (25.0%) |
| BM25, top-3, budget 1800 | dev (24) | 6/24 (25.0%) | 8/24 (33.3%) | 8/24 (33.3%) | 8/24 (33.3%) |
| BM25, top-5, budget 2400 | dev (24) | 6/24 (25.0%) | 8/24 (33.3%) | 11/24 (45.8%) | 11/24 (45.8%) |
| Hybrid lexical RRF, top-3, budget 800 | dev (24) | 5/24 (20.8%) | 8/24 (33.3%) | 8/24 (33.3%) | 5/24 (20.8%) |
| TF-IDF, top-3, budget 800 | holdout (5) | 5/5 | 5/5 | 5/5 | 5/5 |
| BM25, top-5, budget 2400 (đã chọn trên dev) | holdout (5) | 5/5 | 5/5 | 5/5 | 5/5 |

BM25 top-5/budget 2400 được chọn vì có coverage Hit@k và context cao nhất trên dev trong các cấu hình đã thử. Trên holdout, hai cấu hình retrieval đều tìm được evidence cho cả 5 câu answerable; đây chỉ là kết quả retrieval, không phải đánh giá câu trả lời của agent. Hai câu unanswerable được loại khỏi retrieval coverage.

Các trace và số liệu offline:

- Dev: `starter/results/day07-offline-dev/`
- Retrieval holdout: `starter/results/day07-offline-holdout-retrieval/`
- Bộ câu hỏi để chấm agent trên holdout mới: `results/holdout_test_cases.json`

## Agent, dense retrieval và giới hạn xác minh

Chưa xác minh được lượt chạy Day 7 end-to-end với dense pgvector và model thật. Kiểm tra môi trường trước đó cho thấy PostgreSQL Day 6 (`127.0.0.1:15432`) và Ollama (`127.0.0.1:11434`) không hoạt động; `DATABASE_URL` và `OLLAMA_MODEL` chưa được cấu hình. Do đó:

- Không có kết quả dense pgvector hoặc hybrid dense được báo cáo.
- Chưa có trace agent thật trên dev hay holdout Q29–Q35, và chưa có human review cho holdout mới.
- `results/test_summary.json` đã được gắn nhãn lịch sử; không dùng làm điểm test hiện tại.
- Không thể kết luận checklist hoàn tất cho đến khi chạy được agent/model thật, đánh giá citation và lưu summary mới.

## Kiểm thử code

Đã chạy bộ unit tests: **21 passed, 0 failures, 0 errors**. Bộ test bao gồm retrieval, multi-hop evidence, citation/context của agent, xử lý lỗi, Day 6 adapter và giới hạn Ollama local.

## Bước còn lại để hoàn tất checkpoint

1. Khởi động PostgreSQL/pgvector Day 6 có strategy B và Ollama local; đặt `DATABASE_URL` và `OLLAMA_MODEL`.
2. Chạy retrieval benchmark trên dev với các phương pháp TF-IDF, BM25, dense và hybrid RRF; chỉ chọn cấu hình sau khi so sánh đầy đủ.
3. Chạy agent trên dev để chốt hành vi. Sau đó chạy một lần trên Q29–Q35:
   ```bash
   python starter/run_agent.py --split test --output results/agent-holdout-test.json --top-k 5 --budget 2400
   ```
4. Đối chiếu từng câu trả lời/citation với context; lưu nhãn người chấm và tạo summary mới cho holdout. Không tái sử dụng nhãn hoặc summary của split test cũ.
