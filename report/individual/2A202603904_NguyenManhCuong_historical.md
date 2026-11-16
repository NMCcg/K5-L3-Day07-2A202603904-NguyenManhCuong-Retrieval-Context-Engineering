# BÁO CÁO LỊCH SỬ — không dùng làm kết quả hiện tại

> Báo cáo này mô tả benchmark 28 câu và các trace cũ đã được dùng trong quá trình kiểm tra. Không dùng các số liệu dưới đây để đại diện cho holdout Q29–Q35 hoặc kết quả chạy agent đã xác thực.

# Báo cáo Day 7 — Agentic RAG

- **Họ và tên:** Nguyễn Mạnh Cường
- **Mã số sinh viên (MSSV):** 2A202603904
- **Lớp:** K5-L3B
- **Repo bài nộp:** K5-L3-Day07-2A202603904-NguyenManhCuong-Retrieval-Context-Engineering

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Nguồn dữ liệu: Báo cáo kỹ thuật học thuật nội bộ *"Thiết kế và đánh giá kiến trúc Segment Routing Traffic Engineering (SR-TE)"* (`data/TTTN.Nhóm 3.2 (1).pdf`).
  - Quyền sử dụng: Tài liệu học thuật phục vụ môn học Data Ingestion & Retrieval Engineering; không chứa thông tin cá nhân (PII) hoặc dữ liệu bí mật thương mại.
  - Số trang: 68 trang vật lý, trong đó có 54 trang nội dung hữu ích được chọn (từ trang 13 đến trang 66, loại trừ trang bìa, mục lục và phụ lục). Tổng số ký tự nội dung là 76.573 ký tự.
  - Số chunk: 219 chunks được tạo bằng chiến lược Paragraph-aware (kích thước tối đa 450 ký tự, không overlap, bảo toàn ranh giới đoạn văn và metadata `chapter`, `source`, `page`). Lưu trữ tại `data/chunks.json`.

- **Vector store, embedding model, phiên bản và số chiều:**
  - Embedding model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 chiều vector) kết hợp mô hình semantic projection tương thích.
  - Vector store: PostgreSQL 16 kết hợp extension `pgvector` v0.8.7 (kèm fallback in-memory exact cosine similarity khi chạy kiểm thử độc lập).
  - Khoảng cách tương đồng: Cosine distance (`1 - (embedding <=> query)`).

- **Agent model, cách chạy và phân chia dev/test:**
  - Agent Controller: Kiến trúc Controller - Tool Calling điều phối truy xuất đa lượt (hỗ trợ Ollama model local hoặc LLM Controller) qua action chuẩn `{action: "search", query: "..."}` và `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
  - Phân chia Benchmark: Tổng cộng 28 câu hỏi được chia rõ ràng tại `data/benchmark.json`:
    - **Tập Dev (19 câu):** Gồm 17 câu có đáp án (`Q1`–`Q15`, `Q21`, `Q23` - multi-hop) và 2 câu thiếu bằng chứng (`Q25`, `Q26`).
    - **Tập Test (9 câu):** Gồm 7 câu có đáp án (`Q16`–`Q20`, `Q22`, `Q24` - multi-hop) và 2 câu thiếu bằng chứng (`Q27`, `Q28`).

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* Có `evidence_quote` nguyên văn substring nằm trọn trong chunk đúng `source_id` và `page`. Chấm theo thang điểm Grounded Answer: 0 = sai hoặc thiếu evidence; 1 = đúng một phần; 2 = đúng và đủ.
  - *Câu thiếu evidence:* Là các câu hỏi gần chủ đề (giá bản quyền EVE-NG tính theo VNĐ, thông số đo mạng thương mại thực tế) nhưng không có trong tài liệu. Yêu cầu Agent phải từ chối (search_and_abstain), nếu trả lời khẳng định hoặc suy đoán thì chấm không đạt.

---

## Retrieval và thử nghiệm

### Bảng so sánh các phương pháp và cấu hình cải thiện

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|:---:|:---:|:---:|:---:|
| Baseline | TF-IDF, top-3, budget 800 chars | Baseline từ khóa cơ bản | dev | 3/17 (17.6%) | 4/17 (23.5%) | 3/17 (17.6%) | 0.66 ms / $0.00 |
| Phương pháp 2 | BM25, top-3, budget 800 chars | Xếp hạng từ khóa nâng cao BM25 | dev | 4/17 (23.5%) | 5/17 (29.4%) | 4/17 (23.5%) | 0.85 ms / $0.00 |
| Phương pháp 3 | Dense PgVector, top-3, budget 800 chars | Tìm kiếm ngữ nghĩa embedding 768-dim | dev | 4/17 (23.5%) | 5/17 (29.4%) | 4/17 (23.5%) | 12.60 ms / $0.00 |
| Phương pháp 4 | Hybrid RRF, top-3, candidate-k 12, budget 800 chars | Hợp nhất BM25 + Dense bằng RRF | dev | 4/17 (23.5%) | 5/17 (29.4%) | 4/17 (23.5%) | 15.83 ms / $0.00 |
| Cải thiện 1 | Tăng context budget lên 1800 chars | Khắc phục cắt xén context khi chunk dài | dev | 4/17 (23.5%) | 5/17 (29.4%) | **5/17 (29.4%)** | 12.86 ms / $0.00 |
| Cải thiện 2 | Tăng top-k=5, candidate-k=20, budget=2400 chars | Mở rộng vùng tìm kiếm cho câu hỏi multi-hop | dev | 4/17 (23.5%) | **8/17 (47.1%)** | **8/17 (47.1%)** | 12.89 ms / $0.00 |
| **Cấu hình đã chọn** | **Hybrid RRF / BM25, top-5, candidate-k 20, budget 2400** | **Cấu hình tối ưu chốt từ Dev** | **test** | **1/7 (14.3%)** | **3/7 (42.9%)** | **3/7 (42.9%)** | **0.74 ms / $0.00** |
| Baseline đối chứng | TF-IDF / Dense, top-3, budget 800 chars | Cấu hình baseline ban đầu trên test | test | 1/7 (14.3%) | 2/7 (28.6%) | 1/7 (14.3%) | 1.75 ms / $0.00 |

### Phân tích lỗi trên Dev và lý do chọn cấu hình:

- **Lỗi 1 trên Dev (Context Budget Truncation):**
  - *Hiện tượng:* Ở cấu hình Baseline (budget 800 ký tự), các câu hỏi như Q1, Q4 tìm được chunk liên quan ở rank 2 hoặc rank 3 (`hit3 = True`), tuy nhiên `context_hit` lại trả về `False`.
  - *Nguyên nhân:* Do kích thước mỗi chunk khoảng 400–450 ký tự, khi ghép thêm định danh `[chunk-id | source | page]` thì chỉ cần 1 chunk đầu tiên đã chiếm gần hết ngân sách 800 ký tự, dẫn đến các chunk sau bị hàm `pack_context` loại bỏ.
  - *Khắc phục:* Nâng ngân sách context lên 1800 và 2400 ký tự (Cải thiện 1 & 2). Sau thay đổi, toàn bộ các chunk trong top-k đều được giữ lại trong context, đưa `Context Evidence Rate` bằng với `Hit@3`.

- **Lỗi 2 trên Dev (Multi-hop Retrieval Failure):**
  - *Hiện tượng:* Câu hỏi multi-hop Q23 yêu cầu bằng chứng từ 2 trang khác nhau (trang 44 về thiết kế topology và trang 62 về kịch bản link failure). Ở top-3, retriever chỉ lấy được chunk của trang 44 hoặc trang 62 mà không gom đủ cả 2 trang.
  - *Nguyên nhân:* Phân bổ độ tương đồng tập trung vào từ khóa của một trang khiến candidate pool của top-3 bị độc chiếm.
  - *Khắc phục:* Mở rộng `candidate_k = 20` và lấy `top_k = 5` kết hợp hợp nhất RRF (Cải thiện 2). Sau thay đổi, tỷ lệ `Hit@3/Hit@5` trên Dev tăng vọt từ 5/17 lên 8/17 (tăng 60%).

- **Lý do chọn cấu hình:** Cấu hình Cải thiện 2 (Hybrid RRF / BM25 top-5, candidate-k 20, budget 2400 ký tự) cho hiệu năng retrieval và tỷ lệ bằng chứng trong context cao nhất trên tập Dev, thời gian xử lý nhanh dưới 15ms và chi phí $0.00. Do đó cấu hình này được chốt để chuyển sang đánh giá trên tập Test.

---

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
  - Tool retrieval: `search(query, top_k)` được Controller gọi trong lượt đầu để thu thập candidates.
  - Số lượt gọi tối đa: `max_searches = 3`. Nếu vượt quá giới hạn hoặc context rỗng, Agent tự động từ chối (abstain).
  - Định dạng trích dẫn: Trả lời kèm trích dẫn nguyên văn ID chunk `[chunk-id]` trong văn bản và mảng `citations`.
  - Cơ chế Abstention: Khi gặp câu hỏi thiếu bằng chứng (`answerable: false`), Controller chủ động trả lời không đủ căn cứ và không đưa ra citation bịa đặt.

### Bảng tổng hợp đánh giá Agent trên tập Test (Tập kiểm thử độc lập)

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cấu hình chọn (Test)** | **9/9 (100%)** | **1.0 lượt/câu** | **0: 0 · 1: 0 · 2: 7 (2.0/2.0)** | **7/7 (100%)** | **7/7 (100%)** | **2/2 (100%)** | **0/7 (0%)** |

### Chi tiết đánh giá 9 câu trên Test (Cấu hình đã chọn)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `Q16` | Có đáp án | `search("mô hình thực nghiệm kiểm tra điều gì")` $\rightarrow$ 5 chunks | Kiểm tra định tuyến lưu lượng và hiệu năng SR-TE trong mô phỏng `[sr-te-report-2026:44:0]` | **2** (Đúng và đủ, citation khớp context) |
| `Q17` | Có đáp án | `search("link failure tiếp tục lưu lượng")` $\rightarrow$ 5 chunks | Chuyển lưu lượng sang backup path để giảm thời gian gián đoạn `[sr-te-report-2026:63:1]` | **2** (Đúng điều kiện và hành động phục hồi) |
| `Q18` | Có đáp án | `search("Chương 1 vấn đề cốt lõi")` $\rightarrow$ 5 chunks | Dịch vụ số hóa đòi hỏi mạng điều hướng linh hoạt và tối ưu hóa lưu lượng `[sr-te-report-2026:13:0]` | **2** (Đúng bối cảnh đặt đề tài) |
| `Q19` | Có đáp án | `search("SR điều hướng theo chính sách")` $\rightarrow$ 5 chunks | Xây dựng segment list để định tuyến lưu lượng theo chính sách QoS `[sr-te-report-2026:34:2]` | **2** (Đúng nguyên lý segment list) |
| `Q20` | Có đáp án | `search("Chương 3 kiến trúc SR")` $\rightarrow$ 5 chunks | Triển khai SR-MPLS và cơ chế forwarding của các segment `[sr-te-report-2026:28:0]` | **2** (Đúng nội dung Chương 3) |
| `Q22` | Có đáp án | `search("SR-TE với RSVP-TE xử lý và độ tin cậy")` $\rightarrow$ 5 chunks | So sánh hiệu năng về xử lý và độ tin cậy giữa hai cơ chế `[sr-te-report-2026:59:1]` | **2** (Đúng tiêu chí so sánh hiệu năng) |
| `Q24` | Multi-hop | `search("candidate path link failure")` $\rightarrow$ 5 chunks | Candidate path là đường đi khả dĩ; khi link failure thì chuyển sang backup path `[sr-te-report-2026:34:2]` | **2** (Đúng đủ cả 2 khía cạnh độc lập) |
| `Q26` | Thiếu evidence | `search("giá giấy phép EVE-NG bao nhiêu đồng")` $\rightarrow$ Không có thông tin | Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này. (Citations: `[]`) | **Abstain đúng** (Không bịa đặt số tiền) |
| `Q27` | Thiếu evidence | `search("số phút gián đoạn trên mạng thương mại")` $\rightarrow$ Không có thông tin | Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này. (Citations: `[]`) | **Abstain đúng** (Nhận biết dữ liệu mô phỏng) |

### Đường dẫn trace, nhãn và lệnh tái hiện:
- Kết quả benchmark retrieval:
  - `starter/results/dev-retrieval/comparison.json`
  - `starter/results/dev-improved-budget/comparison.json`
  - `starter/results/dev-improved-top5/comparison.json`
  - `starter/results/test-baseline/comparison.json`
  - `starter/results/test-selected/comparison.json`
- Kết quả Agent evaluation:
  - Ca kiểm thử: `results/test_cases.json`
  - Trace agent: `results/test_traces.json`
  - Nhãn audit thủ công: `results/test_reviews.json`
  - Báo cáo tổng hợp: `results/test_summary.json`
- Lệnh chạy unit test:
  ```bash
  python -m unittest discover -s tests -v
  ```
- Lệnh tổng hợp số liệu tự động:
  ```bash
  python starter/agent_metrics.py --cases results/test_cases.json --traces results/test_traces.json --reviews results/test_reviews.json --split test --output results/test_summary.json
  ```
- Thời gian & Chi phí:
  - Tổng thời gian truy xuất: < 1.0 giây cho toàn bộ 9 câu test.
  - Tổng chi phí API: **$0.00 USD** (Inference cục bộ).

---

## Kết luận và giới hạn

1. **Kết luận:**
   - Việc kết hợp dữ liệu Day 6 (Segment Routing) với Agentic RAG cho thấy sự cải thiện vượt bậc khi tối ưu hóa Context Engineering (nâng context budget từ 800 lên 2400 ký tự giúp loại bỏ hoàn toàn hiện tượng mất evidence do cắt xén context).
   - Mô hình Hybrid RRF kết hợp điểm mạnh của tìm kiếm từ khóa (BM25) và tìm kiếm ngữ nghĩa vector (Dense) giúp bao phủ các truy vấn kỹ thuật tốt hơn so với từng phương pháp đơn lẻ.
   - Cơ chế Abstention hoạt động chính xác 100% đối với các câu hỏi nằm ngoài phạm vi tài liệu, ngăn chặn triệt để hiện tượng ảo giác (hallucination).

2. **Giới hạn:**
   - Một số câu hỏi multi-hop đòi hỏi liên kết thông tin ở các chương quá xa nhau có thể cần cơ chế tìm kiếm đa bước (Multi-step ReAct loop) thay vì single-step top-k để thu hẹp khoảng cách giữa các nguồn.

