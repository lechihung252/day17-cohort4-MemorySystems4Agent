# Báo cáo Day 17: Memory Systems for AI Agent

Báo cáo so sánh hai agent chạy trên cùng bộ dữ liệu tiếng Việt:

- **Baseline**: chỉ nhớ trong cùng một thread.
- **Advanced**: có short-term memory, `User.md` bền vững và compact memory.

Mọi số liệu bên dưới lấy từ chế độ **offline**, nên chạy lại sẽ ra đúng kết quả này. Ngưỡng compact là 500 token, sau khi compact giữ lại 4 message, token được ước lượng bằng `ceil(len/4)`.

```bash
python src/benchmark.py          # offline, kết quả lặp lại được
python src/benchmark.py --live   # gọi LLM thật theo .env
pytest src/test_agents.py -v     # 38 test
```

## 1. Kết quả benchmark

**Standard Benchmark**: 10 hội thoại, 101 lượt, 14 câu hỏi recall.

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 1875 | 9989 | 0.00 | 0.15 | 0 | 0 |
| Advanced | 1996 | **15960 (+60%)** | **1.00** | 1.00 | 256 | 0 |

**Long-Context Stress Benchmark**: 1 hội thoại, 16 lượt dài (~2.300 token phía người dùng), 3 câu hỏi recall.

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2436 | 21127 | 0.00 | 0.15 | 0 | 0 |
| Advanced | 2447 | **6916 (−67%)** | **1.00** | 1.00 | 214 | 14 |

## 2. Ba lớp memory của Advanced

| Lớp | Ở đâu | Giữ cái gì | Sống bao lâu |
|---|---|---|---|
| Short-term | `CompactMemoryManager.messages` | N message gần nhất, nguyên văn | Trong một thread |
| Compact | `CompactMemoryManager.summary` | Message cũ, mỗi message cắt còn một dòng, tối đa 8 dòng | Trong một thread |
| Persistent | `state/profiles/<user>/User.md` | Fact ổn định: tên, nơi ở, nghề, style, sở thích… | Qua mọi thread và mọi lần khởi động lại |

Mỗi lượt, prompt context của Advanced gồm `User.md` + summary + các message gần nhất. Prompt context của Baseline là **toàn bộ lịch sử thread**.

## 3. Phân tích

### 3.1. Vì sao Advanced có recall tốt hơn Baseline?

Câu hỏi recall luôn được hỏi ở **một thread mới**.

- **Baseline** không có gì để mang sang thread mới, nên recall luôn bằng 0. Điều này đúng như thiết kế: test `test_baseline_remembers_within_same_thread` xác nhận Baseline vẫn nhớ được trong cùng thread, nên nó không bị làm yếu đi một cách giả tạo.
- **Advanced** đọc `User.md`, file này nằm trên đĩa. Một instance agent mới tạo vẫn trả lời đúng (`test_cross_session_recall`).

Recall đạt 1.00 không chỉ nhờ *lưu* fact, mà còn nhờ *lưu đúng* fact:

- **Correction:** Đà Nẵng → Huế, backend → MLOps. `upsert_fact` ghi đè giá trị cũ, nên `User.md` không bao giờ chứa hai nơi ở cùng lúc.
- **Nhiễu:** "Hà Nội chỉ là nơi đi họp", "product manager chỉ là câu đùa". Các câu này bị loại trước khi trích fact.
- **Câu hỏi trông như câu khẳng định:** *"…đồ uống yêu thích của mình là gì."* kết thúc bằng dấu chấm. Nếu không lọc, giá trị `"gì"` sẽ ghi đè `cà phê sữa đá`. Lỗi này đã thật sự xảy ra với câu *"…mình nuôi con gì."* trong lúc phát triển, và làm recall tụt xuống 32/33.

### 3.2. Vì sao Advanced tốn hơn ở hội thoại ngắn?

Ở Standard, Advanced xử lý nhiều hơn **+60% prompt token** (15960 so với 9989), và **không có lần compact nào**.

- Mỗi hội thoại chỉ khoảng 150–300 token, chưa bao giờ chạm ngưỡng 500, nên compact không có cơ hội tiết kiệm gì.
- Ngược lại, lượt nào Advanced cũng phải mang theo `User.md`, khoảng 60 token. 101 lượt × ~60 token ≈ 6.000 token, khớp gần đúng với chênh lệch 5.971 token.

Kết luận: với hội thoại ngắn, persistent memory là **chi phí cố định cho mỗi lượt** để đổi lấy khả năng nhớ qua phiên. Đây là trade-off có chủ đích, và test `test_short_thread_costs_more_for_advanced` khẳng định điều này.

### 3.3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Prompt token theo từng lượt của bộ stress:

| Lượt | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 187 | 357 | 510 | 670 | 991 | 1286 | 1526 | 1826 | 2103 | 2397 |
| Advanced | 220 | 390 | 379 | 413 | 479 | 463 | 413 | 473 | 443 | 456 |
| Số lần compact (cộng dồn) | 0 | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |

- **Hai lượt đầu** Advanced vẫn đắt hơn, vì phải mang thêm `User.md` mà chưa có compact nào.
- **Từ lượt 3** compact bắt đầu chạy, và context của Advanced **đi ngang quanh 400–480 token**. Context của Baseline tăng tuyến tính theo số lượt.
- Vì vậy **tổng** prompt tokens của Baseline tăng theo bình phương số lượt (O(n²)), còn của Advanced tăng tuyến tính (O(n)). Thread càng dài, khoảng cách càng lớn: 16 lượt đã giảm được 67%.

**Compact chủ yếu tối ưu `Prompt tokens processed`, không phải `Agent tokens only`.** Agent tokens của hai agent gần như bằng nhau (2436 so với 2447), vì đó là token người dùng gửi vào cộng token agent sinh ra, và compact không làm thay đổi chúng. Thứ compact cắt giảm là lượng ngữ cảnh *phải gửi lại* ở mỗi lượt. Trong thực tế đây là phần chiếm chủ yếu chi phí API và độ trễ.

### 3.4. Chọn ngưỡng compact

Chạy lại Advanced với các ngưỡng khác nhau, các thông số khác giữ nguyên:

| Ngưỡng (token) | Standard: prompt tokens | Standard: compactions | Stress: prompt tokens | Stress: compactions | Stress: recall |
|---|---|---|---|---|---|
| 200 | 15567 | 4 | 6916 | 28 | 1.00 |
| 300 | 15960 | 0 | 6916 | 28 | 1.00 |
| **500** | 15960 | 0 | 6916 | 14 | 1.00 |
| 800 | 15960 | 0 | 9453 | 4 | 1.00 |
| 1500 | 15960 | 0 | 13732 | 1 | 1.00 |
| 5000 | 15960 | 0 | **21922** | 0 | 1.00 |

- **Ngưỡng quá thấp (200–300):** số lần compact tăng gấp đôi nhưng prompt tokens **không giảm thêm**. Lý do: riêng 4 message được giữ lại đã vượt ngưỡng, nên lượt nào cũng phải compact (*compaction thrashing*). Với compact dùng LLM để tóm tắt, mỗi lần như vậy là thêm một lời gọi API thừa.
- **Ngưỡng quá cao (5000):** compact không bao giờ chạy. Advanced còn **đắt hơn Baseline** (21922 so với 21127) vì vẫn phải mang `User.md`.
- **500** là điểm cân bằng cho bộ dữ liệu này: hội thoại thường không bị compact, hội thoại dài được compact đủ để tiết kiệm.

Recall giữ nguyên 1.00 ở mọi ngưỡng, vì fact dài hạn nằm trong `User.md` chứ không phải trong summary. Tách *persistent* khỏi *compact* chính là để compact mạnh tay được mà không mất fact quan trọng.

### 3.5. File memory tăng trưởng ra sao và có rủi ro gì?

Kích thước `User.md` sau mỗi phiên của bộ Standard:

| Phiên | 01 | 02 | 03 | 04 | 05 | 06 | 07–10 |
|---|---|---|---|---|---|---|---|
| Bytes | 196 | 204 | 216 | 258 | 258 | 256 | 256 |
| Số fact | 6 | 6 | 6 | 8 | 8 | 8 | 8 |

File **đi ngang ở khoảng 256 byte**:

- `upsert_fact` ghi đè chứ không nối thêm.
- Nếu giá trị không đổi thì không ghi lại file.
- Phiên 06 file còn nhỏ đi 2 byte, vì "MLOps engineer" ngắn hơn "backend engineer".

Đây là kết quả của việc lưu **fact có cấu trúc** thay vì lưu log hội thoại. Dù vậy vẫn còn các rủi ro thật:

1. **Phình theo thời gian:** `style` và `interests` chỉ gộp thêm, không bao giờ bớt. Sau nhiều tháng, style có thể thành một danh sách dài và tự mâu thuẫn ("ngắn gọn" nhưng lại "giải thích chi tiết"). Mỗi byte thêm vào bị trả phí ở **mọi lượt** của **mọi thread** (xem 3.2).
2. **Lưu sai fact thì sai vĩnh viễn:** khác với short-term memory, một fact sai trong `User.md` sẽ đi theo người dùng qua mọi phiên cho tới khi bị đính chính. Lỗi `pet: gì` ở mục 3.1 là ví dụ cụ thể.
3. **Fact cũ không tự hết hạn:** "đang ở Đà Nẵng vài tháng" là fact có thời hạn, nhưng `User.md` không có thông tin thời gian.
4. **Quyền riêng tư:** `User.md` là văn bản thuần chứa thông tin cá nhân (tên, nơi ở, thú cưng). Môi trường production cần mã hóa, có quyền xóa dữ liệu, và `state/` không được commit lên git (đã có trong `.gitignore`).

## 4. Giới hạn của bài làm

- **Recall 1.00 ở chế độ offline phần nào nhờ luật trích fact được viết dựa trên chính bộ dữ liệu này.** Ví dụ: danh sách `PLACES` cố định, nghề nghiệp phải có dạng `<X> engineer`, từ khóa nhiễu như `đùa` hay `chỉ là nơi`. Gặp người dùng thật nói khác đi ("mình chuyển ra Sài Gòn rồi", "mình là data scientist"), regex sẽ bỏ sót. Chế độ live xử lý tốt hơn: LLM có tool `save_user_fact` để tự lưu fact mà regex không bắt được.
- **Summary dạng heuristic làm mất thông tin.** Sau 14 lần compact, summary không còn nhắc tới X-59, WMO hay British Columbia. Một nửa số dòng summary chỉ là `assistant: Đã ghi nhận.` Nếu người dùng hỏi về tin đầu tiên trong cùng thread, Advanced sẽ trả lời kém hơn Baseline. Cải thiện được bằng cách bỏ các câu trả lời rỗng nội dung khỏi summary, hoặc dùng LLM để tóm tắt (chế độ live đã dùng `SummarizationMiddleware`).
- **`Response quality` ở chế độ offline là điểm heuristic**: 70% độ phủ fact, 15% ngắn gọn, 15% có bullet. Nó không đo được độ tự nhiên của câu trả lời. Chế độ `--live` dùng LLM làm giám khảo (`judge_model`).
- **Chế độ live chưa có kết quả benchmark sạch.** Phần nối các thành phần (tool call, dynamic prompt, summarization middleware, đọc `usage_metadata`) đã được kiểm tra bằng model giả. Lần chạy thật đầu tiên với `gpt-4o-mini` làm lộ ra một lỗi race condition: LangGraph chạy song song các lời gọi tool `save_user_fact`, và lần đọc-sửa-ghi `User.md` không có khóa đã xóa mất `name` và `drink`. Lỗi đã được sửa bằng lock và ghi file nguyên tử (`os.replace`), kèm test hồi quy `test_parallel_upserts_do_not_lose_facts`. Vì lần chạy đó dùng code có lỗi, số liệu của nó không được dùng. Lần chạy đó vẫn cho thấy 3 xu hướng cần xác nhận lại bằng một lần chạy sạch:
  - Ở bộ Standard, prompt tokens của Advanced cao hơn Baseline nhiều hơn so với chế độ offline (khoảng +170% so với +60%), vì mỗi lượt còn phải gửi system prompt và schema của tool.
  - Câu trả lời thật của LLM dài hơn, nên compact đã chạy ngay cả ở bộ Standard.
  - Baseline đạt recall khoảng 0.1 dù không có bộ nhớ, vì thước đo kiểm tra chuỗi con bị đánh lừa bởi các câu trả lời chung chung (có chứa "Python", "AI", "ngắn gọn").
  - Ngoài ra, ở chế độ live LLM lưu rất nhiều thứ vào `User.md`. Ví dụ, `style` bị gộp cả "chạy bộ lúc 6 giờ sáng". Đây là bằng chứng thực tế cho rủi ro phình file và lưu sai đã nêu ở mục 3.5.

## 5. Phần mở rộng (bonus)

### 5.1. Confidence threshold (bonus chính)

**Giải quyết vấn đề gì?** Người dùng thường nói về *kế hoạch*, *phỏng đoán* hoặc *giả định*, chẳng hạn "có lẽ tháng sau mình sẽ chuyển ra Hà Nội" hay "mình đang cân nhắc chuyển sang data engineer". Regex vẫn bắt được fact trong những câu này, nhưng fact đó chưa đúng ở thời điểm hiện tại. Nếu ghi ngay vào `User.md`, fact sai sẽ đi theo người dùng qua mọi phiên (rủi ro 2 ở mục 3.5).

**Cách làm** (`memory_store.py`: `extract_profile_candidates`, `agent_advanced.py`: `_remember`):

1. **Độ tin cậy gốc** theo loại fact (`BASE_CONFIDENCE`): `name` 0.95, `drink`/`food` 0.9, `location`/`profession`/`pet` 0.85, `style` 0.8, `interests` 0.75.
2. **Trừ 0.35** khi câu có từ do dự: `có lẽ`, `hình như`, `chắc là`, `định`, `sắp`, `sẽ`, `đang cân nhắc`. Cũng trừ 0.35 khi câu **bắt đầu** bằng `Nếu`, trừ trường hợp fact là style.
3. **Cộng 0.1** khi câu có từ xác nhận: `đính chính`, `nhớ là`, `hiện tại`, `thực ra`.
4. Chỉ ghi vào `User.md` khi độ tin cậy ≥ `min_fact_confidence`. Giá trị mặc định là 0.7, đổi được qua biến `MIN_FACT_CONFIDENCE` trong `.env`.
5. Fact chưa đủ ngưỡng được đưa vào `pending_facts`. Nếu người dùng nhắc lại đúng fact đó, độ tin cậy được cộng dồn theo công thức `1 − (1 − a)(1 − b)`, ví dụ hai lần 0.5 thành 0.75, và fact được đưa vào `User.md`.

**Tác dụng, so sánh có và không có threshold** (test `test_hedged_statements_*`, `test_without_threshold_*`, `test_repeated_pending_fact_is_promoted`):

| | Ngưỡng 0.0 (tắt) | Ngưỡng 0.7 |
|---|---|---|
| 6 câu do dự làm hỏng hồ sơ đang đúng | **6/6** | **0/6** |
| 4 câu khẳng định hoặc đính chính được lưu | 4/4 | 4/4 |
| Recall Standard / Stress | 1.00 / 1.00 | 1.00 / 1.00 |
| Memory growth Standard | 256 B | 256 B |

Threshold chặn được toàn bộ lỗi lưu sai trên bộ câu thử, mà không làm giảm recall và không tốn thêm token, vì chỉ có thêm vài phép so sánh chuỗi. Cả 36 fact thật trong 2 bộ dữ liệu đều vượt ngưỡng.

**Rủi ro và đánh đổi** (phát hiện trong lúc làm):

- **Bỏ sót fact thật.** Bản đầu coi mọi chữ `nếu` là do dự, và chặn nhầm 2 fact thật trong dữ liệu: *"…mình đang ở Huế để dùng ví dụ địa phương **nếu cần**"* và *"**Nếu** bạn giải thích, hãy trả lời ngắn gọn…"*. Mình đã sửa thành chỉ tính `nếu` khi nó đứng đầu câu, và miễn trừ cho style. Bài học: từ khóa do dự luôn có ngoại lệ, và đặt ngưỡng chặt quá thì recall giảm.
- **Nhắc lại một kế hoạch có thể biến nó thành fact.** "Có lẽ sẽ chuyển ra Hà Nội" rồi "dự định chuyển ra Hà Nội" cho 0.5 + 0.5 → 0.75, và nơi ở hiện tại bị đổi thành Hà Nội dù người dùng chưa chuyển. Cách cộng dồn hợp với phỏng đoán về hiện tại ("hình như…"), nhưng không hợp với kế hoạch tương lai. Cách sửa tốt hơn là tách kế hoạch thành một trường riêng (`planned_location`) thay vì cộng dồn.
- **Cộng dồn chỉ khi giá trị khớp chính xác.** "phở" và "phở thật" bị coi là hai fact khác nhau; một test đã bị fail vì đúng lỗi này. Cần chuẩn hóa giá trị trước khi so sánh.
- **`pending_facts` chỉ nằm trong RAM**, nên khởi động lại agent là mất. Muốn bền thì phải lưu ra file riêng, không đưa vào `User.md` để file này không phình.
- Các con số 0.35, 0.1 và 0.7 được chọn bằng tay. Với LLM thật, có thể để model tự trả về độ tin cậy qua tool `save_user_fact`.

### 5.2. Các mở rộng khác đã có trong code

**Conflict handling**
- *Giải quyết gì:* khi người dùng đính chính, agent không giữ đồng thời fact cũ và fact mới.
- *Cách làm:*
  - Xóa mệnh đề phủ định (`chứ không còn…`, `không phải…`) trước khi khớp regex.
  - Trong một câu, lấy kết quả khớp **cuối cùng** (*"lúc đầu ở Huế, nhưng giờ ở Đà Nẵng"* → Đà Nẵng).
  - `upsert_fact` ghi đè giá trị cũ.
- *Tác dụng:* các câu hỏi về correction (nghề MLOps, nơi ở Huế hoặc Đà Nẵng) trả lời đúng 100%.
- *Rủi ro:* luôn tin câu mới nhất. Nếu người dùng nói đùa mà không có từ khóa nhiễu, fact đúng sẽ bị ghi đè.

**Trích fact có cấu trúc (entity extraction)**
- *Cách làm:* fact được tách thành các trường cố định (`name`, `location`, `profession`, `drink`, `food`, `pet`, `style`, `interests`) thay vì lưu nguyên câu.
- *Tác dụng:*
  - `User.md` gọn và đi ngang ở ~256 byte.
  - Trả lời đúng trường được hỏi.
  - Ghi đè được theo từng trường.
- *Rủi ro:* thông tin không thuộc trường nào sẽ bị bỏ qua, ví dụ "chạy bộ lúc 6 giờ sáng" hay "đi biển Mỹ Khê".

**Không lưu nhầm khi người dùng đặt câu hỏi**
- *Cách làm:* bỏ câu kết thúc bằng `?`, bỏ giá trị là từ để hỏi (`gì`, `nào`, `đâu`, `ai`), bỏ câu chứa từ khóa nhiễu.
- *Tác dụng:* chặn được lỗi đã từng làm recall tụt (mục 3.1).
- *Rủi ro:* danh sách từ khóa phải bảo trì tay.

### 5.3. Hướng tiếp theo (chưa làm)

**Memory decay**
- Lưu thêm `updated_at` và số lần được nhắc lại cho mỗi fact.
- Fact lâu không được nhắc thì giảm ưu tiên, hoặc chuyển ra khỏi phần prompt luôn được chèn.
- Cách này giải quyết rủi ro 1 và 3 ở mục 3.5.

## 6. Kết luận

1. **Baseline** không nhớ qua phiên (recall 0.00), nhưng rẻ ở hội thoại ngắn.
2. **`User.md`** đưa recall lên 1.00, với cái giá là một chi phí cố định khoảng 60 token cho mỗi lượt.
3. **Hội thoại dài** làm prompt cost của Baseline tăng theo bình phương số lượt.
4. **Compact memory** giữ context mỗi lượt đi ngang quanh 450 token, nhờ đó giảm 67% prompt tokens ở bộ stress mà không mất recall.
5. Hệ thống mạnh hơn nhưng phức tạp hơn. Chất lượng của nó phụ thuộc vào việc **lưu đúng fact**, nên cần guardrail: xử lý correction, lọc câu hỏi và nhiễu, và confidence threshold. Threshold chặn 6/6 câu do dự mà không làm giảm recall. Bước tiếp theo là memory decay.
