# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G00 |
| Challenge | T02 · Raw QC Triage — ảnh mờ, lóa và thiếu sáng |
| Thành viên (5–6) + vai trò | Điền tên thật vào `reports/team_contributions.csv` + Bảng 2 (mục 10) khi chốt nhóm: 1. ____ (Lead/Integrator) · 2. ____ (Data & fixtures) · 3. ____ (Core algorithm) · 4. ____ (QA/metric) · 5. ____ (Review + report) · 6. ____ |
| Git URL | https://github.com/ThachDuc123/K5-Day6-L2-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | ____ (chạy `make tag-demo` khi freeze demo) |
| Tag `final-24h` (commit) | ____ (chạy `make tag-final` khi nộp) |
| Deadline (giờ VN) | ____ |

## 01 · Problem statement

- **Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai?** Đội thu thập dữ liệu xe và người dùng dataset: ảnh lỗi (mờ/tối/lóa/nén vỡ) lọt vào dataset huấn luyện làm model học tín hiệu nhiễu; ảnh tốt bị loại oan làm đội chụp lại tốn tiền và mất thời gian. Reviewer bị "ngập" nếu queue toàn false alarm.
- **Điểm nghẽn cụ thể:** mỗi buổi chụp trả về hàng nghìn ảnh, chỉ có budget review B=20 ảnh/ngày. Một ngưỡng Laplacian toàn ảnh duy nhất không phân biệt được: (a) ảnh tối vì **ban đêm** (bình thường) và ảnh tối vì **chụp hỏng**; (b) camera optics mềm (cam_B) và ảnh mờ thật; (c) lóa **cục bộ** (vùng biển số cháy sáng) khi toàn ảnh vẫn ổn.
- **Đầu vào → đầu ra:** `fixtures/t02_raw_qc/` (96 ảnh + `images.csv`: image_id, path, camera_id, light, split) → `out/ranked_review.csv` (image_id, score, pred_defect_type, pred_action=KEEP/RECAPTURE) + `review_queue.csv` top-B + contact sheet + `decision.json`.
- **Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval):** theo tỷ lệ ảnh bị flag trong run: ≤ 0.10 → **GO** (release, ảnh flagged giữ lại review riêng); ≥ 0.35 → **REWORK** (buổi chụp lỗi hệ thống, cần chụp lại); giữa hai ngưỡng → **HOLD** (cần người ký duyệt). Mọi decision bắt buộc `reason_code` + `owner` (QA Lead — Group G00).
- **Giả định của nhóm (cần DPO/legal/domain owner xác nhận):** fixture là synthetic CC0, không có người/biển số thật; khi chạy trên ảnh thật (BDD100K) phải giữ nguyên quy tắc không đưa PII chưa có quyền lên repo/cloud. Không tuyên bố "đạt GDPR" từ pipeline này — đây chỉ là gate kỹ thuật QC, không phải pháp lý.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 (14 cảnh × 4 variant) | `fixtures/t02_raw_qc/` — sinh bởi `tools/make_fixtures.py` (giảng viên, seed 42) | CC0-1.0 (synthetic) | Có |
| eval / heldout | 40 (10 cảnh × 4 variant) | cùng nguồn, split **theo cảnh gốc** (img_XX) — không rò rỉ giữa dev/eval | CC0-1.0 (synthetic) | Có |

- **Schema & class mapping:** `images.csv: image_id, path, camera_id (cam_A/cam_B), light (day/dusk/night), split`. GT ẩn (chỉ giảng viên): `gt/labels.csv: defect_type ∈ {none, blur, glare, dark, compression}, severity 0–2, is_defect`.
- **Seed:** 42 cho mọi run (`run_info.json` ghi lại); mỗi cảnh gốc có đúng 4 ảnh variant, chia split theo `group_id = img_XX` trong manifest.
- **`manifest.csv` verified (sha256):** có — `python -m tools.verify_manifests` và `tools.preflight`: 0 FAIL (sau khi ép `core.autocrlf=false` và re-checkout fixtures; checkout Windows CRLF làm lệch hash CSV — đã fix ở máy làm việc, xem mục 09).
- **Dữ liệu thật dùng (nếu có):** chưa dùng (`data/real/` chưa được giảng viên cấp). Nếu dùng: ảnh đêm BDD100K là hard negative; nhóm sẽ tự gán nhãn ≥10 ảnh và khai báo protocol GT trong report, không commit PII thật.

## 03 · Research & design

- **References:**
  1. ACDC — Adverse Conditions Dataset (ETH Zürich, https://acdc.vision.ee.ethz.ch/, xem 2026-11-14): ảnh đêm/mưa có *cấu trúc* hợp lệ → động lực cho việc không flag mọi ảnh tối.
  2. ISO/IEC TR 5259-6:2026 (analytics data quality for AI): QC theo *slice* điều kiện chụp (light/camera) thay vì ngưỡng toàn cục.
  3. CVAT automated QA (https://docs.cvat.ai/, xem 2026-11-14): review queue ưu tiên theo score + human override có audit.
- **Sơ đồ data flow:** `load()` (verify sha256 manifest + images.csv) → `process()` (features: log-Laplacian z-score theo camera, brightness ratio theo (camera, light), saturated fraction global+tile 4×4, JPEG blockiness → 4 score blur/dark/glare/compression → score = max) → `rank()` (top-B=20, reason + evidence) → `review()` (override từ `review_queue_reviewed.csv`, `apply_overrides`, AuditLog JSONL append-only) → `decide()` (GO/HOLD/REWORK theo flag-rate policy frozen) → `run_info.json` + `qc_features.csv` (full feature table để kiểm tra chéo).
- **Thành phần offline:** 100% numpy/Pillow/scipy, không model pretrained, không network (`--offline` + offline guard). Mọi run đều reproducible: chạy 2 lần cùng seed 42 cho `ranked_review.csv` **byte-identical** (đã verify bằng diff).
- **Trade-offs đã cân nhắc:**
  - Chuẩn hoá theo `camera_id` cho blur (cam_B optics mềm, lap ~400 vs cam_A ~1100 nhưng đều nét) — đánh đổi: nếu cam_B thật sự mờ hệ thống thì sẽ bị "bảo hộ" quá mức → nêu trong limitations.
  - Reference brightness theo `(camera, light)`, fallback (any camera, light) → cam_A night eval (không có dev night) mượn thống kê night của cam_B.
  - Khi ảnh tối ≥ ngưỡng flag, **triệt tiêu** thành phần blur (ảnh tối lap thấp là hệ quả) để không đếm đôi một ảnh lỗi.
  - Glare dùng tile 4×4 để bắt lóa cục bộ (img_17_2: global sat 0.24 nhưng tile 1.00).

## 04 · Baseline vs PoC

- **Baseline** (`src/baseline.py`, config `tau_percentile: 10` trên dev): một ngưỡng Laplacian variance **toàn ảnh, toàn cục duy nhất** — tau = percentile-10 của dev = **5.29**. `pred_action=RECAPTURE` khi lap < tau; score = −log(lap). Đúng định nghĩa baseline trong handout T02.
- **PoC** (`src/pipeline.py`, params frozen trong `configs/lab.yaml`): 4 họ feature + chuẩn hoá dev-only, chi tiết ở mục 03.
- **Cùng budget / cùng heldout / cùng seed:** xác nhận — cả hai chạy `--seed 42`, cùng pool 96 ảnh, cùng B=20, cùng split dev để chọn tham số, cùng eval split để giảng viên chấm.

**Bảng 1 · Baseline vs PoC & hard slices**

> ⚠️ `reports/metrics_baseline.json`, `reports/metrics_poc.json`, `reports/table1_baseline_vs_poc.md` **chưa sinh được** trên repo học viên vì `gt/labels.csv` chỉ có ở overlay giảng viên (`make eval-baseline eval-poc compare` → exit 2). Giảng viên chạy lệnh sau khi cài overlay; nhóm **không điền tay** các file này. Bảng dưới ghi phần nhóm tự đo được không cần nhãn (nguồn: `out/*/run_info.json` + `qc_features.csv`).

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Flag rate (fraction, toàn pool) | 0.104 (10/96) | 0.406 (39/96) | Baseline chỉ bắt blur cực đoan; PoC bắt thêm dark/glare/compression; higher flag rate ≠ tốt hơn — Recall@B/Precision@B do giảng viên chấm |
| Defect types được phát hiện (count) | chỉ `blur` (10) | blur 12 · dark 10 · glare 4 · compression 13 | Baseline = 0 cho 3/4 họ lỗi |
| Defect Recall@B / Precision@B | **đợi overlay giảng viên** | **đợi overlay giảng viên** | lệnh: `make eval-baseline eval-poc compare TRACK=t02` |
| Ảnh đêm hữu ích bị flag (hard slice) | 2/20 ảnh night bị flag (một phần do lap đêm thấp) | 6/20 night bị flag nhưng **tất cả đều hỏng thật** (near-black mean≈3, lap≈2, blockiness≈0.26–0.34); 14 night KEEP có lap 235–2263, mean 30–78 | chính là failure mode 1 của handout |
| cam_B optics mềm bị flag oan (hard slice) | 2 flagged toàn pool | cam_B day: 7/12 KEEP; 5 flagged đều có lap ≈ 2–6 (mờ thật) hoặc blockiness ≈ 0.22–0.25 | failure mode 3 của handout |
| Runtime / 100 items | 0.40 s / 96 ≈ **0.42 s / 100** | 0.64 s / 96 ≈ **0.67 s / 100** | ≈ 6.7 ms/ảnh, CPU Intel x64 Model 154, Python 3.11.9, Windows |

## 05 · Evaluation protocol

- **Công thức primary metric:** `Defect Recall@B` = #(GT defects trong top-B theo score) / #(GT defects, split eval); `Precision@B` = #(GT defects trong top-B) / B; đơn vị = image; denominator cố định B=20. Chi tiết đầy đủ: `reports/metric_contract.md`.
- **Threshold / budget B / K và lúc chốt:** B=20 lấy từ `configs/tracks/t02.yaml`; mọi ngưỡng frozen trong `configs/lab.yaml` (xem Threshold policy) — chốt 2026-11-14, trước khi nhìn kết quả eval nào.
- **Độ bất định khi N nhỏ:** eval N=40 → báo kèm số tuyệt đối (x/20); slice night eval chỉ 8 ảnh → luôn kèm N; nếu chỉ 1–2 ảnh lệch thì nêu case-by-case trong bảng slice, không nói "% cải thiện" trên N nhỏ.
- **Lệnh evaluate chính xác:**
  ```bash
  python -m src.evaluate --track t02 --pred out/baseline/ranked_review.csv --run-name baseline --out reports/metrics_baseline.json
  python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv      --run-name poc      --out reports/metrics_poc.json
  python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
  ```

## 06 · Result slices

Số nhóm tự đo được (không cần GT), từ `out/poc/ranked_review.csv` (seed 42):

| Slice | N | Baseline flag | PoC flag | Ghi chú |
|---|---|---|---|---|
| light=day | 56 | 6 | 25 | gồm blur + glare + compression |
| light=dusk | 20 | 2 | 8 | |
| light=night | 20 | 2 | 6 | 6 flagged đều hỏng thật (near-black/mờ/nén); 14 KEEP có cấu trúc tốt (lap 235–2263) |
| camera=cam_B | 28 | 2 | 9 | optics mềm — chuẩn hoá theo camera giúp không flag oan ảnh nét |
| camera=cam_A | 68 | 8 | 30 | |

- **Failure case 1 (đã phát hiện trên fixture, minh hoạ bằng metadata):** `img_17_2` (cam_A, day, eval) — lóa cục bộ: toàn ảnh mean=181, sat_global=0.24 nhưng có tile 4×4 sat **= 1.00**; Laplacian giảm (465 vs ~1100 cùng cảnh). Ngưỡng Laplacian toàn cục (baseline) bỏ qua; PoC flag glare nhờ tile. Root cause: glare là hiện tượng *vùng*, không phải toàn ảnh.
- **Failure case 2:** `img_01_2` (cam_A, night, eval) — ảnh gần đen hoàn toàn (mean=3, ratio=0.11 so với ref night≈33): PoC flag `dark`. Ngược lại các ảnh night hữu ích (mean 30–78, lap 235–2263) đều KEEP. Rủi ro còn lại: ảnh night vừa tối vừa nhẹ độ tương phản có thể lọt giữa hai ngưỡng `ratio_floor=0.45` — cần reviewer xác nhận.
- **Failure case 3 (cam_B optics mềm):** nếu không chuẩn hoá theo `camera_id`, một nửa cam_B day (lap ~400) sẽ bị ngưỡng cam_A (lap ~1100) báo mờ oan. PoC dùng z-score log-lap trong từng camera → chỉ lap ≈ 2–6 mới bị flag.
- **Latency / compute (máy, CPU, ms/item):** PoC ≈ 6.7 ms/ảnh (0.64 s / 96 ảnh), Intel x64 Family 6 Model 154, Python 3.11.9, Windows-10-10.0.26200 — ghi trong `run_info.json`.

## 07 · Human & decision

- **Review queue:** `out/poc/review_queue.csv` — top-B=20 theo `score` giảm dần, cột `reason` (diễn giải từng họ lỗi + z-score/tỷ lệ), `slice` = light, `evidence_path` = ảnh gốc; kèm **contact sheet offline** `out/poc/contact_sheet.html` (thumbnail base64, dropdown quyết định, nút Export CSV — xem `src/common/contact_sheet.py`).
- **Cách override + audit log:** reviewer điền `reviewer`, `reviewer_decision` (approve/reject/edit/escalate/skip), `override_value`, `review_note` rồi lưu thành `out/poc/review_queue_reviewed.csv`; chạy lại `make run` → `review()` đọc file (thiếu reviewer → fail rõ ràng), `apply_overrides` ghi đè `pred_action`, mỗi thay đổi append 1 dòng JSONL vào `out/audit_log.jsonl` (actor/action/before/after/reason, append-only). Test: `tests/test_pipeline_t02.py::test_t02_review_override_changes_action_and_audits`.
- **Decision policy + `reason_code` + owner:** GO (`DEFECT_RATE_WITHIN_TOLERANCE`) khi flag_rate ≤ 0.10; REWORK (`DEFECT_RATE_TOO_HIGH`) khi ≥ 0.35; HOLD (`DEFECT_RATE_NEEDS_REVIEW`) ở giữa. Owner = `QA Lead — Group G00` (config `owner`). Run hiện tại: flag_rate = 0.406 → **REWORK** với evidence là review queue (hợp lý: 39/96 ảnh lỗi hệ thống, không thể "release kèm review").
- **Khi nào HOLD/REWORK thay vì tự động release:** mọi flag_rate trong dừng mù (0.10–0.35); mọi run có `n_reviewed > 0` nhưng còn quyết định `escalate`; hoặc khi pool có camera/light chưa từng thấy ở dev (thống kê fallback) → bắt buộc HOLD.

## 08 · Reproduce & tests

```bash
make setup          # hoặc: python -m venv .venv && pip install -r requirements.txt
python -m src.main --input fixtures/t02_raw_qc --config configs/lab.yaml --output out/ --offline --seed 42
# baseline:
python -m src.main --track t02 --pipeline src.baseline --output out/baseline --run-name baseline --offline --seed 42
python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --out reports/metrics_poc.json   # cần overlay gt/
pytest -q
```

- **Output của `pytest -q`:** **54 passed, 39 skipped** (39 skip = test cần `gt/` của overlay giảng viên; `pytest.skip` theo conftest). Nhóm thêm 7 test mới trong `tests/test_pipeline_t02.py`: E2E PoC + baseline, light-awareness, rank thiếu `score` → ValueError, reviewed-file sai decision/thiếu reviewer → ValueError, override + audit log, unit night-semantics.
- **Máy / CPU / OS / Python:** Windows-10-10.0.26200 · Intel x64 Model 154 · Python 3.11.9 · git sha tại thời điểm viết: `b0ce8bf` (base) + commit của nhóm (điền sau khi commit).
- **Lưu ý Windows:** máy làm việc không có `make` trên PATH và venv Windows dùng `.venv/Scripts/python.exe` (Makefile/run.sh mặc định `.venv/bin/python`). Tương đương đã chạy trực tiếp: `PY=.venv/Scripts/python.exe ./run.sh t02` hoặc gọi `python -m src.main / -m pytest` như ở trên — tất cả exit 0.

## 09 · Limitations & ethics

- **Rủi ro pháp lý / PII / license:** fixture synthetic CC0 — an toàn. Ảnh thật (nếu dùng) có người/biển số → **không commit, không đưa lên cloud/public repo**; phải ở `data/real/` (gitignored). Pipeline này là gate kỹ thuật, **không** tuyên bố tuân thủ GDPR/luật VN.
- **Lỗi môi trường đã xử lý:** checkout trên Windows với `core.autocrlf=true` làm CSV fixtures thành CRLF → sha256 lệch manifest → `pytest` fail 14 test. Fix: `git config core.autocrlf false` + re-checkout `fixtures/`. Bài học: nên có `.gitattributes` (`* text eol=lf`) cho repo — đề xuất cho giảng viên.
- **Điều chưa biết; false alarms; scale 10×:** Recall/Precision thật chưa đo được không có GT — có thể PoC flag thừa compression (13 ảnh, blockiness 0.20–0.34) nếu fixture ít lỗi nén; các ảnh glare borderline (sat=0.10, đúng bằng offset+scale=0.15 ngưỡng… img_02_2, img_20_3 score 0.50–0.52) có thể là false alarm → đúng việc cho reviewer top-B. Scale 10×: vẫn ~7 ms/ảnh = ~7 phút/10k ảnh (CPU) — ổn; thống kê dev nên cập nhật định kỳ theo camera mới.
- **Giả định cần DPO/legal xác nhận:** định nghĩa "ảnh dùng được" cho ban đêm do nhóm đặt (mean ≥ ~13 = 0.4×ref night) — cần domain owner xác nhận theo yêu cầu nghiệp vụ thực tế.
- **Kế hoạch cải tiến:** (1) ROI scoring cho biển số/vùng quan trọng; (2) threshold theo từng nhóm camera học từ dữ liệu nhiều buổi chụp; (3) thêm metric phụ blockiness-based precision khi có GT; (4) kiểm 10 ảnh out-of-domain (mốc +24h).

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

> Điền tên/email thật sau khi mọi thành viên commit; quy tắc rubric: mỗi người ≥1 commit thực chất (commit chỉ đổi khoảng trắng không tính).

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Member 1 | (điền) | Integration / CLI · wiring config | | |
| Member 2 | (điền) | Data + GT protocol · fixtures/provenance | | |
| Member 3 | (điền) | Core algorithm (`src/pipeline.py`, `src/baseline.py`) | | |
| Member 4 | (điền) | Metric + tests (`tests/test_pipeline_t02.py`) | | |
| Member 5 | (điền) | Review queue + report + contact sheet | | |
| Member 6 (nếu có) | (điền) | Research / ablation / debate | | |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| https://acdc.vision.ee.ethz.ch/ (ACDC, ETH Zürich) | 2026-11-14 | Ảnh adverse-condition (đêm/mưa/sương) vẫn có cấu trúc hợp lệ → không được coi mọi ảnh tối là lỗi. |
| ISO/IEC TR 5259-6:2026 (analytics data quality) | 2026-11-14 | Khung QC dữ liệu cho AI: kiểm tra theo slice điều kiện thu thập thay vì một ngưỡng duy nhất. |
| https://docs.cvat.ai/ (CVAT automated QA) | 2026-11-14 | Mô hình queue ưu tiên + human-in-the-loop có audit trail — tham chiếu cho bước review. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [x] Link repo trong README; input mẫu không chứa PII thật (synthetic CC0).
- [x] Đúng 1 lệnh run (`python -m src.main ...` exit 0), 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy (offline guard, không model ngoài).
- [x] Có baseline/PoC outputs + 3 failure cases phân tích bằng metadata + reviewer evidence (contact sheet, review queue) + decision log.
- [x] Có license/source provenance (manifest sha256 verified), data splits, seed 42, offline environment, eval N ghi rõ trong metric contract.
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên (cần tên/email thật của nhóm).
