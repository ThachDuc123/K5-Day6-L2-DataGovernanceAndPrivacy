# Demo script — T02 Raw QC Triage (Group G00, 3–5 phút)

## Chuẩn bị trước (offline, không download gì)
```bash
make setup
make preflight          # 0 FAIL
python -m src.main --input fixtures/t02_raw_qc --config configs/lab.yaml --output out/poc --run-name poc --offline --seed 42
python -m src.main --track t02 --pipeline src.baseline --output out/baseline --run-name baseline --offline --seed 42
```

## 1. Vấn đề (30 giây)
- Budget review chỉ B=20 ảnh → phải xếp hạng đúng ảnh lỗi nhất lên đầu.
- Baseline handout = 1 ngưỡng Laplacian toàn ảnh → chỉ bắt blur cực đoan (10/96), bỏ trắng dark/glare/compression.

## 2. Chạy live (1 lệnh, ~1 giây)
```bash
python -m src.main --config configs/lab.yaml --output out/poc --run-name poc --offline --seed 42
```
Chỉ ra: exit 0, `out/poc/ranked_review.csv`, `run_info.json` (seed 42, runtime ~0.6 s), `decision.json` (REWORK, reason_code, owner).

## 3. Contact sheet — human-in-the-loop (1 phút)
Mở `out/poc/contact_sheet.html` bằng trình duyệt (offline, thumbnail base64):
- 20 ảnh ưu tiên + reason (blur z, dark ratio, glare tile, blockiness).
- Nhập tên reviewer → chọn quyết định cho 1–2 ảnh (ví dụ `edit` → `KEEP`) → Export CSV →
  lưu thành `out/poc/review_queue_reviewed.csv` → chạy lại lệnh trên → chỉ ra
  `audit_log.jsonl` có thêm dòng `override:edit` (append-only, ai/sửa gì/lý do).

## 4. 3 failure mode của handout — minh hoạ trên fixture (1 phút)
| Failure mode | Ảnh chứng minh | Metadata |
|---|---|---|
| Đêm hữu ích bị loại hết | `img_01_2` (flag: mean=3) **vs** 14 ảnh night KEEP (mean 30–78, lap 235–2263) | chuẩn hoá theo (camera, light) |
| Glare cục bộ bị bỏ qua | `img_17_2` — sat toàn ảnh chỉ 0.24, nhưng tile 4×4 = 1.00 | tile-based glare |
| cam_B optics mềm bị báo nhầm | cam_B day lap ~400 vẫn KEEP; chỉ lap ≈ 2–6 mới flag | z-score log-lap theo camera |

## 5. Reproducibility + tests (30 giây)
```bash
pytest -q   # 54 passed, 39 skipped (skip = cần overlay gt/ giảng viên)
```
- Chạy 2 lần cùng seed 42 → `ranked_review.csv` byte-identical (đã diff).
- Guards: `--offline` chặn network (exit 4), GT guard chặn đọc `gt/` (exit 5).

## 6. Kết quả + điều trung thực (30 giây)
- Flag rate: PoC 0.406 (39/96) vs baseline 0.104 (10/96); baseline = 0 trên 3/4 họ lỗi.
- Recall@B / Precision@B **chưa tự chấm được** vì `gt/` ở overlay giảng viên — số cuối do
  `python -m src.evaluate` quyết định; nhóm không chỉnh file metrics tay.
- Decision hiện tại: REWORK (flag_rate 0.406 ≥ 0.35) — buổi chụp lỗi hệ thống.
