# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T02 · Raw QC Triage (`fixtures/t02_raw_qc/`) |
| Primary metric (tên đúng như evaluator in ra) | **Defect Recall@B** (kèm **Precision@B**, cài sẵn trong `src/metrics/tracks.py::eval_t02`) |
| Tăng tốt hay giảm tốt? | higher (cả Recall@B và Precision@B) |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | **image** (96 ảnh synthetic 320×180; eval split = 40 ảnh theo cảnh gốc) |
| Công thức + numerator / denominator | Recall@B = #(ảnh lỗi thật (is_defect=1) xuất hiện trong top-B theo `score`) / #(ảnh lỗi thật trên split eval); Precision@B = #(ảnh lỗi thật trong top-B) / B |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | join theo `image_id`; xếp theo `score` giảm dần; chỉ tính ảnh có trong GT eval split |
| Candidate pool cố định | toàn bộ 96 ảnh trong `fixtures/t02_raw_qc/images.csv` (dev 56 / eval 40, chia **theo cảnh gốc** img_XX) |
| Review budget B / K | **B = 20** (`configs/tracks/t02.yaml`, không đổi giữa baseline và PoC) |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Không dùng nhãn. Mọi thống kê chuẩn hoá (median/MAD log-Laplacian theo `camera_id`, median brightness theo `(camera_id, light)`) chỉ tính trên split **dev** (`params.stat_ref_split: dev`, `min_ref_n: 4`). Các hằng số ngưỡng frozen trong `configs/lab.yaml` trước khi chạy eval: `flag_threshold: 0.5`, `blur.z_scale: 4.0`, `dark.ratio_floor: 0.45`, `glare: offset 0.05/scale 0.10`, `compression: offset 0.10/scale 0.10`, trọng số 4 họ lỗi đều 1.0, decision `go_max_flag_rate: 0.10` / `rework_min_flag_rate: 0.35`. Baseline dùng tau = percentile-10 của Laplacian variance trên dev (chạy runtime, không nhãn). |
| Eval split (không dùng để tune) | split **eval** (40 ảnh, 8 cảnh gốc chưa từng thấy ở dev) — chỉ giảng viên chấm qua `src.evaluate` |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | False alarm trong top-B theo slice `light` (day/dusk/night) — ảnh sạch bị kéo vào top-B làm reviewer lãng phí budget |
| Slices sẽ báo cáo (≥2) | (1) `light=day/dusk/night` — đặc biệt ảnh đêm hữu ích không bị flag; (2) `defect_type` (blur/glare/dark/compression); (3) `camera_id` cam_A vs cam_B (optics mềm) |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Eval N=40, B=20: báo kèm số tuyệt đối (x/20) chứ không chỉ %; slice night chỉ có 8 ảnh eval → luôn kèm N trong bảng slice |
| Ngày giờ chốt + người chốt | 2026-11-14 · NgocAnh (bài cá nhân) (trước khi nhìn bất kỳ kết quả eval nào; repo học viên không có `gt/`) |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.

**Lưu ý trung thực:** do `gt/labels.csv` chỉ có ở overlay giảng viên, nhóm không tự tính được Recall@B trước khi nộp.
Mọi con số nhóm tự đo được (không cần nhãn) đều xuất phát từ `run_info.json` / `qc_features.csv` do script sinh ra:
flag rate, phân bổ theo light/camera, tỷ lệ ảnh đêm hữu ích bị giữ (KEEP), và Failure case phân tích bằng metadata.
Recall@B/Precision@B sẽ do giảng viên chạy `src.evaluate` và đối chiếu — nhóm không chỉnh file metrics tay.
