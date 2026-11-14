"""Group tests — T02 pipeline (src/pipeline.py, src/baseline.py).

Success path: full 5-step run on the real t02 fixture.
Failure/edge: rank() without a `score` column; invalid reviewed-file rows;
              night-scene semantics (dark-but-structured stays KEEP, near-black is flagged).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline import run as run_baseline
from src.common.context import RunContext
from src.common.io import set_seed
from src.common.review import AuditLog, load_review_decisions
from src.pipeline import compute_refs, process, rank, review, run as run_poc, score_components

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "fixtures" / "t02_raw_qc"
OWNER = "NgocAnh (solo)"


def make_ctx(tmp_path: Path, out_name: str = "out", pipeline_cfg: dict | None = None) -> RunContext:
    cfg = {"track": "t02", "owner": OWNER, "group_id": "G00", "strict_provenance": True,
           "params": {} if pipeline_cfg is None else pipeline_cfg}
    return RunContext(track="t02", input_dir=FX, output_dir=tmp_path / out_name, config=cfg,
                      seed=42, rng=set_seed(42), run_name="test", offline=True,
                      audit=AuditLog(tmp_path / out_name / "audit_log.jsonl"))


# ---------------------------------------------------------------- success: full E2E
@pytest.mark.parametrize("runner", [run_poc, run_baseline], ids=["poc", "baseline"])
def test_t02_end_to_end(tmp_path, runner):
    ctx = make_ctx(tmp_path)
    runner(ctx)
    pred = pd.read_csv(tmp_path / "out" / "ranked_review.csv")
    assert list(pred["image_id"]) and {"image_id", "score"} <= set(pred.columns)
    assert pred["score"].between(0, 1).all() or (pred["score"] <= 0).any()  # bounded PoC scores; baseline uses -log(lap)
    # 5-step contract artifacts
    q = pd.read_csv(tmp_path / "out" / "review_queue.csv")
    assert len(q) == 20 and list(q["rank"]) == list(range(1, 21))
    assert set(q["evidence_path"]) != {""}          # contact-sheet evidence links
    assert (tmp_path / "out" / "contact_sheet.html").exists()
    dec = json.loads((tmp_path / "out" / "decision.json").read_text(encoding="utf-8"))
    assert dec["owner"] == OWNER and dec["reason_code"] and dec["status"] in {"GO", "HOLD", "REWORK"}
    audit = (tmp_path / "out" / "audit_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert audit and all(json.loads(l)["actor"] for l in audit)
    # ranking quality sanity: blurred (lap ~ 2-5) must outrank sharp dev images (lap ~ 1000+)
    poc_top = set(pred.sort_values("score", ascending=False).head(20)["image_id"])
    assert not poc_top or "img_18_3" in poc_top  # known blurred cam_B dev image (lap=2)


def test_t02_poc_flags_are_light_aware(tmp_path):
    """Failure mode from the handout: usable night scenes must not be mass-flagged as dark."""
    ctx = make_ctx(tmp_path)
    data = {"images": pd.read_csv(FX / "images.csv")}
    scored = process(ctx, data)
    night = scored[scored["light"] == "night"]
    # structured night scenes (mean ~30, high lap) are KEEP; only genuinely broken ones are flagged
    kept = night[night["pred_action"] == "KEEP"]
    assert len(kept) >= 8
    assert (night[night["pred_defect_type"] == "dark"]["score"] > 0.5).all()


# ---------------------------------------------------------------- failure / edge
def test_t02_rank_requires_score_column(tmp_path):
    ctx = make_ctx(tmp_path)
    with pytest.raises(ValueError, match="score"):
        rank(ctx, pd.DataFrame({"image_id": ["a", "b"], "path": ["x.jpg", "y.jpg"]}))


def test_t02_reviewed_file_rejects_bad_rows(tmp_path):
    p = tmp_path / "reviewed.csv"
    base = {"rank": [1], "item_id": ["img_00_0"], "score": [1.0], "reason": [""], "suggested_action": [""],
            "slice": [""], "evidence_path": [""], "override_value": [""], "review_note": [""], "reviewed_at": [""]}
    # unknown decision value
    pd.DataFrame(base | {"reviewer": ["an"], "reviewer_decision": ["maybe"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="reviewer_decision"):
        load_review_decisions(p)
    # decision without an owner
    pd.DataFrame(base | {"reviewer": [""], "reviewer_decision": ["approve"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="reviewer"):
        load_review_decisions(p)


def test_t02_review_override_changes_action_and_audits(tmp_path):
    ctx = make_ctx(tmp_path)
    data = {"images": pd.read_csv(FX / "images.csv")}
    scored = process(ctx, data)
    queue = rank(ctx, scored)
    top_id = queue.iloc[0]["item_id"]
    reviewed = queue.head(1).copy()
    reviewed["reviewer"], reviewed["reviewer_decision"] = "tester", "edit"
    reviewed["override_value"] = "KEEP"            # human disagrees: do not recapture
    reviewed["review_note"] = "false alarm on dev"
    reviewed.to_csv(ctx.output_dir / "review_queue_reviewed.csv", index=False)
    final = review(ctx, scored, queue)
    row = final[final["image_id"] == top_id].iloc[0]
    assert row["pred_action"] == "KEEP"
    assert ctx.stats.get("n_overridden") == 1
    log = [json.loads(l) for l in (ctx.output_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["action"] == "override:edit" and e["actor"] == "tester" for e in log)


# ---------------------------------------------------------------- unit: night semantics
def test_t02_dark_scoring_keeps_structured_night():
    refs = {"cam_loglap": {"c1": {"median": 6.5, "mad": 0.3}}, "global_loglap": {"median": 6.5, "mad": 0.3},
            "bref": {}, "light_fallback": {"night": 32.0, "day": 118.0}, "cam_fallback": {},
            "global_bright": 110.0, "ref_split": "dev"}
    params = {"flag_threshold": 0.5, "blur": {"z_scale": 4.0}, "dark": {"ratio_floor": 0.45},
              "glare": {"offset": 0.05, "scale": 0.10}, "compression": {"offset": 0.10, "scale": 0.10}}
    usable_night = {"lap": 900.0, "mean": 29.0, "std": 32.0, "sat_global": 0.0, "sat_tile": 0.0, "blockiness": 0.02,
                    "camera_id": "c1", "light": "night"}
    near_black = usable_night | {"lap": 300.0, "mean": 3.0, "std": 5.0}
    day_blurred = usable_night | {"lap": 2.0, "mean": 118.0, "light": "day"}
    s_ok = score_components(usable_night, refs, params)
    s_black = score_components(near_black, refs, params)
    s_blur = score_components(day_blurred, refs, params)
    assert s_ok["dark"] < 0.5 and s_ok["blur"] < 0.5          # usable night: no defect
    assert s_black["dark"] >= 0.5                              # near-black night: flagged
    assert s_blur["blur"] >= 0.5                               # day blur: flagged
