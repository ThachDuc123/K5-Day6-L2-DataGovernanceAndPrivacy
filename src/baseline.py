"""T02 *baseline* (group-owned): a single global Laplacian-variance threshold.

Exactly the bar the handout sets for T02: one threshold on `image_ops.
laplacian_variance` for every image regardless of camera or light. The PoC
(src/pipeline.py) must beat it on the same pool / budget / seed.

    python -m src.main --track t02 --pipeline src.baseline --output out/baseline --run-name baseline --offline

The threshold tau is chosen WITHOUT labels: tau = the `tau_percentile`-th
percentile of Laplacian variance over the dev split (default 10, frozen in
configs/lab.yaml). Everything below tau is flagged RECAPTURE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .common.image_ops import laplacian_variance, load_rgb
from .metrics.tracks import get_track

REQUIRED_COLUMNS = ["image_id", "path", "camera_id", "light", "split"]
DEFAULTS = {"budget": 20, "dev_split": "dev", "tau_percentile": 10.0,
            "decision": {"go_max_flag_rate": 0.10, "rework_min_flag_rate": 0.35}}


def load(ctx: RunContext) -> pd.DataFrame:
    manifest = ctx.input_dir / "manifest.csv"
    if manifest.exists():
        problems = verify_manifest(manifest)
        if problems:
            if ctx.config.get("strict_provenance", True):
                raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    images = pd.read_csv(ctx.input_dir / "images.csv")
    missing = [c for c in REQUIRED_COLUMNS if c not in images.columns]
    if missing:
        raise ValueError(f"images.csv missing required columns {missing}")
    ctx.audit.log(actor="baseline", action="load", item_id=str(ctx.input_dir), after={"n_images": len(images)})
    return images


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    p = {**DEFAULTS, **(ctx.config.get("params") or {})}
    images = data["images"]
    laps = []
    for r in images.itertuples(index=False):
        laps.append(laplacian_variance(load_rgb(ctx.input_dir / r.path)))
    df = images.copy()
    df["lap"] = laps
    dev = df[df["split"] == p["dev_split"]]
    if len(dev) == 0:
        raise ValueError(f"no images in dev split '{p['dev_split']}' — cannot choose tau")
    tau = float(np.percentile(dev["lap"], p["tau_percentile"]))
    ctx.stats["tau"] = tau
    df["score"] = (-np.log(np.maximum(df["lap"], 1e-6))).round(6)  # higher = blurrier = review first
    df["pred_defect_type"] = np.where(df["lap"] < tau, "blur", "none")
    df["pred_action"] = np.where(df["lap"] < tau, "RECAPTURE", "KEEP")
    df["reason"] = [f"global laplacian={l:.1f} {'<' if l < tau else '>='} tau={tau:.1f}" for l in df["lap"]]
    ctx.stats["flag_rate"] = float((df["pred_action"] == "RECAPTURE").mean())
    return df.drop(columns=["lap"])


def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    p = {**DEFAULTS, **(ctx.config.get("params") or {})}
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column (higher = review first)")
    q = make_review_queue(scored, id_col="image_id", score_col="score", budget=int(p["budget"]),
                          reason_col="reason", action_col="pred_action", slice_col="light",
                          evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                         title=f"t02 baseline (global Laplacian threshold) — review queue ({ctx.run_name})")
    return q


def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if reviewed_path.exists():
        reviewed = load_review_decisions(reviewed_path)
        ctx.stats["n_reviewed"] = len(reviewed)
        return apply_overrides(scored, reviewed, id_col="image_id", value_col="pred_action", audit=ctx.audit)
    return scored


def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    p = {**DEFAULTS, **(ctx.config.get("params") or {})}
    d = p["decision"]
    rate = float((final["pred_action"] == "RECAPTURE").mean())
    return Decision(status="HOLD", reason_code="BASELINE_ONLY",
                    reason=f"Single global Laplacian threshold (tau={ctx.stats.get('tau', 0):.1f}), flagged {rate:.2f}: baseline only, never release on this.",
                    owner=ctx.config.get("owner") or "TODO-owner", scope=ctx.track,
                    evidence=[str(ctx.out("review_queue.csv"))], metrics={"flag_rate": rate, "tau": ctx.stats.get("tau", 0.0)})


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = {"images": load(ctx)}
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    cols = spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]
    final[cols].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
