"""PoC pipeline — T02 Raw QC Triage (group G00).

Idea: one global rule cannot tell "blurry" from "dark night scene" from "soft
optics of cam_B", so we score each defect family separately and normalise every
feature against statistics computed on the **dev split only** (no labels used,
no eval leakage — thresholds frozen in configs/lab.yaml before evaluation).

Feature families (per image, see compute_features):
  blur         log Laplacian variance, robust z-score within camera_id
               (cam_B has softer optics -> camera-level normalisation)
  dark         mean brightness ratio vs the (camera, light) dev reference
               (night scenes are dark by design -> light-aware reference +
               near-black absolute rule; structured night stays KEEP)
  glare        saturated-pixel fraction, global and per-tile (4x4 grid) to
               catch *local* glare that a global threshold misses
  compression  JPEG blockiness: gradient at 8-px block boundaries minus
               gradient elsewhere, normalised by contrast

Final `score` = max weighted component (higher = review first);
`pred_defect_type` = argmax component when flagged; `pred_action` =
RECAPTURE when score >= flag_threshold else KEEP.

Lab contract (5 steps, run by `python -m src.main ...`):
    01 load      images.csv + manifest provenance check
    02 process   features -> component scores -> flag
    03 rank      top-B review queue (budget from track config)
    04 review    apply reviewer overrides (pred_action) + audit log
    05 decide    GO / HOLD / REWORK from the flagged-rate policy (frozen)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .common.image_ops import brightness_stats, laplacian_variance, load_rgb, to_gray
from .metrics.tracks import get_track

REQUIRED_COLUMNS = ["image_id", "path", "camera_id", "light", "split"]

# Defaults; every value here can be overridden (and is frozen) in configs/lab.yaml.
DEFAULTS = {
    "budget": 20,
    "dev_split": "dev",
    "stat_ref_split": "dev",       # all normalisation statistics come from this split only
    "min_ref_n": 4,                # min images in a (camera, light) dev group to trust its median
    "flag_threshold": 0.5,         # score >= this -> pred_action = RECAPTURE
    "weights": {"blur": 1.0, "dark": 1.0, "glare": 1.0, "compression": 1.0},
    "blur": {"z_scale": 4.0},      # blur_score = clip(-z_loglap / z_scale, 0, 1)
    "dark": {"ratio_floor": 0.45}, # dark_score = clip((ratio_floor - mean/ref) / ratio_floor, 0, 1)
    "glare": {"offset": 0.05, "scale": 0.10},  # glare_score = clip((sat - offset)/scale, 0, 1)
    "compression": {"offset": 0.10, "scale": 0.10},
    "decision": {"go_max_flag_rate": 0.10, "rework_min_flag_rate": 0.35},
}


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> pd.DataFrame:
    manifest = ctx.input_dir / "manifest.csv"
    if manifest.exists():
        problems = verify_manifest(manifest)
        if problems:
            ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
            if ctx.config.get("strict_provenance", True):
                raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    images = pd.read_csv(ctx.input_dir / "images.csv")
    missing = [c for c in REQUIRED_COLUMNS if c not in images.columns]
    if missing:
        raise ValueError(f"images.csv missing required columns {missing}")
    if len(images) == 0:
        raise ValueError("images.csv is empty — nothing to triage")
    ctx.audit.log(actor="pipeline", action="load", item_id=str(ctx.input_dir), after={"n_images": len(images)})
    return images


# ---------------------------------------------------------------- 02 process (features)
def compute_features(img: np.ndarray) -> dict:
    """Per-image QC features. Pure numpy/Pillow — deterministic, offline."""
    g = to_gray(img)
    feat = {"lap": laplacian_variance(img)}
    feat |= brightness_stats(img)
    # local glare: worst 4x4 tile by saturated fraction
    H, W = g.shape[:3] if g.ndim == 3 else g.shape
    th, tw = H // 4, W // 4
    sat_tile = 0.0
    if th > 0 and tw > 0:
        for r in range(4):
            for c in range(4):
                tile = g[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                sat_tile = max(sat_tile, float((tile > 245).mean()))
    feat["sat_tile"] = sat_tile
    # JPEG blockiness: gradient on 8-px block boundaries vs elsewhere (contrast-normalised)
    gx = np.abs(np.diff(g, axis=1))          # (H, W-1)
    gy = np.abs(np.diff(g, axis=0))          # (H-1, W)
    col_b = np.arange(gx.shape[1]) % 8 == 7  # vertical 8-px block boundaries
    row_b = np.arange(gy.shape[0]) % 8 == 7  # horizontal 8-px block boundaries
    b = (gx[:, col_b].mean() if col_b.any() else 0.0) + (gy[row_b, :].mean() if row_b.any() else 0.0)
    o = (gx[:, ~col_b].mean() if (~col_b).any() else 0.0) + (gy[~row_b, :].mean() if (~row_b).any() else 0.0)
    feat["blockiness"] = (b - o) / (float(g.std()) + 1e-6)
    feat["sat_global"] = feat.pop("frac_saturated")
    return feat


def extract_features(images: pd.DataFrame, input_dir, audit=None) -> pd.DataFrame:
    rows = []
    for r in images.itertuples(index=False):
        img = load_rgb(input_dir / r.path)
        rows.append({"image_id": r.image_id, "camera_id": r.camera_id, "light": r.light,
                     "split": r.split, "path": r.path} | compute_features(img))
    feat = pd.DataFrame(rows)
    if audit:
        audit.log(actor="pipeline", action="features_extracted", item_id="all",
                  after={"n": len(feat), "features": [c for c in feat.columns if c not in
                          ("image_id", "camera_id", "light", "split", "path")]})
    return feat


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med))) if len(x) else 0.0


def compute_refs(feat: pd.DataFrame, params: dict) -> dict:
    """Normalisation statistics from the reference split (dev) ONLY."""
    ref_split = params["stat_ref_split"]
    dev = feat[feat["split"] == ref_split]
    if len(dev) == 0:
        raise ValueError(f"no images in reference split '{ref_split}' — cannot normalise")
    loglap = np.log(np.maximum(dev["lap"].values, 1e-6))
    cam_loglap = {}
    for cam, sub in dev.groupby("camera_id"):
        v = np.log(np.maximum(sub["lap"].values, 1e-6))
        cam_loglap[cam] = {"median": float(np.median(v)), "mad": max(_mad(v), 1e-3)}
    global_loglap = {"median": float(np.median(loglap)), "mad": max(_mad(loglap), 1e-3)}
    # brightness reference: (camera, light) -> light (any camera) -> camera -> global
    bref: dict[tuple, float] = {}
    for (cam, light), sub in dev.groupby(["camera_id", "light"]):
        if len(sub) >= params["min_ref_n"]:
            bref[(cam, light)] = float(np.median(sub["mean"]))
    light_fallback = {light: float(np.median(sub["mean"])) for light, sub in dev.groupby("light")
                      if len(sub) >= params["min_ref_n"]}
    cam_fallback = {cam: float(np.median(sub["mean"])) for cam, sub in dev.groupby("camera_id")
                    if len(sub) >= params["min_ref_n"]}
    global_bright = float(np.median(dev["mean"]))
    return {"cam_loglap": cam_loglap, "global_loglap": global_loglap, "bref": bref,
            "light_fallback": light_fallback, "cam_fallback": cam_fallback,
            "global_bright": global_bright, "ref_split": ref_split}


def _bright_ref(cam: str, light: str, refs: dict) -> float:
    for key in [(cam, light), (None, light)]:
        if key in refs["bref"]:
            return refs["bref"][key]
    if light in refs["light_fallback"]:
        return refs["light_fallback"][light]
    if cam in refs["cam_fallback"]:
        return refs["cam_fallback"][cam]
    return refs["global_bright"]


def score_components(f: dict, refs: dict, params: dict) -> dict:
    """Four defect-family scores in [0, 1]; higher = more suspicious."""
    out = {}
    # blur: robust z of log Laplacian variance within the same camera
    ll = np.log(max(f["lap"], 1e-6))
    ref = refs["cam_loglap"].get(f["camera_id"], refs["global_loglap"])
    z = (ll - ref["median"]) / ref["mad"]
    out["blur_z"] = z
    out["blur"] = float(np.clip(-z / params["blur"]["z_scale"], 0.0, 1.0))
    # dark: brightness ratio vs the (camera, light) reference; night stays usable
    ratio = f["mean"] / max(_bright_ref(f["camera_id"], f["light"], refs), 1e-6)
    out["bright_ratio"] = ratio
    out["dark"] = float(np.clip((params["dark"]["ratio_floor"] - ratio) / params["dark"]["ratio_floor"], 0.0, 1.0))
    # glare: global OR local (tile) saturation
    sat = max(f["sat_global"], f["sat_tile"])
    out["sat"] = sat
    out["glare"] = float(np.clip((sat - params["glare"]["offset"]) / params["glare"]["scale"], 0.0, 1.0))
    # compression: JPEG block artifacts
    out["compression"] = float(np.clip((f["blockiness"] - params["compression"]["offset"])
                                       / params["compression"]["scale"], 0.0, 1.0))
    # a dark image also loses Laplacian variance; do not double-count it as blur
    if out["dark"] >= params["flag_threshold"]:
        out["blur"] = min(out["blur"], 0.3)
    return out


def _reason(comp: dict, f: dict, refs: dict) -> str:
    parts = []
    if comp["blur"] > 0.2:
        ref = refs["cam_loglap"].get(f["camera_id"], refs["global_loglap"])
        parts.append(f"blur (lap={f['lap']:.0f} vs {ref['median']:.1f} dev-median log-lap, z={comp['blur_z']:.1f})")
    if comp["dark"] > 0.2:
        parts.append(f"dark (mean={f['mean']:.0f}, ratio={comp['bright_ratio']:.2f} of dev ref)")
    if comp["glare"] > 0.2:
        parts.append(f"glare (saturated frac={comp['sat']:.2f}, tile={f['sat_tile']:.2f})")
    if comp["compression"] > 0.2:
        parts.append(f"compression (blockiness={f['blockiness']:.2f})")
    return "; ".join(parts) if parts else "no strong defect signal"


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    images = data["images"]
    p = {**DEFAULTS, **(ctx.config.get("params") or {})}
    p["weights"] = {**DEFAULTS["weights"], **(p.get("weights") or {})}
    feat = extract_features(images, ctx.input_dir, ctx.audit)
    refs = compute_refs(feat, p)
    ctx.stats["n_ref_images"] = int((feat["split"] == p["stat_ref_split"]).sum())
    rows = []
    for f in feat.to_dict(orient="records"):
        comp = score_components(f, refs, p)
        weighted = {k: comp[k] * p["weights"][k] for k in ("blur", "dark", "glare", "compression")}
        score = float(max(weighted.values()))
        flagged = score >= p["flag_threshold"]
        defect = max(weighted, key=weighted.get) if flagged else "none"
        rows.append({"image_id": f["image_id"], "path": f["path"], "camera_id": f["camera_id"],
                     "light": f["light"], "split": f["split"], "score": round(score, 6),
                     "pred_defect_type": defect,
                     "pred_action": "RECAPTURE" if flagged else "KEEP",
                     "reason": _reason(comp, f, refs)})
    out = pd.DataFrame(rows)
    ctx.stats["flag_rate"] = float((out["pred_action"] == "RECAPTURE").mean())
    feat.to_csv(ctx.out("qc_features.csv"), index=False)   # transparency: full feature table
    return out


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    p = {**DEFAULTS, **(ctx.config.get("params") or {})}
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column (higher = review first)")
    q = make_review_queue(scored, id_col="image_id", score_col="score", budget=int(p["budget"]),
                          reason_col="reason", action_col="pred_action", slice_col="light",
                          evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                         title=f"t02 Raw QC Triage — review queue ({ctx.run_name})")
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Apply reviewer overrides to `pred_action`; every change goes to the audit log."""
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        return scored
    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = len(reviewed)
    final = apply_overrides(scored, reviewed, id_col="image_id", value_col="pred_action", audit=ctx.audit)
    changed = (final["pred_action"] != scored["pred_action"]).sum()
    ctx.stats["n_overridden"] = int(changed)
    return final


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    p = {**DEFAULTS, **(ctx.config.get("params") or {})}
    d = p["decision"]
    rate = float((final["pred_action"] == "RECAPTURE").mean())
    owner = ctx.config.get("owner") or "TODO-owner"
    if rate >= d["rework_min_flag_rate"]:
        return Decision(status="REWORK", reason_code="DEFECT_RATE_TOO_HIGH",
                        reason=f"Flagged rate {rate:.2f} >= {d['rework_min_flag_rate']}: capture session is systematically broken, recapture.",
                        owner=owner, scope=ctx.track, evidence=[str(ctx.out("review_queue.csv"))],
                        metrics={"flag_rate": rate})
    if rate <= d["go_max_flag_rate"]:
        return Decision(status="GO", reason_code="DEFECT_RATE_WITHIN_TOLERANCE",
                        reason=f"Flagged rate {rate:.2f} <= {d['go_max_flag_rate']}: release with the flagged images held out for review.",
                        owner=owner, scope=ctx.track, evidence=[str(ctx.out("review_queue.csv"))],
                        metrics={"flag_rate": rate})
    return Decision(status="HOLD", reason_code="DEFECT_RATE_NEEDS_REVIEW",
                    reason=f"Flagged rate {rate:.2f} is between {d['go_max_flag_rate']} and {d['rework_min_flag_rate']}: needs human sign-off.",
                    owner=owner, scope=ctx.track, evidence=[str(ctx.out("review_queue.csv"))],
                    metrics={"flag_rate": rate})


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = {"images": load(ctx)}                       # 01 load
    scored = process(ctx, data)                       # 02 process
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)                         # 03 rank
    final = review(ctx, scored, queue)                # 04 review
    cols = spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]
    final[cols].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
