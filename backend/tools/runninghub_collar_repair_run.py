"""Submit ONE minimal, single-pass local collar repair to RunningHub (P2.9).

This tool is deliberately *not* the two-pass Clothing -> Hair generator in
``tools.runninghub_crop_stitch_run``. The brief for this round freezes the best
render and forbids regenerating the whole look to fix one local defect:

* baseline  = ``evidence/p29-submit-run/final.png`` (task 2108330225116082177),
  sha256 ``1b9634d1eb3f24738d9caf67d1009b21053a9790a1947400c67449bc5331590a``;
* defect    = the pale knit collar strip the batch passes left under the chin;
* fix       = one masked inpaint of that strip only, on the finished frame.

It shares the proven CropAndStitch graph, checkpoint and CropAndStitch settings
with ``runninghub_crop_stitch_run`` by *importing* them, so "compatible with the
last successful configuration" is a property of the code, not a promise. The
only structural differences are:

    * the image is the finished render, not the original photo;
    * the mask is ``collar_repair_mask.png`` (released residual) rather than the
      clothing/hair silhouettes;
    * there is exactly ONE pass -- one KSampler, one InpaintCropImproved, one
      InpaintStitchImproved, one SaveImage. No hair pass.

Node graph (ids stable so the gate can assert the wiring)::

    1  LoadImage             <- final.png
    2  LoadImageMask (red)   <- collar_repair_mask.png
    3  CheckpointLoaderSimple
    4  InpaintCropImproved   <- 1, 2
    5  CLIPTextEncode +      <- repair prompt
    6  CLIPTextEncode -      <- repair negatives (white/beige inner layer, trim)
    7  InpaintModelConditioning <- 5, 6, 3.vae, 4.cropped_image, 4.cropped_mask, noise_mask
    8  KSampler              <- 3.model, 7.(pos,neg,latent), denoise < 1.0
    9  VAEDecode
    10 InpaintStitchImproved <- stitcher 4, inpainted 9  (restores the 904x1190 canvas)
    11 SaveImage

Stages, from ``backend/``::

    # build + validate + offline gate only -- no network, no key
    ../.venv/bin/python -m tools.runninghub_collar_repair_run --dry-run

    # upload the two inputs (free)
    ../.venv/bin/python -m tools.runninghub_collar_repair_run --upload

    # create the one billed task
    ../.venv/bin/python -m tools.runninghub_collar_repair_run \
        --submit --i-understand-this-costs-money

Gate 8 is the keep-vs-remove conflict the brief asks about: the central pale
block under the jaw sits inside the subject's own skin guard, so "keep the
original pixel" and "remove the residual" cannot both be satisfied there. When
that region is non-empty the run HALTs by default (exit 2) and reports the
exact pixel count and bbox. ``--accept-guarded-residual`` proceeds knowing the
central block is left to the operator, and the released strip -- the only
region the mask paints -- is still repaired.

The API key is read from ``~/.runninghub_key`` and is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from tools.runninghub_crop_stitch_run import (  # shared, proven graph config
    CHECKPOINT,
    CROP_DEFAULTS,
    DENOISE_BAND,
    SAMPLER,
    PROTECTED_CORE_ERODE_PX,
    _crop_box,
    _mask_pipeline_np,
    validate_checkpoint,
    validate_workflow,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The frozen best render this round repairs. The sha256 is the identity check;
#: a different frame must not be silently substituted.
BASELINE_FINAL = REPO_ROOT / "evidence/p29-submit-run/final.png"
BASELINE_SHA256 = "1b9634d1eb3f24738d9caf67d1009b21053a9790a1947400c67449bc5331590a"
EXPECTED_FRAME = (904, 1190)

DEFAULT_MASK = REPO_ROOT / "evidence/p29-collar-repair/collar_repair_mask.png"
DEFAULT_NO_EDIT = REPO_ROOT / "evidence/p29-collar-repair/collar_repair_no_edit.png"
DEFAULT_STATS = REPO_ROOT / "evidence/p29-collar-repair/collar_repair_stats.json"
DEFAULT_LABELS = REPO_ROOT / "evidence/p29-collar-repair/labels_fresh.png"

#: Denoise sits in the shared band, one notch below the clothing pass's proven
#: 0.75: the frame is already correct nearly everywhere, so the repair wants the
#: smallest strength that still replaces the pale knit inside the mask.
COLLAR_REPAIR_DENOISE = 0.70

FILENAME_PREFIX = "p29_collar_repair"

#: Minimal repair prompt. One job: continue the surrounding blouse across the
#: masked strip. No facial/identity words, no accessory, no second garment.
POSITIVE_PROMPT = (
    "repaint only the masked area as the same deep olive-green cloth as the "
    "surrounding blouse, continuing the existing collar into one natural closed "
    "neckline with soft vintage folds, matching the neighbouring colour, shading "
    "and fabric texture"
)

#: A short, targeted negative list -- the artefacts a repair at the neckline can
#: actually produce. Deliberately NOT the dozens of terms the full generator
#: accumulated: this pass repaints ~3.3k px and needs only the "no lighter inner
#: layer" rules the last run proved necessary.
NEGATIVE_PROMPT = (
    "white collar, cream collar, beige collar lining, ivory lining, white inner "
    "layer, white undershirt, pale knit at the neckline, beige knitted scarf, "
    "white scarf, lace trim, ribbon, bow, necklace, duplicated neckline, "
    "extra collar flap, blurry, lowres, watermark"
)

#: Words a masked edit must never put in a positive prompt (a whole head is in
#: the crop as context; naming it reads as "draw it"). Same rule as the generator.
FORBIDDEN_POSITIVE_WORDS = (
    "face", "facial", "eye", "eyebrow", "nose", "lip", "mouth", "teeth",
    "smile", "glasses", "person", "portrait", "identity", "chin", "neck",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_workflow(final_name: str, mask_name: str, seed: int) -> dict:
    """The single-pass repair graph. One sampler, one crop, one stitch."""
    crop = CROP_DEFAULTS
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": final_name}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "4": {
            "class_type": "InpaintCropImproved",
            "inputs": {"image": ["1", 0], "mask": ["2", 0], **crop},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": POSITIVE_PROMPT, "clip": ["3", 1]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["3", 1]},
        },
        "7": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["3", 2],
                "pixels": ["4", 1],
                "mask": ["4", 2],
                "noise_mask": True,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed,
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2],
                **SAMPLER,
                "denoise": COLLAR_REPAIR_DENOISE,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 2]}},
        "10": {
            "class_type": "InpaintStitchImproved",
            "inputs": {"stitcher": ["4", 0], "inpainted_image": ["9", 0]},
        },
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": FILENAME_PREFIX}},
    }


def _load_gray(path: Path) -> "np.ndarray":
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def _hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    diff = mx - mn
    hue = np.zeros_like(mx)
    nz = diff > 1e-6
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    idx = nz & (mx == r)
    hue[idx] = ((g - b)[idx] / diff[idx]) % 6
    idx = nz & (mx == g)
    hue[idx] = ((b - r)[idx] / diff[idx]) + 2
    idx = nz & (mx == b)
    hue[idx] = ((r - g)[idx] / diff[idx]) + 4
    hue *= 60.0
    sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return hue, sat


def collect_gate_checks(workflow: dict, object_info: dict | None, *,
                        final_path: Path, mask_path: Path, no_edit_path: Path,
                        stats_path: Path, labels_path: Path) -> tuple[list[str], dict]:
    """The eight offline gates from the round brief. No network, all local."""
    errors: list[str] = []
    report: dict = {
        "checkpoint": CHECKPOINT,
        "denoise": COLLAR_REPAIR_DENOISE,
        "denoise_band": list(DENOISE_BAND),
        "prompt": POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "shares_config_with_proven_run": {
            "checkpoint": CHECKPOINT,
            "crop": {k: CROP_DEFAULTS[k] for k in
                     ("mask_fill_holes", "mask_expand_pixels", "mask_blend_pixels",
                      "context_from_mask_extend_factor", "output_target_width",
                      "output_target_height", "output_resize_to_target_size")},
            "sampler": SAMPLER,
        },
    }

    # -- gate 1: the repaired image is the frozen best final --------------------
    final_sha = _sha256(final_path)
    with Image.open(final_path) as image:
        final_size = image.size
    report["gate1_input_is_best_final"] = {
        "path": str(final_path.relative_to(REPO_ROOT)),
        "sha256": final_sha,
        "expected_sha256": BASELINE_SHA256,
        "matches_frozen_best": final_sha == BASELINE_SHA256,
        "size": list(final_size),
    }
    if final_sha != BASELINE_SHA256:
        errors.append(f"gate1: input is not the frozen best final (sha256 {final_sha})")

    # -- gate 2: exactly one local pass ---------------------------------------
    samplers = sorted(nid for nid, n in workflow.items() if n["class_type"] == "KSampler")
    crops = sorted(nid for nid, n in workflow.items() if n["class_type"] == "InpaintCropImproved")
    conds = sorted(nid for nid, n in workflow.items() if n["class_type"] == "InpaintModelConditioning")
    stitches = sorted(nid for nid, n in workflow.items() if n["class_type"] == "InpaintStitchImproved")
    saves = sorted(nid for nid, n in workflow.items() if n["class_type"] == "SaveImage")
    forbidden = sorted(nid for nid, n in workflow.items()
                       if n["class_type"] in {"EmptyLatentImage", "EmptySD3LatentImage", "VAEEncodeForInpaint"})
    report["gate2_single_local_pass"] = {
        "ksampler_nodes": samplers,
        "inpaint_crop_nodes": crops,
        "inpaint_model_conditioning_nodes": conds,
        "inpaint_stitch_nodes": stitches,
        "save_nodes": saves,
        "passes": 1 if len(samplers) == 1 else len(samplers),
    }
    if len(samplers) != 1:
        errors.append(f"gate2: expected exactly one KSampler, found {samplers}")
    if len(crops) != 1 or len(conds) != 1 or len(stitches) != 1:
        errors.append("gate2: the graph is not a single crop->condition->stitch pass")
    if len(saves) != 1:
        errors.append(f"gate2: expected one SaveImage, found {saves}")
    if forbidden:
        errors.append(f"gate2: a text-to-image / fill node is present {forbidden}")
    if workflow["10"]["inputs"]["stitcher"] != ["4", 0] or workflow["4"]["inputs"]["image"] != ["1", 0]:
        errors.append("gate2: the repaired pixels do not come from the loaded final via one crop")
    if workflow["8"]["inputs"]["latent_image"] != ["7", 2]:
        errors.append("gate2: the sampler latent is not the encoded crop")
    denoise = workflow["8"]["inputs"].get("denoise")
    if denoise is None or not DENOISE_BAND[0] <= denoise <= DENOISE_BAND[1]:
        errors.append(f"gate2: denoise {denoise} is outside the proven band {DENOISE_BAND}")
    if workflow["7"]["inputs"]["noise_mask"] is not True:
        errors.append("gate2: noise_mask must be on or the mask is not honoured")

    # prompt hygiene: no facial/identity word (whole word -- "neckline" is fine,
    # "neck" as a standalone noun is not), and short.
    lowered = POSITIVE_PROMPT.lower()
    hits = [w for w in FORBIDDEN_POSITIVE_WORDS if re.search(rf"\b{re.escape(w)}\b", lowered)]
    report["prompt_hygiene"] = {"forbidden_words_found": hits, "chars": len(POSITIVE_PROMPT),
                                "negative_terms": len([t for t in NEGATIVE_PROMPT.split(",") if t.strip()])}
    if hits:
        errors.append(f"prompt names a facial/identity word {hits}")

    # -- gates 3/4/5: mask scope, disjointness, no correct-content edits -------
    #
    # Two regions are distinguished on purpose. The *raw mask* is what the
    # operator authored (``collar_repair_mask.png``). The *painted region* is
    # what InpaintCropImproved will actually hand the sampler after its
    # hipass -> expand -> blur prep: the raw mask plus a ``mask_blend_pixels``
    # feather rim, which the stitch composites as generated*alpha + original*(1-alpha).
    # Identity is argued on the raw mask (strictly disjoint) and the feather is
    # bounded and reported; the strict "core" test uses the same erosion the
    # proven two-pass gate used, so the two tools share one definition of "the
    # generated pixels must not reach the protected interior".
    from scipy import ndimage
    from app.domains.portrait.human_parsing import ATR_CLASSES

    mask_raw = _load_gray(mask_path) > 127
    no_edit = _load_gray(no_edit_path) > 127
    with Image.open(final_path) as image:
        final_rgb = np.array(image.convert("RGB"))
    labels = _load_gray(labels_path)
    hair = labels == [k for k, v in ATR_CLASSES.items() if v == "Hair"][0]
    background = labels == [k for k, v in ATR_CLASSES.items() if v == "Background"][0]
    hue, sat = _hue_sat(final_rgb)
    olive = (hue >= 45.0) & (hue <= 115.0) & (sat > 0.12)

    prepared = _mask_pipeline_np(mask_raw.astype(np.float32),
                                 blend_pixels=int(CROP_DEFAULTS["mask_blend_pixels"]),
                                 expand_pixels=int(CROP_DEFAULTS["mask_expand_pixels"]),
                                 fill_holes=bool(CROP_DEFAULTS["mask_fill_holes"]))
    painted = prepared > 0.5
    feather = painted & ~mask_raw
    feather_max_dist = 0.0
    if feather.any():
        feather_max_dist = float(ndimage.distance_transform_edt(~mask_raw)[feather].max())

    def _bbox(m):
        ys, xs = np.where(m)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if m.any() else None

    def _within(inner, outer, tol=0):
        return (inner is not None and outer is not None
                and inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
                and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)

    def _guard_stats(guard, erode_px):
        inter = painted & guard
        depth = float(ndimage.distance_transform_edt(~guard)[inter].max()) if inter.any() else 0.0
        raw_inter = mask_raw & guard
        raw_depth = float(ndimage.distance_transform_edt(~guard)[raw_inter].max()) if raw_inter.any() else 0.0
        core = ndimage.binary_erosion(guard, iterations=erode_px) if guard.any() else guard
        return {
            "guard_px": int(guard.sum()),
            "raw_mask_intersection_px": int(raw_inter.sum()),
            "raw_mask_max_depth_px": raw_depth,
            "painted_intersection_px": int(inter.sum()),
            "painted_max_depth_px": depth,
            "core_erode_px": erode_px,
            "painted_into_core_px": int((painted & core).sum()),
        }

    stats = json.loads(stats_path.read_text())
    window_bbox = stats.get("window_bbox")
    feather_tol = int(CROP_DEFAULTS["mask_blend_pixels"]) + 1
    report["gate3_mask_is_collar_only"] = {
        "raw_mask_bbox": _bbox(mask_raw),
        "painted_bbox": _bbox(painted),
        "release_window_bbox": window_bbox,
        "feather_tolerance_px": feather_tol,
        "raw_mask_bbox_in_window": _within(_bbox(mask_raw), window_bbox),
        "painted_bbox_within_window_plus_feather": _within(_bbox(painted), window_bbox, feather_tol),
        "mask_stats_release_px": stats["counts"]["release_px"],
        "mask_stats_release_px_in_collar_class": stats["counts"]["release_px_in_collar_class"],
        "mask_stats_release_over_hair_px": stats["counts"]["release_over_hair_px"],
        "mask_stats_release_over_background_px": stats["counts"]["release_over_background_px"],
        "mask_stats_release_over_protected_px": stats["counts"]["release_over_protected_px"],
    }

    # ``no_edit`` also folds in the already-correct olive so the *painted* region
    # can be proven disjoint from it, but the identity argument is made against
    # the guards proper: identity, teeth, glasses, chin/throat, the subject's own
    # skin inside the collar window, parsed hair and background. Those are the
    # pixels "keep the original" is non-negotiable for.
    guards = no_edit & ~olive
    guards_g = _guard_stats(guards, PROTECTED_CORE_ERODE_PX)
    report["gate4_painted_disjoint_from_no_edit"] = {
        "guards_px": int(guards.sum()),
        "raw_mask_disjoint_from_guards": int((mask_raw & guards).sum()) == 0,
        "raw_mask_over_guards_px": int((mask_raw & guards).sum()),
        "feather_rim_px": int(feather.sum()),
        "feather_rim_max_distance_px": feather_max_dist,
        "note": ("the feather is the node's mask_blend_pixels transition; it is "
                 "bounded and composites against the original, it does not replace it"),
        **guards_g,
    }

    hair_g = _guard_stats(hair, 3)
    olive_g = _guard_stats(olive, 3)
    bg_g = _guard_stats(background, 3)
    report["gate5_no_correct_content_edited"] = {
        "painted_px": int(painted.sum()),
        "hair": hair_g,
        "olive_blouse": olive_g,
        "background": bg_g,
        "olive_blouse_px_total": int(olive.sum()),
    }

    if int(painted.sum()) == 0:
        errors.append("gate3: the repair mask is empty")
    if not report["gate3_mask_is_collar_only"]["raw_mask_bbox_in_window"]:
        errors.append("gate3: the authored mask leaves the collar window")
    if not report["gate3_mask_is_collar_only"]["painted_bbox_within_window_plus_feather"]:
        errors.append(f"gate3: the painted region leaves the collar window by more than {feather_tol} px")
    if feather_max_dist > int(CROP_DEFAULTS["mask_blend_pixels"]):
        errors.append(f"gate4: the feather reaches {feather_max_dist:.1f} px, beyond mask_blend_pixels")
    if guards_g["raw_mask_intersection_px"]:
        errors.append(f"gate4: the authored mask touches {guards_g['raw_mask_intersection_px']} px "
                      f"of identity / skin / hair / background guard")
    if guards_g["painted_into_core_px"]:
        errors.append(f"gate4: the painted region reaches {guards_g['painted_into_core_px']} px "
                      f"into the guard core (eroded {PROTECTED_CORE_ERODE_PX} px)")
    for label, g in (("hair", hair_g), ("olive blouse", olive_g), ("background", bg_g)):
        if g["painted_into_core_px"]:
            errors.append(f"gate5: the painted region reaches {g['painted_into_core_px']} px "
                          f"into the {label} core")

    # -- gate 6: original 904x1190 coordinates are preserved -------------------
    box = _crop_box(prepared, int(CROP_DEFAULTS["output_target_width"]),
                    int(CROP_DEFAULTS["output_target_height"]), 32,
                    float(CROP_DEFAULTS["context_from_mask_extend_factor"]), *final_size)
    uniform = abs((box["target_w"] / box["w"]) / (box["target_h"] / box["h"]) - 1) <= 0.01
    report["gate6_frame_preserved"] = {
        "input_size": list(final_size),
        "expected_size": list(EXPECTED_FRAME),
        "output_resize_to_target_size": CROP_DEFAULTS["output_resize_to_target_size"],
        "crop": box,
        "crop_scale_uniform": uniform,
        "stitch_restores_canvas": workflow["10"]["inputs"]["stitcher"] == ["4", 0],
    }
    if final_size != EXPECTED_FRAME:
        errors.append(f"gate6: input size {final_size} != {EXPECTED_FRAME}")
    if not CROP_DEFAULTS["output_resize_to_target_size"]:
        errors.append("gate6: output_resize_to_target_size is off; the stitch could not restore the canvas")
    if not uniform:
        errors.append("gate6: the crop would be rescaled non-uniformly")

    # -- gate 7: node table + checkpoint compatibility -------------------------
    wf_errors = validate_workflow(workflow, object_info)
    ckpt_notes = validate_checkpoint(object_info)
    report["gate7_runtime_compatibility"] = {
        "validate_workflow_errors": wf_errors,
        "checkpoint_notes": ckpt_notes,
        "object_info": str(object_info and "loaded" or "missing"),
    }
    errors.extend(f"gate7: {e}" for e in wf_errors)
    errors.extend(f"gate7: {n}" for n in ckpt_notes if n.startswith("ERROR"))

    # -- gate 8: the keep-original vs remove-residual conflict -----------------
    guarded = int(stats.get("guarded_pale_px", 0))
    report["gate8_keep_vs_remove_conflict"] = {
        "guarded_pale_px": guarded,
        "guarded_pale_bbox": stats.get("guarded_pale_bbox"),
        "reason": stats.get("guarded_pale_note"),
        "unresolvable": guarded > 0,
        "resolution": (
            "the mask is already conservative and excludes this block; a single pass "
            "repairs only the released strip, the central block under the jaw stays as "
            "rendered and needs an operator decision"
        ),
    }

    report["errors"] = errors
    return errors, report


def _post(client: httpx.Client, path: str, key: str, payload: dict) -> dict:
    response = client.post(path, json=payload, headers={"Authorization": f"Bearer {key}"})
    response.raise_for_status()
    body = response.json()
    if body.get("code") not in (0, None):
        raise SystemExit(f"{path} -> code={body.get('code')} msg={body.get('msg')}")
    return body


def upload(client: httpx.Client, key: str, path: Path) -> str:
    with path.open("rb") as handle:
        response = client.post(
            "/task/openapi/upload",
            data={"apiKey": key},
            files={"file": (path.name, handle, "application/octet-stream")},
        )
    response.raise_for_status()
    body = response.json()
    data = body.get("data") or {}
    name = data.get("fileName") or data.get("filename")
    if body.get("code") not in (0, None) or not name:
        raise SystemExit(f"upload {path.name} failed: code={body.get('code')}")
    print(f"  uploaded {path.name} ({path.stat().st_size} B) -> {name}")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", type=Path, default=BASELINE_FINAL)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--no-edit", type=Path, default=DEFAULT_NO_EDIT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/p29-collar-repair")
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".runninghub_key")
    parser.add_argument("--object-info", type=Path,
                        default=REPO_ROOT / "evidence/p29-live-run/object_info_live.json")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--instance-type", default="default")
    parser.add_argument("--workflow-id", default="1")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="build + validate + gate only (default)")
    parser.add_argument("--upload", action="store_true", help="upload the two input files (free)")
    parser.add_argument("--submit", action="store_true", help="create the one paid task")
    parser.add_argument("--i-understand-this-costs-money", action="store_true")
    parser.add_argument("--accept-guarded-residual", action="store_true",
                        help="proceed even though gate 8 finds an unremovable central pale block")
    args = parser.parse_args(argv)

    for name, path in (("final", args.final), ("mask", args.mask), ("no-edit", args.no_edit),
                       ("stats", args.stats), ("labels", args.labels)):
        if not path.is_file():
            raise SystemExit(f"{name} input is missing: {path}")

    object_info = None
    if args.object_info.is_file():
        object_info = json.loads(args.object_info.read_text())
        print(f"validating against {args.object_info.name} ({len(object_info)} installed nodes)")
    else:
        print(f"WARNING: {args.object_info} not found; structural checks only", file=sys.stderr)

    workflow = build_workflow("__FINAL__", "__MASK__", args.seed)
    errors, gate = collect_gate_checks(
        workflow, object_info,
        final_path=args.final, mask_path=args.mask, no_edit_path=args.no_edit,
        stats_path=args.stats, labels_path=args.labels,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    gate["accept_guarded_residual"] = bool(args.accept_guarded_residual)
    conflict = gate["gate8_keep_vs_remove_conflict"]
    gate["gate_passed"] = not errors
    gate["halted_by_gate8"] = bool(conflict["unresolvable"] and not args.accept_guarded_residual)
    (args.out / "gate_report.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2))

    print(f"gate2 single pass: {gate['gate2_single_local_pass']}")
    print(f"gate3 raw mask bbox {gate['gate3_mask_is_collar_only']['raw_mask_bbox']} "
          f"in window {gate['gate3_mask_is_collar_only']['release_window_bbox']} "
          f"-> {gate['gate3_mask_is_collar_only']['raw_mask_bbox_in_window']} "
          f"(painted {gate['gate3_mask_is_collar_only']['painted_bbox']})")
    print(f"gate4 raw mask over guards: {gate['gate4_painted_disjoint_from_no_edit']['raw_mask_over_guards_px']} px, "
          f"feather rim {gate['gate4_painted_disjoint_from_no_edit']['feather_rim_px']} px "
          f"<= {gate['gate4_painted_disjoint_from_no_edit']['feather_rim_max_distance_px']} px, "
          f"into core {gate['gate4_painted_disjoint_from_no_edit']['painted_into_core_px']}")
    for label in ("hair", "olive_blouse", "background"):
        g = gate["gate5_no_correct_content_edited"][label]
        print(f"gate5 {label}: raw {g['raw_mask_intersection_px']} px "
              f"(depth {g['raw_mask_max_depth_px']}), core {g['painted_into_core_px']}")
    print(f"gate6 frame {gate['gate6_frame_preserved']['input_size']} uniform={gate['gate6_frame_preserved']['crop_scale_uniform']}")
    for note in gate["gate7_runtime_compatibility"]["checkpoint_notes"]:
        print(f"  checkpoint: {note}")
    print(f"gate8 guarded pale block: {conflict['guarded_pale_px']} px at {conflict['guarded_pale_bbox']} "
          f"(unresolvable={conflict['unresolvable']})")

    if errors:
        for err in errors:
            print(f"  GATE FAILED  {err}", file=sys.stderr)
        print(f"gate report -> {args.out / 'gate_report.json'}", file=sys.stderr)
        return 1
    if gate["halted_by_gate8"]:
        print("HALT: gate 8 found a keep-vs-remove conflict that one pass cannot resolve "
              "(the central pale block sits inside the subject's own skin guard). "
              "The mask excludes it; nothing in this workflow edits it. "
              "Re-run with --accept-guarded-residual to build the workflow for the released strip only, "
              "or ask the operator how to treat the guarded block.", file=sys.stderr)
        print(f"gate report -> {args.out / 'gate_report.json'}", file=sys.stderr)
        return 2
    print(f"gate passed -> {args.out / 'gate_report.json'}")

    if not (args.upload or args.submit):
        (args.out / "workflow.json").write_text(json.dumps(workflow, indent=2))
        print(f"dry run complete -> {args.out / 'workflow.json'}")
        return 0

    key = args.key_file.read_text().strip()
    if len(key) < 16:
        raise SystemExit(f"credential file {args.key_file} does not look like an API key")
    client = httpx.Client(base_url="https://www.runninghub.cn", timeout=60.0)
    print("uploading inputs")
    final_name = upload(client, key, args.final)
    mask_name = upload(client, key, args.mask)
    workflow = build_workflow(final_name, mask_name, args.seed)
    wf_errors = validate_workflow(workflow, object_info)
    if wf_errors:
        for err in wf_errors:
            print(f"  INVALID  {err}", file=sys.stderr)
        return 1
    (args.out / "workflow.json").write_text(json.dumps(workflow, indent=2))
    print(f"workflow written -> {args.out / 'workflow.json'}")

    if not args.submit:
        print("upload stage complete; no task created")
        return 0
    if not args.i_understand_this_costs_money:
        raise SystemExit("--submit needs --i-understand-this-costs-money (this creates one billed task)")

    payload = {
        "apiKey": key,
        "workflowId": args.workflow_id,
        "workflow": json.dumps(workflow),
        "nodeInfoList": [],
        "instanceType": args.instance_type,
    }
    created = _post(client, "/task/openapi/create", key, payload)
    task_id = created["data"]["taskId"]
    print(f"task created: {task_id} status={created['data'].get('taskStatus')}")
    (args.out / "task.json").write_text(json.dumps({"taskId": task_id}, indent=2))
    print("poll and download are intentionally left to a follow-up so no result is fetched without review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
