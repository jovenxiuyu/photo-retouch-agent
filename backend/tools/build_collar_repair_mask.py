"""Build the *collar repair* mask for one already-generated final render.

Every other mask in this repo is built from the ORIGINAL photo: white = the
silhouette a generator may repaint. This tool is different on purpose. It is
built for the P2.9 collar round, where the approved best render
(``evidence/p29-submit-run/final.png``, RunningHub task 2108330225116082177)
carries one local defect -- a pale knit collar strip left under the chin, where
the batch passes stopped short of the deep olive blouse -- and the brief forbids
regenerating the whole look to fix it.

So the mask is built against the **final render** and every rule below is a
*garment-release* or *identity-keep* test:

white (255) = the residual pale collar pixel a single local inpaint may repaint
black (0)   = keep the final render's own pixel

The rule is deliberately conservative, and each clause is a named piece of
evidence rather than a tuned number:

* ``final is still pale fabric``  -- bright, mid-chroma, so it is neither the
  dark olive blouse (already correct) nor the throat's high-chroma skin.
* ``not the olive blouse``        -- the final pixel's own hue/saturation; an
  olive-hued pixel is the pass's correct output and must not be touched.
* ``the original was pale knit``  -- the pixel had garment, not skin, evidence
  *before* generation; ``SCHP`` filed this band as ``Face`` (a high collar that
  reaches the jaw), so the class map alone would freeze it.
* ``the original was not skin``   -- the subject's own skin tone, sampled from
  her face, kept out of the release.
* ``outside the identity guard``  -- face, teeth, glasses, chin and every
  guarded pixel stay black, and the pixels where a release would collide with
  that guard are counted and reported instead of being edited.

It never touches hair, background, or an olive-hued pixel. If the release and
the guard cannot both be satisfied the collision is reported; the tool never
silently widens the mask to win.

Two conflicts are reported, not resolved. ``blocked_by_identity`` counts a
release that would collide with the identity guard (zero here). ``guarded_pale``
counts the pale cloth the *skin* guard holds back: a shadowed fold under the
chin that the skin model reads as skin, so "keep the original pixel" and
"remove the residual" cannot both be satisfied. Those pixels stay black so the
operator can see the size and position of the region the tool refuses to decide.

Usage, from ``backend/``::

    ../.venv/bin/python -m tools.build_collar_repair_mask \
        --final ../evidence/p29-submit-run/final.png \
        --original ../data/inputs/IMG_3651.jpg \
        --labels ../evidence/p29-collar-repair/labels_fresh.png \
        --out ../evidence/p29-collar-repair

Offline: no provider is called and no key is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):  # allow ``python tools/build_collar_repair_mask.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.domains.portrait.edit_mask import CHIN, build_semantic_edit_masks  # noqa: E402
from app.domains.portrait.face_lock import (  # noqa: E402
    adaptive_skin_mask,
    config_from_settings,
    face_width,
    skin_reference_colour,
)
from app.services.native_mask_inpainting import NativeMaskInpainting  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

#: "the final pixel is still pale fabric": brighter than the olive blouse's own
#: body (mean luma ~106 in the collar, ~142 in the lit body) but its chroma
#: still sits below the throat skin's (mean ~81). Measured from the two renders
#: in ``probe_fichu.json``; not a free parameter.
PALE_LUMA_MIN = 132.0
PALE_CHROMA_MIN = 25.0
PALE_CHROMA_MAX = 72.0

#: "the original pixel was pale knit": the sweater's own tone.
SOURCE_KNIT_LUMA_MIN = 140.0
SOURCE_KNIT_CHROMA_MAX = 50.0

#: "the final pixel is the olive blouse, not residual knit". The reference blouse
#: reads hue 56-87 at saturation 0.15-0.35; residual cream reads hue < 45.
OLIVE_HUE_MIN = 45.0
OLIVE_HUE_MAX = 115.0
OLIVE_SAT_MIN = 0.12

#: The repair is confined to the collar's own neighbourhood below the chin.
WINDOW_DOWN_FACE_WIDTHS = 0.62
WINDOW_LEFT_FACE_WIDTHS = 0.62
WINDOW_RIGHT_FACE_WIDTHS = 0.75

#: A release blob smaller than this is specks of knit texture, not a strip.
MIN_BLOB_PX = 400
#: Reported separately: a blob this small cannot be reviewed by eye.
MIN_REPORTED_BLOB_PX = 100


def _luma_chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = rgb.astype(np.float32)
    luma = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    chroma = a.max(2) - a.min(2)
    return luma, chroma


def _hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    d = mx - mn
    sat = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    hue = np.zeros_like(mx)
    m = d > 1e-6
    i = (mx == r) & m
    hue[i] = (60 * ((g[i] - b[i]) / d[i]) + 360) % 360
    i = (mx == g) & m
    hue[i] = 60 * ((b[i] - r[i]) / d[i]) + 120
    i = (mx == b) & m
    hue[i] = 60 * ((r[i] - g[i]) / d[i]) + 240
    return hue, np.where(m, sat, 0.0)


def _keep_blobs(mask: np.ndarray, min_px: int) -> np.ndarray:
    u8 = (mask * 255).astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros_like(u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[labels == i] = 255
    return keep > 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final", type=Path, default=REPO_ROOT / "evidence/p29-submit-run/final.png",
                   help="the approved best render (must be the frozen baseline)")
    p.add_argument("--original", type=Path, default=REPO_ROOT / "data/inputs/IMG_3651.jpg")
    p.add_argument("--labels", type=Path, default=REPO_ROOT / "evidence/p29-collar-repair/labels_fresh.png",
                   help="SCHP ATR class map of --original, in its own coordinates")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "evidence/p29-collar-repair")
    p.add_argument("--face-index", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("final", "original", "labels", "out"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    original = np.asarray(Image.open(args.original).convert("RGB"))
    final = np.asarray(Image.open(args.final).convert("RGB"))
    labels = np.asarray(Image.open(args.labels))
    height, width = labels.shape
    if original.shape[:2] != (height, width) or final.shape[:2] != (height, width):
        raise SystemExit(
            f"size mismatch: labels {labels.shape}, original {original.shape[:2]}, "
            f"final {final.shape[:2]} -- the mask must stay in the photo's own coordinates"
        )

    service = NativeMaskInpainting(settings)
    landmarks = service._detect_landmarks(args.original, original)
    mask_set = build_semantic_edit_masks(original, landmarks, labels)
    protected = np.asarray(mask_set.protected) > 0
    hair = np.asarray(mask_set.hair) > 0
    clothing = np.asarray(mask_set.clothing) > 0
    collar = np.asarray(mask_set.collar) > 0
    background = labels == 0

    face_w = face_width(landmarks)
    chin_x, chin_y = (float(v) for v in landmarks[CHIN])
    reference, _ = skin_reference_colour(original, landmarks, face_w)
    lock_cfg = config_from_settings(settings)
    skin_original = adaptive_skin_mask(original, reference, face_w, lock_cfg) > 0

    orig_luma, orig_chroma = _luma_chroma(original)
    final_luma, final_chroma = _luma_chroma(final)
    final_hue, final_sat = _hue_sat(final)

    source_knit = (orig_luma > SOURCE_KNIT_LUMA_MIN) & (orig_chroma < SOURCE_KNIT_CHROMA_MAX)
    pale_fabric = (
        (final_luma > PALE_LUMA_MIN)
        & (final_chroma > PALE_CHROMA_MIN)
        & (final_chroma < PALE_CHROMA_MAX)
    )
    olive_blouse = (final_hue >= OLIVE_HUE_MIN) & (final_hue <= OLIVE_HUE_MAX) & (final_sat > OLIVE_SAT_MIN)

    window = np.zeros((height, width), bool)
    window[
        int(chin_y) : int(chin_y + WINDOW_DOWN_FACE_WIDTHS * face_w),
        max(int(chin_x - WINDOW_LEFT_FACE_WIDTHS * face_w), 0) : min(
            int(chin_x + WINDOW_RIGHT_FACE_WIDTHS * face_w), width
        ),
    ] = True

    residual = pale_fabric & source_knit & ~skin_original & window & ~hair & ~background
    residual = residual & ~olive_blouse

    release = residual & ~protected
    collision = residual & protected

    # The *other* half of the same collar band: pale cloth in the final render
    # where the original pixel is a shadowed fold that the subject's own skin
    # model reads as skin. "Keep the original pixel" and "remove the pale
    # residual" cannot both hold there, so these pixels are never released --
    # they are counted and reported so the conflict is visible, not silently
    # resolved. Gate 8 of the collar-repair brief asks for exactly this region.
    guarded_pale = pale_fabric & window & ~background & ~hair & ~olive_blouse & skin_original

    release = _keep_blobs(release, MIN_BLOB_PX)
    # A one-pixel rim closes antialiasing holes without touching a kept pixel.
    release = cv2.dilate(release.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    release = release & ~protected & ~hair & ~background

    release_u8 = (release * 255).astype(np.uint8)
    Image.fromarray(release_u8).save(out_dir / "collar_repair_mask.png")

    # The mirror of the release: every pixel the repair pass must NOT paint --
    # the identity guard, the subject's own skin inside the collar window, parsed
    # hair, the background and every already-correct olive pixel. Saved as its own
    # map so the repair workflow's offline gate can prove the painted region and
    # this one are disjoint without re-deriving the guards from the photo.
    no_edit = protected | hair | background | olive_blouse | (skin_original & window)
    Image.fromarray((no_edit * 255).astype(np.uint8)).save(out_dir / "collar_repair_no_edit.png")

    # The keep-vs-remove conflict as its own layer: the pale residual the skin
    # guard refuses to release. Zero here would mean every residual pixel is
    # safely editable; a non-empty layer is the region an operator must decide.
    Image.fromarray((guarded_pale * 255).astype(np.uint8)).save(
        out_dir / "collar_repair_guarded_pale.png"
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(release_u8, 8)
    blobs = sorted(
        [
            {"px": int(stats[i, cv2.CC_STAT_AREA]),
             "bbox": [int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])]}
            for i in range(1, n)
        ],
        key=lambda b: -b["px"],
    )
    ys, xs = np.where(release)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if release.any() else None
    win_ys, win_xs = np.where(window)
    window_bbox = [int(win_xs.min()), int(win_ys.min()), int(win_xs.max()), int(win_ys.max())]

    guarded_ys, guarded_xs = np.where(guarded_pale)
    guarded_bbox = (
        [int(guarded_xs.min()), int(guarded_ys.min()), int(guarded_xs.max()), int(guarded_ys.max())]
        if guarded_pale.any()
        else None
    )

    collisions = _keep_blobs(collision, MIN_REPORTED_BLOB_PX)
    ny, nys = np.where(collisions)
    collision_bbox = [int(nys.min()), int(ny.min()), int(nys.max()), int(ny.max())] if collisions.any() else None

    stats_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "tools.build_collar_repair_mask",
        "baseline_final": str(args.final.relative_to(REPO_ROOT)),
        "original": str(args.original.relative_to(REPO_ROOT)),
        "labels": str(args.labels.relative_to(REPO_ROOT)),
        "size": [width, height],
        "face_width_px": round(face_w, 2),
        "chin_px": [round(chin_x, 2), round(chin_y, 2)],
        "window_px": int(window.sum()),
        "window_bbox": window_bbox,
        "no_edit_px": int(no_edit.sum()),
        "rules": {
            "pale_fabric": "final luma > 132 and 25 < chroma < 72",
            "source_knit": "original luma > 140 and chroma < 50",
            "skin": "adaptive skin mask of the subject's own face sample, on the original",
            "olive_blouse": "final hue 45-115 at saturation > 0.12 is kept",
            "guards": "protected (identity/glasses/chin/throat) and parsed hair and background are kept",
        },
        "counts": {
            "pale_fabric_px": int(pale_fabric.sum()),
            "source_knit_px": int(source_knit.sum()),
            "residual_px": int(residual.sum()),
            "release_px": int(release.sum()),
            "release_px_in_collar_class": int((release & collar).sum()),
            "release_px_in_clothing_class": int((release & clothing).sum()),
            "release_px_over_still_pale_original": int((release & source_knit).sum()),
            "release_over_hair_px": int((release & hair).sum()),
            "release_over_background_px": int((release & background).sum()),
            "release_over_protected_px": int((release & protected).sum()),
            "blocked_by_identity_px": int(collision.sum()),
            "dilated_over_protected_px": int((release & protected).sum()),
        },
        "release_bbox": bbox,
        "release_blobs": blobs[:20],
        "guarded_pale_px": int(guarded_pale.sum()),
        "guarded_pale_bbox": guarded_bbox,
        "guarded_pale_note": (
            "pale cloth in the final inside the same collar band, held back by the "
            "subject's own skin guard on the original; keep-vs-remove conflict, "
            "reported for the operator to decide"
        ),
        "blocked_by_identity_bbox": collision_bbox,
        "blocked_by_identity_blobs": sorted(
            [
                {"px": int(stats[i, cv2.CC_STAT_AREA]),
                 "bbox": [int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])]}
                for i in range(1, cv2.connectedComponentsWithStats((collisions * 255).astype(np.uint8), 8)[0])
            ],
            key=lambda b: -b["px"],
        )[:20],
    }
    (out_dir / "collar_repair_stats.json").write_text(json.dumps(stats_out, indent=1), encoding="utf-8")
    print(json.dumps(stats_out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
