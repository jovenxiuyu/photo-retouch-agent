"""Local edit masks that protect the face and open only hair and clothing.

The styling model repaints the *whole* person -- a prompt cannot keep a smile --
so the pipeline needs a mask expressed in the original photo's own coordinates
that says which pixels a generator may touch and which must come back unchanged.

The mask semantics are the ones every mature inpainting stack uses (diffusers
``AutoPipelineForInpainting``, BrushNet, the hosted fal.ai endpoints):

    white (255) = repaint          black (0) = keep the original pixels

``editable`` is therefore the generator input, and :meth:`EditMaskSet.preserve`
is its exact complement: identity plus every pixel the module did not select.
"Unselected stays original" is what that complement means in practice.

What is real and what is a stand-in
-----------------------------------
* **Identity is real geometry.** The eyes, brows, nose and lips come from the
  pretrained MediaPipe face mesh, and the mouth hull covers the teeth of an open
  smile, so no generator can rewrite the expression.
* **The face is protected from the subject's own skin tone**, the same adaptive
  test the face lock uses, so a fringe that happens to overlap the face oval
  stays editable instead of being frozen into the hairstyle.
* **The hair band and the clothing band are geometric heuristics.** This
  environment has no hair matte and no human-parsing model, so they are an
  ellipse around the head and a rectangle under the chin -- NOT segmentation
  results. They are good enough to review a mask and to prototype a masked edit;
  they are not good enough to promise a clean hairline or a clothing cut-out.
  The gaps are reported on the result instead of being hidden.

Nothing here generates pixels, trains anything or calls a paid provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.domains.portrait.face_lock import (
    CHIN,
    FACE_OVAL,
    LIPS_INNER,
    FaceLockConfig,
    adaptive_skin_mask,
    dilate_mask,
    face_width,
    fill_feature_guard,
    fill_polygon,
    gaussian_mask,
    skin_reference_colour,
)

#: Photo indices whose bounding box defines the face rectangle handed to humans.
FACE_BOX_SOURCES = FACE_OVAL


@dataclass(frozen=True)
class EditMaskConfig:
    """Geometry knobs. All ratios are relative to the detected face width."""

    #: Grow the identity guard so a feathered composite cannot eat an eyelash.
    feature_grow_ratio: float = 0.030
    #: Grow the protected face so its boundary sits just outside the jaw/cheek.
    face_grow_ratio: float = 0.020
    #: Head ellipse: how far the hair band reaches past the face, sideways/up.
    hair_side_ratio: float = 0.85
    hair_up_ratio: float = 0.55
    #: Clothing band: gap below the chin, and how wide past the face it runs.
    clothing_gap_ratio: float = 0.10
    clothing_side_ratio: float = 1.60
    #: Soft transition allowed where an editable region meets a kept region.
    feather_ratio: float = 0.020
    #: Below this a "region" is speckle; it is dropped and reported instead.
    min_region_px: int = 256


@dataclass
class EditMaskSet:
    """The three masks a masked edit needs, plus what a human must verify."""

    #: Identity and face pixels that must be the original photograph.
    protected: Any
    #: Pixels a generator may repaint (hair band, clothing band).
    editable: Any
    face_width_px: float
    feather_px: float
    stats: dict[str, Any] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()

    @property
    def preserve(self) -> Any:
        """Complement of ``editable``: identity plus everything unselected."""
        return cv2.bitwise_not(self.editable)

    def editable_weight(self) -> Any:
        """Soft 0-1 editable weight with the boundary feather applied."""
        return gaussian_mask(self.editable, self.feather_px)

    def protected_weight(self) -> Any:
        """Soft 0-1 protected weight with the boundary feather applied."""
        return gaussian_mask(self.protected, self.feather_px)


Graph = dict[str, Any]


def build_edit_masks(
    image_rgb: Any,
    landmarks: Any,
    *,
    config: EditMaskConfig | None = None,
    face_lock_config: FaceLockConfig | None = None,
) -> EditMaskSet:
    """Mask one face in one photo. Pure geometry -- no model call, no download."""
    cfg = config or EditMaskConfig()
    lock_cfg = face_lock_config or FaceLockConfig()

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("build_edit_masks expects an HxWx3 RGB image")
    height, width = int(image.shape[0]), int(image.shape[1])
    shape = (height, width)

    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[1] != 2:
        raise ValueError("landmarks must be an (N, 2) array of pixel coordinates")
    face_w = face_width(lm)
    if not np.isfinite(face_w) or face_w <= 0:
        raise ValueError("degenerate landmarks: face width is not positive")

    gaps: list[str] = []

    # 1. Identity guard. Teeth of an open smile live inside the convex hull of
    #    the outer lip ring, so protecting that hull protects the expression.
    features = fill_feature_guard(shape, lm)
    features = np.maximum(features, fill_polygon(shape, _hull(lm[list(LIPS_INNER)])))
    features = dilate_mask(features, cfg.feature_grow_ratio * face_w)

    # 2. The face itself, judged by the subject's own skin tone. A dark fringe
    #    over the forehead is not skin, so it stays in the editable hair band.
    oval = fill_polygon(shape, _hull(lm[list(FACE_OVAL)]))
    reference, sample_px = skin_reference_colour(image, lm, face_w)
    if reference is None:
        face_skin = oval
        gaps.append(
            "skin_reference_unavailable: the face was too small or too stylised to "
            "sample its own skin tone, so the whole face oval is protected and no "
            "fringe overlapping it can be restyled"
        )
    else:
        face_skin = cv2.bitwise_and(
            oval, adaptive_skin_mask(image, reference, face_w, lock_cfg)
        )
        # A thin occluder crossing the skin (glasses wire, lip line, specular
        # highlight) must not punch a hole that a generator could paint through.
        face_skin = dilate_mask(face_skin, cfg.face_grow_ratio * face_w)

    protected = np.maximum(features, face_skin)

    # 3. Hair band: an ellipse around the head minus everything protected. The
    #    part of the ellipse that lands on the face disappears with the
    #    subtraction, which is exactly the "hair may change, face may not" rule.
    oval_pts = lm[list(FACE_OVAL)]
    x0, x1 = float(oval_pts[:, 0].min()), float(oval_pts[:, 0].max())
    y0, y1 = float(oval_pts[:, 1].min()), float(oval_pts[:, 1].max())
    face_h = max(y1 - y0, 1.0)
    face_cx = 0.5 * (x0 + x1)
    head = _ellipse_mask(
        shape,
        center=(face_cx, y0 + 0.42 * face_h),
        axes=(
            0.5 * (x1 - x0) + cfg.hair_side_ratio * face_w,
            0.55 * face_h + cfg.hair_up_ratio * face_w,
        ),
    )
    hair = cv2.bitwise_and(head, cv2.bitwise_not(protected))

    # 4. Clothing band: below the chin, out towards the shoulders. It cannot be
    #    separated from the neck, the arms or the background without a human
    #    parsing model, and that limit is reported rather than papered over.
    chin_y = float(lm[CHIN][1]) if len(lm) > CHIN else y1
    top = int(round(max(chin_y, y1) + cfg.clothing_gap_ratio * face_h))
    left = int(round(max(0.0, x0 - cfg.clothing_side_ratio * face_w)))
    right = int(round(min(float(width), x1 + cfg.clothing_side_ratio * face_w)))
    clothing = np.zeros(shape, np.uint8)
    if top < height and right > left:
        clothing[top:height, left:right] = 255
    clothing = cv2.bitwise_and(clothing, cv2.bitwise_not(protected))

    # 5. Drop speckle regions instead of shipping a mask that looks broken.
    for name, region in (("hair", hair), ("clothing", clothing)):
        pixels = int((region > 0).sum())
        if 0 < pixels < cfg.min_region_px:
            region[:] = 0
            gaps.append(
                f"{name}_region_below_floor: only {pixels} pixels survived the "
                f"face guard (< {cfg.min_region_px}); the region was dropped"
            )

    editable = np.maximum(hair, clothing)
    feather_px = cfg.feather_ratio * face_w

    gaps.append(
        "hair_band_is_geometric: no hair-segmentation model is available in this "
        "environment, so the hair region is an ellipse around the head and not a "
        "hair matte; its boundary is a heuristic, not a measured hairline"
    )
    gaps.append(
        "clothing_band_is_geometric: no human-parsing/segmentation model is "
        "available here, so the clothing rectangle also covers neck, arms and "
        "background pixels that share the band"
    )

    stats: Graph = {
        "imageWidth": width,
        "imageHeight": height,
        "faceBox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "faceWidthPx": round(face_w, 2),
        "faceHeightPx": round(face_h, 2),
        "featherPx": round(feather_px, 2),
        "skinSamplePixels": int(sample_px),
        "protectedPixels": int((protected > 0).sum()),
        "hairPixels": int((hair > 0).sum()),
        "clothingPixels": int((clothing > 0).sum()),
        "editablePixels": int((editable > 0).sum()),
        "protectedShare": round(float((protected > 0).mean()), 4),
        "editableShare": round(float((editable > 0).mean()), 4),
        "identityGuardPixels": int((features > 0).sum()),
        "faceSkinPixels": int((face_skin > 0).sum()),
        "identityCoversEditable": bool(
            cv2.bitwise_and(features, editable).sum() == 0
        ),
    }

    return EditMaskSet(
        protected=protected,
        editable=editable,
        face_width_px=float(face_w),
        feather_px=float(feather_px),
        stats=stats,
        gaps=tuple(gaps),
    )


def _hull(points: Any) -> Any:
    return cv2.convexHull(np.asarray(points, dtype=np.float32))


def _ellipse_mask(
    shape: tuple[int, int], center: tuple[float, float], axes: tuple[float, float]
) -> Any:
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (max(int(round(axes[0])), 1), max(int(round(axes[1])), 1)),
        0.0,
        0.0,
        360.0,
        255,
        thickness=-1,
    )
    return mask
