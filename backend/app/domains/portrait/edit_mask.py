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

Two mask builders live here
---------------------------
:func:`build_edit_masks` (P2.6) is the original *geometric* builder. It needs
nothing but MediaPipe landmarks, and its hair/clothing regions are an ellipse
around the head and a rectangle under the chin -- documented stand-ins, not
measurements. It is kept for compatibility and for the offline review that uses
it, and it is never silently substituted for a failed semantic run.

:func:`build_semantic_edit_masks` (P2.7) is the real one. It takes the ATR
18-class label map produced by :mod:`app.domains.portrait.human_parsing`
(pretrained SCHP, ONNX, CPU) and builds the regions out of *parsed* hair,
clothing and face instead of guessed geometry. It is the path the pipeline uses;
if the parser is unavailable it raises rather than falling back to the ellipse.

What is real and what is a stand-in
-----------------------------------
* **Identity is real geometry.** The eyes, brows, nose and lips come from the
  pretrained MediaPipe face mesh, and the mouth hull covers the teeth of an open
  smile, so no generator can rewrite the expression.
* **Hair, clothing and the face/neck are parsed, not guessed.** The ATR model
  labels them per pixel, which is what replaces the P2.6 ellipse: the hair mask
  follows this subject's actual silhouette, and a new hairstyle may grow a
  bounded margin *outside* it instead of being pinned inside it.
* **The parsed face island is protected whole.** ATR has no neck class, so the
  neck comes back as ``Face``; the guard is therefore the full parsed face+neck
  island unioned with the MediaPipe oval, not a skin-tone-filtered subset. A
  shadowed neck is skin that fails a skin-tone test, and a throat that became
  editable would be repainted by a garment swap. The subject's own skin tone is
  still sampled, but now only as a quality signal on the parse
  (``faceSkinShare``) instead of as a subtraction.
* **Speckle parses are dropped, not shipped.** The net sometimes tags a sliver
  of background at the frame edge as hair or as a garment. Grown by a rim, that
  sliver opens a patch of background to the generator, so islands below a
  fraction of the largest one are removed before the regions are grown, and the
  removal is reported in ``stats`` and ``gaps``.
* **Glasses are geometry, not classification.** ATR only has a ``Sunglasses``
  class; clear prescription glasses come back as ``Face``/``Hair``. The lens
  band is therefore guarded from the MediaPipe eye rings and face sides.
* **The parsing boundary is soft.** ATR is a 512x512 network upsampled to the
  photo, so the hairline and the collar are approximate to a few pixels; that
  residual is reported in ``gaps`` for human correction, not hidden.

Nothing here generates pixels, trains anything or calls a paid provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.domains.portrait.face_lock import (
    CHIN,
    EYE_A,
    EYE_B,
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
from app.domains.portrait.human_parsing import (
    ATR_CLASSES,
    CLASS_COUNT,
    CLOTHING_CLASSES,
    FACE_CLASSES,
    HAIR,
)

#: Photo indices whose bounding box defines the face rectangle handed to humans.
FACE_BOX_SOURCES = FACE_OVAL

#: Landmarks that bracket a pair of spectacles: both eye rings, the nose bridge
#: and the face sides at eye height. ATR only knows ``Sunglasses``, and clear
#: prescription lenses come back as ``Face``/``Hair``, so the lens band has to
#: come from the face mesh instead of from the classifier.
GLASSES_BAND = (
    EYE_A + EYE_B + (168, 6, 197, 195, 5, 4, 127, 356, 234, 454, 93, 323)
)


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
    #: Pixels a generator may repaint. On the geometric path these are the hair
    #: and clothing bands; on the parsed path, the real hair and garment regions.
    editable: Any
    face_width_px: float
    feather_px: float
    stats: dict[str, Any] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()
    #: Semantic regions, present only on the parsed (P2.7) path. They are
    #: ``None`` on the geometric (P2.6) path, where hair and clothing are bands
    #: rather than parsed regions, so a caller can always tell the two apart.
    hair: Any = None
    clothing: Any = None
    background_preserve: Any = None
    label_map: Any = None
    #: The parsed face/neck island *before* the skin and identity guards are
    #: unioned in: what the parser said, kept for review and for the gap report.
    parsed_face: Any = None

    @property
    def semantic(self) -> bool:
        """True when this set came from a parsed label map, not from geometry."""
        return self.label_map is not None

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


@dataclass(frozen=True)
class SemanticMaskConfig:
    """Geometry knobs for the parsed (ATR) mask path.

    Every ratio is relative to the detected face width, exactly like
    :class:`EditMaskConfig`, so the two paths can be compared knob for knob.
    """

    #: Grow the identity guard so a feathered composite cannot eat an eyelash.
    feature_grow_ratio: float = 0.030
    #: Grow the protected face/neck island so its boundary sits just outside
    #: the jaw.
    face_grow_ratio: float = 0.020
    #: Spectacle guard, grown from the eye-ring hull.
    glasses_grow_ratio: float = 0.020
    #: How far a *new* hairstyle may reach outside the parsed hair. This is the
    #: "not locked inside the old hair" allowance: a bounded rim around the real
    #: silhouette, not a head-sized ellipse.
    hair_grow_ratio: float = 0.12
    #: How far a *new* garment may reach outside the parsed one, so a collar or
    #: a shoulder line can move without opening the background or the arms.
    clothing_grow_ratio: float = 0.06
    #: Soft transition allowed where an editable region meets a kept region.
    feather_ratio: float = 0.020
    #: Below this a "region" is speckle; it is dropped and reported instead.
    min_region_px: int = 256
    #: A parsed island smaller than this share of the largest island is speckle
    #: (a background sliver mislabelled as hair or garment) and is dropped.
    component_floor_ratio: float = 0.05


def build_semantic_edit_masks(
    image_rgb: Any,
    landmarks: Any,
    label_map: Any,
    *,
    config: SemanticMaskConfig | None = None,
    face_lock_config: FaceLockConfig | None = None,
) -> EditMaskSet:
    """Mask one face in one photo from a *parsed* ATR human-parsing label map.

    ``label_map`` is the ``uint8`` class map :class:`~app.domains.portrait.
    human_parsing.SchpOnnxSegmenter` returns for the *same* photo, at the same
    HxW. Hair, clothing and the face/neck island are taken from the parse; the
    identity features and the glasses band come from the MediaPipe face mesh;
    the subject's own skin tone separates face skin from a fringe.

    The result is disjoint by construction: ``protected`` and ``editable`` are
    built so that no pixel is in both, which is what lets a downstream
    inpainting pipeline trust ``preserve`` as its keep-mask.
    """
    cfg = config or SemanticMaskConfig()
    lock_cfg = face_lock_config or FaceLockConfig()

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("build_semantic_edit_masks expects an HxWx3 RGB image")
    height, width = int(image.shape[0]), int(image.shape[1])
    shape = (height, width)

    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError("label_map must be a 2-D class map")
    if labels.shape != shape:
        raise ValueError(
            f"label_map must be in the photo's own coordinates: got "
            f"{labels.shape}, expected {shape}"
        )
    if labels.dtype.kind not in "iu":
        raise ValueError("label_map must hold integer ATR class ids")
    top = int(labels.max()) if labels.size else 0
    if top >= CLASS_COUNT:
        raise ValueError(
            f"label_map holds class id {top}, outside the {CLASS_COUNT}-class "
            "ATR taxonomy"
        )

    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[1] != 2:
        raise ValueError("landmarks must be an (N, 2) array of pixel coordinates")
    face_w = face_width(lm)
    if not np.isfinite(face_w) or face_w <= 0:
        raise ValueError("degenerate landmarks: face width is not positive")

    oval_pts = lm[list(FACE_OVAL)]
    x0, x1 = float(oval_pts[:, 0].min()), float(oval_pts[:, 0].max())
    y0, y1 = float(oval_pts[:, 1].min()), float(oval_pts[:, 1].max())
    face_h = max(y1 - y0, 1.0)

    gaps: list[str] = []

    # 1. Identity guard: eyes, brows, nose, lips and the teeth of an open smile.
    features = fill_feature_guard(shape, lm)
    features = np.maximum(features, fill_polygon(shape, _hull(lm[list(LIPS_INNER)])))
    features = dilate_mask(features, cfg.feature_grow_ratio * face_w)

    # 2. Spectacle guard. ATR reports "Sunglasses", and clear prescription
    #    glasses are parsed as Face/Hair, so this band is the only thing that
    #    keeps a lens and its wire from being repainted with the hairstyle.
    glasses = fill_polygon(shape, _hull(lm[list(GLASSES_BAND)]))
    glasses = dilate_mask(glasses, cfg.glasses_grow_ratio * face_w)

    # 3. Parsed face/neck island. ATR has no neck class, so the throat arrives
    #    as Face; the island is protected whole and grown a little, so a parse
    #    that stops a pixel short of the jaw cannot leave a rim of cheek
    #    editable, and a garment swap cannot repaint the throat. The face oval
    #    is unioned in so a sloppy parse edge cannot leave a sliver of cheek
    #    unprotected.
    parsed_face = _class_mask(labels, FACE_CLASSES)
    oval = fill_polygon(shape, _hull(lm[list(FACE_OVAL)]))
    island = np.maximum(parsed_face, oval)
    protected_face = dilate_mask(island, cfg.face_grow_ratio * face_w)

    # The subject's own skin tone is still sampled, but only as a quality
    # signal on the parse (faceSkinShare); it is not allowed to punch holes in
    # the guard, because the shadowed neck is skin that fails the test.
    reference, sample_px = skin_reference_colour(image, lm, face_w)
    if reference is None:
        face_skin = island
        gaps.append(
            "skin_reference_unavailable: the face was too small or too stylised "
            "to sample its own skin tone, so the parsed face island was "
            "protected without a skin-tone cross-check"
        )
    else:
        face_skin = cv2.bitwise_and(
            island, adaptive_skin_mask(image, reference, face_w, lock_cfg)
        )
    island_pixels = max(int((island > 0).sum()), 1)
    face_skin_pixels = int((face_skin > 0).sum())

    protected = np.maximum(np.maximum(features, glasses), protected_face)

    # 4. Hair: the parsed silhouette with its speckle dropped, grown by a
    #    bounded rim so a longer, fuller or differently shaped hairstyle is
    #    allowed, then cut back by the face guard so the forehead and jaw
    #    cannot be repainted.
    parsed_hair = _class_mask(labels, (HAIR,))
    parsed_hair, hair_cleanup = _drop_speckle(parsed_hair, cfg)
    hair = cv2.bitwise_and(
        dilate_mask(parsed_hair, cfg.hair_grow_ratio * face_w),
        cv2.bitwise_not(protected),
    )

    # 5. Clothing: the parsed garment(s) with their speckle dropped, grown a
    #    little for a new collar and shoulder line, minus everything protected
    #    (so skin, the neck and the face survive a garment swap).
    parsed_clothing = _class_mask(labels, CLOTHING_CLASSES)
    parsed_clothing, clothing_cleanup = _drop_speckle(parsed_clothing, cfg)
    clothing = cv2.bitwise_and(
        dilate_mask(parsed_clothing, cfg.clothing_grow_ratio * face_w),
        cv2.bitwise_not(protected),
    )

    # 6. Drop speckle regions instead of shipping a mask that looks broken.
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

    background = _class_mask(labels, (0,))
    background_preserve = cv2.bitwise_and(background, cv2.bitwise_not(editable))

    editable_pixels = int((editable > 0).sum())
    overlap = int(cv2.bitwise_and(protected, editable).sum())
    background_editable = int(
        (cv2.bitwise_and(background, editable) > 0).sum()
    )

    for name, cleanup in (("hair", hair_cleanup), ("garment", clothing_cleanup)):
        if cleanup["droppedIslands"]:
            gaps.append(
                f"speckle_{name}_islands_dropped: the parser tagged "
                f"{cleanup['droppedIslands']} {name} island(s) below "
                f"{cleanup['floorPixels']} px (background slivers at the frame "
                f"edge, {cleanup['droppedPixels']} px in total); they were "
                "removed before the region was grown"
            )

    parsed_share = _class_shares(labels)
    if parsed_share.get(ATR_CLASSES[3], 0.0) <= 0.0:
        gaps.append(
            "glasses_guard_is_geometric: the parser reported no Sunglasses pixels "
            "(clear prescription glasses are not in the ATR taxonomy), so the "
            "lens band comes from the MediaPipe eye rings and is reported as "
            "geometry rather than as a classified region"
        )
    if len(CLOTHING_CLASSES) > 1 and parsed_share.get(ATR_CLASSES[4], 0.0) <= 0.0:
        gaps.append(
            "no_upper_clothing_parsed: the parser found no Upper-clothes region, "
            "so the clothing mask is empty and no garment may be restyled"
        )
    gaps.append(
        "parsing_boundary_is_approximate: the ATR network runs at 512x512 and is "
        "upsampled to the photo, so the hairline and the collar are accurate to "
        "roughly a few pixels -- they still need human correction before a paid "
        "generation is attempted"
    )
    gaps.append(
        "hairline_is_not_a_hair_matte: the hair mask is the parsed hair silhouette "
        "grown by a rim, not a per-strand matte; wispy strands against the "
        "background may fall outside it"
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
        "editablePixels": editable_pixels,
        "protectedShare": round(float((protected > 0).mean()), 4),
        "editableShare": round(float((editable > 0).mean()), 4),
        "identityGuardPixels": int((features > 0).sum()),
        "glassesGuardPixels": int((glasses > 0).sum()),
        "faceSkinPixels": face_skin_pixels,
        "parsedFacePixels": int((parsed_face > 0).sum()),
        "parsedFaceIslandPixels": island_pixels,
        "faceSkinShare": round(face_skin_pixels / island_pixels, 4),
        "backgroundPreservePixels": int((background_preserve > 0).sum()),
        "backgroundEditablePixels": background_editable,
        "backgroundEditableShare": round(
            float(background_editable) / float(max(editable_pixels, 1)), 4
        ),
        "hairIslandsDropped": hair_cleanup["droppedIslands"],
        "clothingIslandsDropped": clothing_cleanup["droppedIslands"],
        "specklePixelsDropped": int(
            hair_cleanup["droppedPixels"] + clothing_cleanup["droppedPixels"]
        ),
        "protectedCoversEditable": bool(overlap == 0),
        "identityCoversEditable": bool(
            cv2.bitwise_and(features, editable).sum() == 0
        ),
        "hairContactWithProtectedPixels": int(
            cv2.bitwise_and(dilate_mask(hair, 1.0), protected).sum() // 255
        ),
        "parsedClassShares": parsed_share,
        "semantic": True,
    }
    stats.update(_legacy_ellipse_comparison(hair, lm, shape))

    return EditMaskSet(
        protected=protected,
        editable=editable,
        face_width_px=float(face_w),
        feather_px=float(feather_px),
        stats=stats,
        gaps=tuple(gaps),
        hair=hair,
        clothing=clothing,
        background_preserve=background_preserve,
        label_map=labels.astype(np.uint8, copy=False),
        parsed_face=parsed_face,
    )


def _class_mask(labels: Any, classes: tuple[int, ...]) -> Any:
    """Binary mask of the pixels whose parsed class is one of *classes*."""
    mask = np.zeros(labels.shape, np.uint8)
    for class_id in classes:
        mask[labels == class_id] = 255
    return mask


def _drop_speckle(
    mask: Any, cfg: SemanticMaskConfig
) -> tuple[Any, dict[str, int]]:
    """Keep the significant islands of one parsed class mask, drop the speckle.

    ATR occasionally tags a sliver of background at the frame edge as hair or
    as a garment. Grown by a rim it would open a patch of background to the
    generator, so an island smaller than ``cfg.component_floor_ratio`` of the
    largest island (and smaller than ``cfg.min_region_px`` in absolute terms)
    is removed. A single subject always has one dominant island, so this cannot
    delete a real limb; what it removes is reported to the caller.
    """
    blank = {"droppedIslands": 0, "droppedPixels": 0, "floorPixels": 0}
    if not mask.any():
        return mask, blank
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 2:  # background + a single island: nothing to prune
        return mask, {**blank, "floorPixels": int(stats[1, cv2.CC_STAT_AREA])}

    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = max(int(cfg.min_region_px), int(round(cfg.component_floor_ratio * int(areas.max()))))
    kept = np.zeros_like(mask)
    dropped_islands = 0
    dropped_pixels = 0
    for index, area in enumerate(areas, start=1):
        if int(area) >= floor:
            kept[components == index] = 255
        else:
            dropped_islands += 1
            dropped_pixels += int(area)
    return kept, {
        "droppedIslands": dropped_islands,
        "droppedPixels": dropped_pixels,
        "floorPixels": floor,
    }


def _class_shares(labels: Any) -> dict[str, float]:
    """Named share of the frame per ATR class that actually occurs."""
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    total = float(max(labels.size, 1))
    shares: dict[str, float] = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        if 0 <= int(value) < CLASS_COUNT:
            shares[ATR_CLASSES[int(value)]] = round(float(count) / total, 4)
    return shares


def _legacy_ellipse_comparison(
    hair: Any, landmarks: Any, shape: tuple[int, int]
) -> dict[str, float]:
    """Measure the parsed hair against the P2.6 head ellipse, for the report.

    "The hair region is no longer a big circle" is a claim, so it is quantified:
    the circle the P2.6 builder would have drawn is reconstructed from the same
    landmarks and its overlap with the parsed hair is reported.
    """
    legacy = EditMaskConfig()
    oval_pts = np.asarray(landmarks)[list(FACE_OVAL)]
    x0, x1 = float(oval_pts[:, 0].min()), float(oval_pts[:, 0].max())
    y0, y1 = float(oval_pts[:, 1].min()), float(oval_pts[:, 1].max())
    face_h = max(y1 - y0, 1.0)
    face_w = face_width(landmarks)
    ellipse = _ellipse_mask(
        shape,
        center=(0.5 * (x0 + x1), y0 + 0.42 * face_h),
        axes=(
            0.5 * (x1 - x0) + legacy.hair_side_ratio * face_w,
            0.55 * face_h + legacy.hair_up_ratio * face_w,
        ),
    )
    hair_bin = hair > 0
    ellipse_bin = ellipse > 0
    intersection = int(np.logical_and(hair_bin, ellipse_bin).sum())
    union = int(np.logical_or(hair_bin, ellipse_bin).sum())
    outside = int(np.logical_and(hair_bin, np.logical_not(ellipse_bin)).sum())
    hair_px = max(int(hair_bin.sum()), 1)
    return {
        "legacyEllipseStyle": "P2.6 head ellipse reconstructed from the same landmarks",
        "legacyEllipsePixels": int(ellipse_bin.sum()),
        "hairIoUWithLegacyEllipse": round(intersection / max(union, 1), 4),
        "hairOutsideLegacyEllipseShare": round(outside / hair_px, 4),
    }


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
