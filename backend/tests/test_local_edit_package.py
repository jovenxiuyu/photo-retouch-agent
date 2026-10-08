"""P2.9: the standard local edit package -- six items, no cloud, no key.

The package is the *contract* a local generation engine reads. These tests pin
the parts of that contract a downstream engine (today: the ComfyUI
``Inpaint-CropAndStitch`` pair) relies on, and they pin the promise the package
must never break: the identity region and the edit region stay disjoint, so the
face can be restored from the original pixels at identical coordinates.

Nothing here touches a provider: the detector and the parser are injected at the
existing seam, a synthetic photo stands in for a real one, and the only file
writes go to ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.services.local_edit_package import (
    ENGINE_TARGET,
    MASK_LEVELS,
    PIPELINE_VERSION,
    LocalEditPackage,
    LocalEditPackageBuilder,
    UnknownStyleError,
    resolve_style_id,
    sha256_file,
    style_ids,
)
from tests.test_semantic_edit_mask import (
    COLLAR_COLS,
    COLLAR_ROWS,
    FACE_CX,
    FACE_CY,
    _collar_image,
    _collar_labels,
    _landmarks,
)
from tools import build_local_edit_package as cli

#: The identity features the guard must never give up.
EYE_CENTRE = (int(FACE_CX - 26.0), int(FACE_CY - 22.0))


class _StubDetector:
    """The mesh the real MediaPipe lock would hand back -- no model, no onnx."""

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        return [_landmarks()]

    def close(self) -> None:
        return None


class _StubSegmenter:
    """Serve a synthetic ATR label map with the real segmenter's interface."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = np.asarray(labels)

    def segment(self, image: np.ndarray) -> np.ndarray:
        return self._labels

    def close(self) -> None:
        return None


def _settings(tmp_path: Path, **overrides) -> Settings:
    models = tmp_path / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)
    landmark = models / "face_landmarker.task"
    landmark.write_bytes(b"stub")
    parsing = models / "schp-atr-18.onnx"
    parsing.write_bytes(b"stub")
    values = {
        "data_dir": tmp_path / "data",
        "face_landmark_model_path": landmark,
        "human_parsing_model_path": parsing,
        "face_lock_mode": "off",
    }
    values.update(overrides)
    return Settings(**values)


def _write_photo(tmp_path: Path, image: np.ndarray | None = None) -> Path:
    """A synthetic portrait whose collar the parser files as Face."""
    path = tmp_path / "portrait.png"
    Image.fromarray(_collar_image() if image is None else image).save(path)
    return path


def _builder(tmp_path: Path, settings: Settings | None = None) -> LocalEditPackageBuilder:
    settings = settings or _settings(tmp_path)
    return LocalEditPackageBuilder(
        settings,
        detector_factory=lambda path: _StubDetector(),
        segmenter_factory=lambda path: _StubSegmenter(_collar_labels()),
        git_info_fn=lambda: {"head": "test", "branch": "test", "dirty": False},
    )


@pytest.fixture()
def package(tmp_path: Path) -> LocalEditPackage:
    return _builder(tmp_path).build(
        _write_photo(tmp_path), "artistic_wave", tmp_path / "pkg"
    )


def _read_mask(package: LocalEditPackage, key: str) -> np.ndarray:
    return np.asarray(Image.open(package.files[key]).convert("L"))


# ------------------------------------------------------- 1. hair edit mask


def test_hair_mask_is_open_and_never_reaches_the_face(package: LocalEditPackage) -> None:
    hair = _read_mask(package, "hair_edit_mask")
    assert set(np.unique(hair)) <= set(MASK_LEVELS), "strictly black/white, no grey"
    assert int((hair == 255).sum()) > 0, "there is hair to restyle"

    overlap = package.manifest["overlap"]
    assert overlap["hair_and_face_core"]["pixels"] == 0
    assert overlap["hair_and_protected"]["pixels"] == 0, "hair never touches the face"


# --------------------------------------------------- 2. clothing edit mask


def test_clothing_mask_is_open_and_includes_the_old_collar(
    package: LocalEditPackage,
) -> None:
    clothing = _read_mask(package, "clothing_edit_mask")
    assert set(np.unique(clothing)) <= set(MASK_LEVELS)
    assert int((clothing == 255).sum()) > 0

    collar = package.manifest["collar"]
    assert collar["split_applied"] is True
    assert collar["pixels"] > 0
    # A collar pixel (left of the throat strip) is editable...
    assert clothing[COLLAR_ROWS[0] + 5, COLLAR_COLS[0] + 10] == 255
    # ...while the throat, in the very same band, is not.
    assert package.manifest["collar"]["neck_skin_protected_pixels"] > 0


# ------------------------------------------------ 3. protected identity mask


def test_protected_identity_mask_covers_features_teeth_and_glasses(
    package: LocalEditPackage,
) -> None:
    protected = _read_mask(package, "protected_identity_mask")
    assert set(np.unique(protected)) <= set(MASK_LEVELS)
    assert int((protected == 255).sum()) > 0
    # The eye centre is inside the guard...
    assert protected[EYE_CENTRE[1], EYE_CENTRE[0]] == 255

    guard = package.manifest["identity_guard"]
    assert guard["identity_guard_pixels"] > 0, "eyes/brows/nose/lips/teeth"
    assert guard["glasses_guard_pixels"] > 0, "the glasses band is guarded too"
    assert guard["face_core_pixels"] > 0


# ------------------------------------------- 4. the masks are separate files


def test_hair_and_clothing_are_shipped_as_independent_masks(
    package: LocalEditPackage,
) -> None:
    hair = _read_mask(package, "hair_edit_mask")
    clothing = _read_mask(package, "clothing_edit_mask")
    combined = _read_mask(package, "combined_edit_mask")

    assert package.files["hair_edit_mask"] != package.files["clothing_edit_mask"]
    assert int((hair == 255).sum()) > 0 and int((clothing == 255).sum()) > 0
    # The combined mask is exactly the union, so each pass can be run alone.
    assert np.array_equal(combined, np.maximum(hair, clothing))

    steps = package.manifest["engine_handoff"]["steps"]
    assert [step["mask"] for step in steps] == [
        "clothing_edit_mask.png",
        "hair_edit_mask.png",
    ]
    # The masks share the hair/garment junction band, so the order decides who
    # owns it: clothing first, hair last (the strands crossing the garment stay
    # hair). Each pass after the first reads the previous pass's output.
    assert steps[0]["image"] == package.files["original_image"].name
    assert steps[1]["image"] == "output of step 'clothing'"


# --------------------------------------- 5. protected and edit stay disjoint


def test_protected_identity_and_edit_masks_never_overlap(
    package: LocalEditPackage,
) -> None:
    protected = _read_mask(package, "protected_identity_mask")
    editable = _read_mask(package, "combined_edit_mask")

    assert not np.any((protected > 0) & (editable > 0))
    assert package.manifest["overlap"]["protected_and_edit"]["pixels"] == 0
    assert package.acceptance["passed"] is True


# ------------------------------------------------- 6. background is not eaten


def test_background_is_not_swallowed_by_the_edit_area(package: LocalEditPackage) -> None:
    background = package.manifest["overlap"]["background_inside_edit"]
    assert background["share_of_edit_area"] <= background["limit"]
    assert background["share_of_edit_area"] < 0.35


# ------------------------------------ 7. every manifest reference is on disk


def test_every_manifest_reference_exists_with_a_matching_digest(
    package: LocalEditPackage,
) -> None:
    for key in (
        "original_image",
        "hair_edit_mask",
        "clothing_edit_mask",
        "protected_identity_mask",
        "combined_edit_mask",
        "prompt_bundle",
    ):
        entry = package.manifest[key]
        target = package.root / entry["path"]
        assert target.is_file(), f"{key} points at a missing file"

    for key, entry in package.manifest["files"].items():
        target = package.root / entry["path"]
        assert target.is_file(), f"{key} points at a missing file"
        assert target.stat().st_size == entry["bytes"]
        assert sha256_file(target) == entry["sha256"]

    assert package.manifest["prompt_bundle"]["path"] == "prompt_bundle.json"
    assert package.manifest["source_image_size"] == [360, 480]
    assert package.manifest["pipeline_version"] == PIPELINE_VERSION
    assert package.manifest["engine_target"] == ENGINE_TARGET


# ------------------------------------------- 8. prompt bundle identity rules


def test_prompt_bundle_carries_the_identity_constraints(
    package: LocalEditPackage,
) -> None:
    bundle = package.prompt_bundle
    for key in (
        "style_name",
        "style_reason",
        "hair_prompt",
        "clothing_prompt",
        "global_prompt",
        "negative_prompt",
        "identity_constraints",
    ):
        assert key in bundle

    constraints = bundle["identity_constraints"]
    rules = {rule["id"]: rule for rule in constraints["rules"]}
    assert {
        "face_features_unchanged",
        "smile_unchanged",
        "glasses_unchanged",
        "background_unchanged",
        "pose_unchanged",
        "edit_only_inside_masks",
    } <= set(rules)
    text = json.dumps(constraints, ensure_ascii=False)
    for token in ("五官不变", "笑容不变", "眼镜不变", "背景不变", "姿势不变"):
        assert token in text
    assert constraints["prompt_text_en"].strip()
    assert constraints["prompt_text_zh"].strip()
    # The bundle is a file, and it round-trips.
    on_disk = json.loads(package.prompt_bundle_path.read_text(encoding="utf-8"))
    assert on_disk["hair_prompt"] == bundle["hair_prompt"]


# ------------------------------------------------------------- 9. the CLI


def test_cli_builds_a_complete_package_directory(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    photo = _write_photo(tmp_path)
    labels_path = tmp_path / "labels.png"
    Image.fromarray(_collar_labels()).save(labels_path)
    out = tmp_path / "package"

    def _fake_build_builder(segmenter_factory=None, config=None):
        return LocalEditPackageBuilder(
            settings,
            detector_factory=lambda path: _StubDetector(),
            segmenter_factory=segmenter_factory,
            git_info_fn=lambda: {"head": "test"},
        )

    monkeypatch.setattr(cli, "build_builder", _fake_build_builder)
    exit_code = cli.main(
        [
            "--photo",
            str(photo),
            "--style",
            "精致优雅风",
            "--labels",
            str(labels_path),
            "--out-dir",
            str(out),
        ]
    )

    assert exit_code == 0
    names = {path.name for path in out.iterdir()}
    assert {
        "original.png",
        "hair_edit_mask.png",
        "clothing_edit_mask.png",
        "protected_identity_mask.png",
        "combined_edit_mask.png",
        "preview_hair.png",
        "preview_clothing.png",
        "preview_protected_identity.png",
        "preview_overlay.png",
        "prompt_bundle.json",
        "manifest.json",
    } <= names

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mask_rules_passed"] is True
    assert manifest["prompt_bundle"]["path"] == "prompt_bundle.json"
    bundle = json.loads((out / "prompt_bundle.json").read_text(encoding="utf-8"))
    assert bundle["style_id"] == "artistic_wave"


def test_cli_lists_the_styles_it_owns(capsys) -> None:
    assert cli.main(["--list-styles"]) == 0
    printed = capsys.readouterr().out
    for style_id in style_ids():
        assert style_id in printed


# ------------------------------------------------------- 10. no API key needed


def test_the_package_is_built_with_no_api_key_or_network(
    tmp_path: Path, monkeypatch
) -> None:
    for name in (
        "FAL_KEY",
        "DASHSCOPE_API_KEY",
        "SILICONFLOW_API_KEY",
        "ARK_API_KEY",
        "DEEPSEEK_API_KEY",
        "ZHIPU_API_KEY",
        "PHOTO_AGENT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = _settings(tmp_path)
    assert not settings.fal_key
    assert not settings.siliconflow_api_key

    package = _builder(tmp_path, settings).build(
        _write_photo(tmp_path), "urban_short", tmp_path / "pkg"
    )

    assert package.manifest["mask_rules_passed"] is True
    provenance = package.manifest["provenance"]
    assert provenance["network_calls"] == 0
    assert provenance["paid_generations"] == 0


# ------------------------------------------------------------ style ids


def test_a_style_axis_alias_resolves_to_the_axis_it_names() -> None:
    assert resolve_style_id("精致优雅风") == "artistic_wave"
    assert resolve_style_id("都市利落短发") == "urban_short"
    assert resolve_style_id("先锋街头高马尾") == "bold_street"
    with pytest.raises(UnknownStyleError):
        resolve_style_id("no-such-style")


def test_the_manifest_quotes_the_photo_it_was_built_from(
    tmp_path: Path, monkeypatch
) -> None:
    photo = _write_photo(tmp_path)
    digest = hashlib.sha256(photo.read_bytes()).hexdigest()
    labels_path = tmp_path / "labels.png"
    Image.fromarray(_collar_labels()).save(labels_path)

    def _fake_build_builder(segmenter_factory=None, config=None):
        return LocalEditPackageBuilder(
            _settings(tmp_path),
            detector_factory=lambda path: _StubDetector(),
            segmenter_factory=segmenter_factory,
            git_info_fn=lambda: {"head": "test"},
        )

    monkeypatch.setattr(cli, "build_builder", _fake_build_builder)
    exit_code = cli.main(
        [
            "--photo",
            str(photo),
            "--style",
            "urban_short",
            "--labels",
            str(labels_path),
            "--out-dir",
            str(tmp_path / "p2"),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "p2" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_image"]["sha256"] == digest
    assert manifest["source_image"]["width"] == 360
    assert manifest["source_image"]["height"] == 480
