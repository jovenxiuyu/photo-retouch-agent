"""Submit ONE paid CropAndStitch local edit to RunningHub (P2.9).

Consumes a package produced by ``tools.build_local_edit_package`` and turns it
into an inline ComfyUI workflow (API format) that runs the two-pass masked edit
on RunningHub with the ``lquesada/ComfyUI-Inpaint-CropAndStitch`` nodes:

    pass 1  clothing_edit_mask  on the original image
    pass 2  hair_edit_mask      on pass 1's stitched output

Pass order is not cosmetic. CropAndStitch composites as
``new = mask * inpainted + (1 - mask) * canvas`` and each pass blends over the
image the previous pass returned, so the mask that runs last owns the
hair/clothing overlap band. Hair runs last on purpose: strands that fall over
the new garment stay hair.

Each pass is conditioned on the *original crop pixels* rather than on an empty
latent: ``InpaintCropImproved`` feeds ``cropped_image`` / ``cropped_mask`` into
``InpaintModelConditioning`` and the sampler takes its positive, negative and
latent from that node, with ``denoise`` below 1.0. The first real run used
``VAEEncodeForInpaint`` (which encodes a flat grey blob inside the mask) at
``denoise=1.0``, so the model had no original signal to preserve and painted a
second face and a second pair of glasses into the hair mask -- see
``evidence/p29-fix-run/``.

Everything is validated locally before anything is sent, because RunningHub
creates a task (and then fails it) even when the workflow references a node the
runtime does not have -- see ``evidence/p29-runninghub-live-audit.md``.

Stages, from ``backend/``::

    # build + validate only -- no network, no key needed
    ../.venv/bin/python -m tools.runninghub_crop_stitch_run \
        --package ../artifacts/local-edit-package/<stamp> --dry-run

    # upload the three input files -- free
    ../.venv/bin/python -m tools.runninghub_crop_stitch_run \
        --package ../artifacts/local-edit-package/<stamp> --upload

    # create the single billed task, poll it and download the result
    ../.venv/bin/python -m tools.runninghub_crop_stitch_run \
        --package ../artifacts/local-edit-package/<stamp> \
        --submit --i-understand-this-costs-money

The API key is read from ``~/.runninghub_key`` and is never printed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "https://www.runninghub.cn"
CREATE_PATH = "/task/openapi/create"
STATUS_PATH = "/task/openapi/status"
OUTPUTS_PATH = "/task/openapi/outputs"
# The v2 binary upload endpoint rejects this account's key with errorCode 1000;
# the classic task-scoped upload endpoint authenticates via the apiKey form
# field and returns the server-side ``fileName`` that LoadImage consumes.
UPLOAD_PATH = "/task/openapi/upload"

CHECKPOINT = "juggernautXL_juggXIByRundiffusion.safetensors"

# The style this round must realise. The masks, the identity constraints and the
# acceptance list were all built for this one plan; driving the run from any other
# style (the previously wired ``artistic_wave`` approximation in particular) would
# silently change what the picture is supposed to be.
EXPECTED_STYLE_ID = "ai_elegant_refined"

# The runtime has no dedicated inpainting checkpoint. Every ``CheckpointLoaderSimple``
# option is a plain SD1.5/SDXL base or a Flux UNet, and the only inpaint weights on
# the platform are ControlNet / LLLite patches that plug into a Qwen-Image or Flux
# graph -- splicing one of those into this SDXL CropAndStitch graph is exactly the
# incompatible-node assembly the brief forbids, so it is not attempted. What *is*
# supported for a base SDXL model is the classic masked-latent path: VAE-encode the
# crop, confine sampling with a ``noise_mask``, and stay below denoise 1.0 so the
# existing pixels still drive the sampler. ``validate_checkpoint`` states this
# verdict explicitly instead of pretending the concat-inpaint path is available.

# Mask geometry decisions (see evidence/p29-fix-run/mask_pipeline_probe.py):
#   * mask_fill_holes=False -- the node's iterative fill *added* 282 px of
#     protected chin/collar skin to the clothing mask and 2 px to the hair mask,
#     i.e. it repainted skin we promised to keep. The 283 px of speckle holes it
#     was closing are far below one latent pixel (a 1024 px crop is a 128x128
#     latent) and they fail *safe*: a hole keeps the original pixel.
#   * mask_blend_pixels=2 -- expand(2)+blur(1) reaches ~4 px past the mask edge
#     rather than ~8 px, and that ring is the only place a generated pixel can
#     touch the protected identity contour. Not 0, because a hard edge would cut
#     the hair/skin boundary the stitch has to feather.
#   * mask_expand_pixels=0 -- the SCHP masks already cover the target silhouette.
#   * context_from_mask_extend_factor=1.2 -- widens the crop the model sees, not
#     the mask it is allowed to paint.
CROP_DEFAULTS: dict[str, object] = {
    "downscale_algorithm": "bilinear",
    "upscale_algorithm": "bicubic",
    "preresize": False,
    "preresize_mode": "ensure minimum resolution",
    "preresize_min_width": 1024,
    "preresize_min_height": 1024,
    "preresize_max_width": 16384,
    "preresize_max_height": 16384,
    "mask_fill_holes": False,
    "mask_expand_pixels": 0,
    "mask_invert": False,
    "mask_blend_pixels": 2,
    "mask_hipass_filter": 0.1,
    "extend_for_outpainting": False,
    "extend_up_factor": 1.0,
    "extend_down_factor": 1.0,
    "extend_left_factor": 1.0,
    "extend_right_factor": 1.0,
    "context_from_mask_extend_factor": 1.2,
    "output_resize_to_target_size": True,
    "output_target_width": 1024,
    "output_target_height": 1024,
    "output_padding": "32",
    "device_mode": "gpu (much faster)",
}

SAMPLER = {
    "steps": 30,
    "cfg": 6.5,
    "sampler_name": "dpmpp_2m_sde",
    "scheduler": "karras",
}

# Denoise is per pass. With ``noise_mask`` set, the sampler starts the masked
# region from ``encode(original_crop) + noise * sigma(denoise)`` and blends the
# unmasked region back to the clean latent after every step, so denoise is the
# img2img strength *inside the mask only*. Because the runtime has no inpainting
# checkpoint, this strength is the only handle on "keep the person, change the
# garment": denoise=1.0 discards the original signal and was the direct cause of
# the duplicated head. The band is the range in which the model still has to
# follow the silhouette that is already there.
PASS_DENOISE = {"clothing": 0.75, "hair": 0.70}
DENOISE_BAND = (0.60, 0.80)

# The plan's Accessories clause asks for new browline glasses, but this round's
# acceptance list pins "glasses unchanged" and the glasses sit inside the
# protected identity mask, so the clause is dropped from the prompt instead of
# contradicting the identity constraints in the same conditioning.
GLASSES_CLAUSE = "replace the current round thin metal-frame glasses"

# The garment clause ends with a new necklace, which is an accessory the masks do
# not cover, so it is dropped the same way.
NECKLACE_CLAUSE = "worn with a slim metal necklace"

# ``hair_prompt`` names the hair being replaced. At denoise below 1.0 a
# description of the input biases the model toward keeping it, so only the
# target colour is kept.
CURRENT_HAIR_CLAUSE = "from the current deep black / dark brown"

# The plan clause is split at the verb that introduces the target, so the positive
# prompt is still the plan's own wording -- just without the "here is what the
# person currently looks like" preamble that made the model redraw the person.
CLOTHING_TARGET_MARKER = "Replace it with"
HAIR_TARGET_MARKER = "Change it into"
CLOTHING_TARGET_SUFFIX = (
    "draped naturally over the body, following the original pose and body shape, "
    "with natural cloth folds"
)
HAIR_TARGET_SUFFIX = (
    "real individual hair strands, natural hairline, growing naturally from the "
    "existing scalp"
)

# Positive prompts may describe only what is inside the mask. Spelled-out facial
# features are the one thing a masked edit must never ask for: the crop already
# contains a whole head as context, and "keep the facial features exactly" reads
# as an instruction to draw them. Matched on word boundaries so "chestnut brown"
# is not mistaken for "brow".
FORBIDDEN_POSITIVE_PATTERN = re.compile(
    r"\b(facial feature|facial features|eye|eyes|eyebrow|eyebrows|brow|brows|"
    r"nose|lips|lip|mouth|teeth|smile|face|glasses|person|portrait|identity)\b",
    re.IGNORECASE,
)

# English rendering of prompt_bundle.negative_prompt (which is Chinese; CLIP for
# SDXL is English-trained). The first eight terms are the artefacts the previous
# run actually produced -- a second face with its own glasses and mouth, and
# patchy hair -- and the rest carry the identity / background / seam rules. The
# negative prompt is a hint only: the mask and the original-pixel conditioning
# are what actually hold the identity.
NEGATIVE_PROMPT = (
    "duplicate face, extra eyes, extra mouth, extra glasses, double head, "
    "second person, bald patches, deformed hairline, "
    "plastic airbrushed skin, blurred face, reshaped or replaced facial features, "
    "a different person's face, obvious age drift, changing the number of people, "
    "a broken or damaged background, extra fingers or limbs, text, watermark, "
    "visible seam, hard cut-out edge"
)
REQUIRED_NEGATIVE_TERMS = (
    "duplicate face",
    "extra eyes",
    "extra mouth",
    "extra glasses",
    "double head",
    "second person",
    "bald patches",
    "deformed hairline",
)

# The plan writes each clause as "here is the current state ... <verb> <target>".
# At denoise below 1.0 the "current state" half is a description of the input,
# which biases the sampler toward keeping it -- and, worse, it is a description
# of the person, which is exactly what made the model draw one. Only the half
# after the verb is used, re-introduced with a neutral sentence frame.
TARGET_INTRO = {"clothing": "Replace the garment with ", "hair": "Replace the hairstyle with "}

# The plan's own identity rule, kept verbatim as the single mask instruction. The
# rest of ``identity_constraints.prompt_text_en`` describes facial features and is
# deliberately NOT sent: with a whole head inside the crop, "keep the facial
# features exactly" reads as an instruction to draw them.
MASK_RULE_ID = "edit_only_inside_masks"


def _plan_mask_rule(bundle: dict) -> str:
    for rule in bundle.get("identity_constraints", {}).get("rules", []):
        if rule.get("id") == MASK_RULE_ID and rule.get("en"):
            return rule["en"]
    raise SystemExit(f"prompt bundle has no {MASK_RULE_ID!r} identity rule")


def _tidy(text: str) -> str:
    """Collapse whitespace and remove the space the clause removal left before punctuation."""
    return re.sub(r"\s+", " ", re.sub(r"\s+([.,])", r"\1", text)).strip()


def _join_head_tail(head: str, tail: str) -> str:
    if head.rstrip().endswith((",", ".", ";", ":")):
        head = head.rstrip()[:-1] + " "
    return _tidy(head + tail)


def _drop_clause(text: str, clause: str) -> tuple[str, bool]:
    """Remove ``clause`` and the punctuation/space that glued it to the sentence."""
    idx = text.find(clause)
    if idx < 0:
        return text, False
    head, tail = text[:idx], text[idx + len(clause):]
    return _join_head_tail(head, tail), True


def _drop_clause_through(text: str, clause: str) -> tuple[str, bool]:
    """Remove ``clause`` *and the rest of the sentence it opens*.

    Used for plan clauses whose tail is a comma-separated descriptor list. The
    plan's glasses clause is ``replace the current round thin metal-frame glasses
    with 半框眉线眼镜 (browline / clubmaster eyeglasses): ... sitting naturally on
    the face.`` -- dropping only the lead marker would leave ``brow`` and
    ``face`` in a positive prompt whose whole point is that facial features are
    not to be described.
    """
    idx = text.find(clause)
    if idx < 0:
        return text, False
    tail = text[idx + len(clause):]
    end = re.search(r"\.(?=\s|$)", tail)
    tail = tail[end.end():] if end else ""
    return _join_head_tail(text[:idx], tail), True


def _target_half(text: str, marker: str, kind: str) -> str:
    """Keep only the clause after ``marker`` and give it a neutral sentence frame."""
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit(f"{kind}: plan prompt has no {marker!r} marker")
    rest = text[idx + len(marker):].strip()
    rest = re.sub(r"\s+", " ", rest)
    frame = TARGET_INTRO[kind]
    return frame + rest[0].lower() + rest[1:] if rest else frame.strip()


def _read_key(key_file: Path) -> str:
    key = key_file.read_text().strip()
    if len(key) < 16:
        raise SystemExit(f"credential file {key_file} does not look like an API key")
    return key


def build_prompts(bundle: dict) -> dict:
    """Turn the stored plan into the two positive prompts and the shared negative.

    Everything is taken from the plan text; the only edits are the ones the mask
    scope demands (drop the accessory clauses whose target sits outside the mask)
    and the ones identity preservation demands (drop the "who the person is"
    preamble, which the first run's model answered by drawing a second person).
    """
    style_id = bundle.get("style_id")
    if style_id != EXPECTED_STYLE_ID:
        raise SystemExit(f"prompt bundle style_id is {style_id!r}, expected {EXPECTED_STYLE_ID!r}")

    clothing = _target_half(bundle["clothing_prompt"], CLOTHING_TARGET_MARKER, "clothing")
    clothing, necklace_dropped = _drop_clause(clothing, NECKLACE_CLAUSE)
    clothing, glasses_dropped = _drop_clause_through(clothing, GLASSES_CLAUSE)
    clothing = f"{clothing.rstrip(' .,;')}, {CLOTHING_TARGET_SUFFIX}"

    hair = _target_half(bundle["hair_prompt"], HAIR_TARGET_MARKER, "hair")
    hair, current_hair_dropped = _drop_clause(hair, CURRENT_HAIR_CLAUSE)
    hair = f"{hair.rstrip(' .,;')}, {HAIR_TARGET_SUFFIX}"

    rule = _plan_mask_rule(bundle)
    prompts = {
        "style_id": style_id,
        "style_name": bundle.get("style_name"),
        "clothing": f"{clothing} {rule}",
        "hair": f"{hair} {rule}",
        "negative": NEGATIVE_PROMPT,
        "glasses_clause_dropped": glasses_dropped,
        "necklace_clause_dropped": necklace_dropped,
        "current_hair_clause_dropped": current_hair_dropped,
        "mask_rule": rule,
    }

    for key in ("clothing", "hair"):
        hit = FORBIDDEN_POSITIVE_PATTERN.search(prompts[key])
        if hit:
            raise SystemExit(
                f"{key} positive prompt names a facial feature or a person: {hit.group(0)!r} "
                f"in {prompts[key]!r}"
            )
    missing = [term for term in REQUIRED_NEGATIVE_TERMS if term not in prompts["negative"]]
    if missing:
        raise SystemExit(f"negative prompt is missing {missing}")
    for kind in ("clothing", "hair"):
        denoise = PASS_DENOISE[kind]
        if not DENOISE_BAND[0] <= denoise <= DENOISE_BAND[1]:
            raise SystemExit(f"{kind} denoise {denoise} outside {DENOISE_BAND}")
    return prompts


def build_workflow(package: Path, filenames: dict[str, str], prompts: dict, seed: int) -> dict:
    hair_mask = filenames["hair_mask"]
    clothing_mask = filenames["clothing_mask"]
    original = filenames["original"]
    crop = CROP_DEFAULTS

    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": original}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": clothing_mask, "channel": "red"}},
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "4": {
            "class_type": "InpaintCropImproved",
            "inputs": {"image": ["1", 0], "mask": ["2", 0], **crop},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["clothing"], "clip": ["3", 1]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["negative"], "clip": ["3", 1]},
        },
        # The crop's own pixels and the crop's own mask go into the conditioning.
        # ``InpaintModelConditioning`` returns a latent whose samples are the VAE
        # encoding of the *original* crop (not of a grey fill) plus a ``noise_mask``
        # built from the crop mask, and it is that latent -- reached through this
        # node's three outputs -- that the sampler must start from.
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
                "denoise": PASS_DENOISE["clothing"],
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 2]}},
        "10": {
            "class_type": "InpaintStitchImproved",
            "inputs": {"stitcher": ["4", 0], "inpainted_image": ["9", 0]},
        },
        "11": {"class_type": "LoadImageMask", "inputs": {"image": hair_mask, "channel": "red"}},
        "12": {
            "class_type": "InpaintCropImproved",
            "inputs": {"image": ["10", 0], "mask": ["11", 0], **crop},
        },
        "13": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["hair"], "clip": ["3", 1]},
        },
        "14": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["negative"], "clip": ["3", 1]},
        },
        "15": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["13", 0],
                "negative": ["14", 0],
                "vae": ["3", 2],
                "pixels": ["12", 1],
                "mask": ["12", 2],
                "noise_mask": True,
            },
        },
        "16": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed + 1,
                "positive": ["15", 0],
                "negative": ["15", 1],
                "latent_image": ["15", 2],
                **SAMPLER,
                "denoise": PASS_DENOISE["hair"],
            },
        },
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 2]}},
        "18": {
            "class_type": "InpaintStitchImproved",
            "inputs": {"stitcher": ["12", 0], "inpainted_image": ["17", 0]},
        },
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "p29_elegant"}},
    }


def _split_field(field):
    """Normalise a ComfyUI INPUT_TYPES entry into (kind, opts).

    ``field`` is either a bare type name, ``[type_name, opts]``, a combo
    ``[[option, ...]]`` or a combo with opts ``[[option, ...], opts]``.
    """
    if isinstance(field, list) and field:
        kind = field[0]
        opts = field[1] if len(field) > 1 and isinstance(field[1], dict) else {}
        return kind, opts
    return field, {}


def validate_workflow(workflow: dict, object_info: dict | None) -> list[str]:
    """Mimic ComfyUI prompt validation so a bad graph never reaches the platform."""
    errors: list[str] = []
    ids = set(workflow)
    slot_count: dict[str, int] = {}

    for node_id, node in workflow.items():
        spec = (object_info or {}).get(node["class_type"])
        if object_info is not None and spec is None:
            errors.append(f"node {node_id}: class_type {node['class_type']!r} is not installed")
            continue
        slot_count[node_id] = len(spec["output"]) if spec else 3

    for node_id, node in workflow.items():
        spec = (object_info or {}).get(node["class_type"])
        declared = {}
        if spec:
            declared = {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}
        for name, value in node["inputs"].items():
            is_link = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            if is_link:
                target, slot = value
                if target not in ids:
                    errors.append(f"node {node_id}.{name}: links to missing node {target!r}")
                elif slot >= slot_count.get(target, 0):
                    errors.append(
                        f"node {node_id}.{name}: node {target} has no output slot {slot} "
                        f"(has {slot_count.get(target)})"
                    )
                continue
            field = declared.get(name)
            if field is None:
                if spec is not None:
                    errors.append(f"node {node_id}.{name}: unknown input for {node['class_type']}")
                continue
            kind, opts = _split_field(field)
            if isinstance(kind, list):  # COMBO
                # image_upload inputs hold a server-side filename, not a bundled
                # example, so any non-empty name is accepted.
                if opts.get("image_upload") and isinstance(value, str) and value:
                    continue
                if value not in kind:
                    errors.append(f"node {node_id}.{name}: {value!r} not in allowed options")
            elif kind == "INT" and not isinstance(value, int):
                errors.append(f"node {node_id}.{name}: expected int, got {value!r}")
            elif kind == "FLOAT" and not isinstance(value, (int, float)):
                errors.append(f"node {node_id}.{name}: expected float, got {value!r}")
            elif kind == "STRING" and not isinstance(value, str):
                errors.append(f"node {node_id}.{name}: expected string, got {value!r}")
            elif kind in {"INT", "FLOAT"} and "min" in opts and value < opts["min"]:
                errors.append(f"node {node_id}.{name}: {value} below min {opts['min']}")
            elif kind in {"INT", "FLOAT"} and "max" in opts and value > opts["max"]:
                errors.append(f"node {node_id}.{name}: {value} above max {opts['max']}")

        if spec is None:
            continue
        provided = set(node["inputs"])
        for name in spec["input"].get("required", {}):
            if name not in provided:
                errors.append(f"node {node_id}.{name}: missing required input")
    return errors


# ---------------------------------------------------------------------------
# Pre-flight gate. Nothing below this line talks to the network; every check is
# local so a graph that would burn a paid run on a broken assumption is stopped
# here. The checks deliberately re-derive the *geometry* the runtime will use
# (mask processing, crop box, resize) instead of trusting the workflow JSON.
# ---------------------------------------------------------------------------

PROTECTED_CORE_ERODE_PX = 8


def _mask_pipeline_np(mask, blend_pixels: int, expand_pixels: int, fill_holes: bool):
    """Port of InpaintCropImproved's mask prep: hipass -> [fill] -> expand -> blur.

    Kept in numpy (the runtime is torch) so the geometry can be recomputed offline;
    ``evidence/p29-fix-run/mask_pipeline_probe.py`` checks this port against the
    real node source line by line.
    """
    import numpy as np
    from scipy import ndimage

    out = mask.copy()
    out[out < CROP_DEFAULTS["mask_hipass_filter"]] = 0
    if fill_holes:
        for threshold in (1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
            closed = ndimage.binary_closing(out >= threshold, structure=np.ones((3, 3)), border_value=1)
            filled = ndimage.binary_fill_holes(closed)
            out = np.maximum(out, np.where(filled != 0, threshold, 0))
    for pixels in (expand_pixels, blend_pixels):
        if pixels > 0:
            sigma = pixels / 4
            size = int(np.ceil(sigma * 1.5 + 1))
            out = np.clip(ndimage.grey_dilation(out, size=(size, size)), 0, 1)
    if blend_pixels > 0:
        out = np.clip(ndimage.gaussian_filter(out, sigma=blend_pixels * 0.5), 0, 1)
    return out


def _crop_box(mask, target_w: int, target_h: int, padding: int, factor: float, width: int, height: int):
    """Port of findcontextarea -> growcontextarea -> crop_magic_im's aspect fit."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    bx, by = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - bx + 1), int(ys.max() - by + 1)
    gx, gy = round(bw * (factor - 1.0) / 2.0), round(bh * (factor - 1.0) / 2.0)
    bx, by = max(0, bx - gx), max(0, by - gy)
    bw = min(width, bx + bw + 2 * gx) - bx
    bh = min(height, by + bh + 2 * gy) - by
    target_w = -(-target_w // padding) * padding
    target_h = -(-target_h // padding) * padding
    aspect = target_w / target_h
    if bw / bh < aspect:
        bw = int(bh * aspect)
    else:
        bh = int(bw / aspect)
    return dict(x=bx, y=by, w=bw, h=bh, target_w=target_w, target_h=target_h)


def _workflow_diff(before: dict, after: dict) -> list[str]:
    """Node-level diff of two API-format graphs, for the report's evidence trail."""
    lines: list[str] = []
    for node_id in sorted(set(before) | set(after), key=int):
        old, new = before.get(node_id), after.get(node_id)
        if old is None:
            lines.append(f"+ node {node_id} {new['class_type']}")
            continue
        if new is None:
            lines.append(f"- node {node_id} {old['class_type']}")
            continue
        if old["class_type"] != new["class_type"]:
            lines.append(f"~ node {node_id} {old['class_type']} -> {new['class_type']}")
            continue
        changed = [
            f"{key}: {old['inputs'].get(key)!r} -> {value!r}"
            for key, value in new["inputs"].items()
            if old["inputs"].get(key) != value
        ]
        changed += [f"{key}: {old['inputs'][key]!r} -> (removed)"
                    for key in old["inputs"] if key not in new["inputs"]]
        if changed:
            lines.append(f"~ node {node_id} {new['class_type']}: " + "; ".join(changed))
    return lines


def validate_checkpoint(object_info: dict | None) -> list[str]:
    """Report what the runtime can actually do with this checkpoint.

    The platform's whole checkpoint list is SD1.5 / SDXL base and Flux UNets -- no
    inpainting UNet, and the only inpaint weights are ControlNet / LLLite patches
    for Qwen-Image and Flux. A base SDXL model has no 9-channel inpainting input,
    so ``InpaintModelConditioning``'s ``concat_latent_image`` / ``concat_mask`` are
    ignored and the only thing that survives is its ``noise_mask`` plus the encoded
    original crop. That is a real, supported path -- google's mask-latent inpainting
    -- and it is what this workflow is built on; the verdict is printed so nobody
    later reads the graph as "concat inpainting that silently did nothing".
    """
    notes = [f"checkpoint {CHECKPOINT} has no inpainting UNet input; the run uses the "
             f"masked-latent path (InpaintModelConditioning noise_mask + encoded original crop)"]
    if object_info is None:
        notes.append("WARNING: no object_info, checkpoint existence not confirmed")
        return notes
    spec = object_info.get("CheckpointLoaderSimple")
    options = ((spec or {}).get("input", {}).get("required", {}).get("ckpt_name") or [[]])[0]
    if CHECKPOINT not in options:
        notes.append(f"ERROR: {CHECKPOINT} is not offered by the runtime")
    if any("inpaint" in name.lower() for name in options):
        notes.append("NOTE: the runtime lists an inpainting checkpoint; a future run may "
                     "prefer it and can then use the concat path")
    return notes


def collect_gate_checks(workflow: dict, object_info: dict | None, prompts: dict, bundle: dict,
                        package: Path) -> tuple[list[str], dict]:
    """Every check that must pass before a billed task is created."""
    import numpy as np
    from PIL import Image

    errors: list[str] = []
    report: dict = {
        "style_id": prompts["style_id"],
        "style_name": prompts.get("style_name"),
        "checkpoint": CHECKPOINT,
        "denoise": dict(PASS_DENOISE),
        "crop": {k: CROP_DEFAULTS[k] for k in
                 ("mask_fill_holes", "mask_expand_pixels", "mask_blend_pixels",
                  "context_from_mask_extend_factor", "output_target_width", "output_target_height")},
        "prompts": {k: prompts[k] for k in ("clothing", "hair", "negative")},
        "style_id_matches_plan": prompts["style_id"] == EXPECTED_STYLE_ID,
    }

    # 1. no plain text-to-image anywhere: an empty latent would ignore the photo.
    for node_id, node in workflow.items():
        if node["class_type"] in {"EmptyLatentImage", "EmptySD3LatentImage", "VAEEncodeForInpaint"}:
            errors.append(f"node {node_id}: {node['class_type']} must not appear in a masked edit")

    passes = {"clothing": ("8", "7", "4"), "hair": ("16", "15", "12")}
    for kind, (sampler_id, cond_id, crop_id) in passes.items():
        sampler, cond = workflow[sampler_id], workflow[cond_id]
        if cond["class_type"] != "InpaintModelConditioning":
            errors.append(f"{kind}: node {cond_id} is {cond['class_type']}, expected InpaintModelConditioning")
            continue
        if sampler["inputs"]["latent_image"] != [cond_id, 2]:
            errors.append(f"{kind}: sampler {sampler_id} latent is not {cond_id}'s latent output")
        if sampler["inputs"]["positive"] != [cond_id, 0] or sampler["inputs"]["negative"] != [cond_id, 1]:
            errors.append(f"{kind}: sampler {sampler_id} conditioning is not {cond_id}'s outputs")
        if cond["inputs"]["pixels"] != [crop_id, 1] or cond["inputs"]["mask"] != [crop_id, 2]:
            errors.append(f"{kind}: {cond_id} is not conditioned on crop node {crop_id}'s image/mask")
        if cond["inputs"]["noise_mask"] is not True:
            errors.append(f"{kind}: noise_mask must be enabled or the mask is not enforced")
        denoise = sampler["inputs"].get("denoise")
        if denoise is None or denoise >= 1.0:
            errors.append(f"{kind}: denoise {denoise} would discard the original crop pixels")
        elif not DENOISE_BAND[0] <= denoise <= DENOISE_BAND[1]:
            errors.append(f"{kind}: denoise {denoise} outside {DENOISE_BAND}")
        if workflow[sampler_id]["inputs"]["model"] != ["3", 0]:
            errors.append(f"{kind}: sampler is not driven by the loaded checkpoint")

    if workflow["10"]["inputs"]["stitcher"] != ["4", 0] or workflow["4"]["inputs"]["image"] != ["1", 0]:
        errors.append("clothing pass does not start from the original photo")
    if workflow["12"]["inputs"]["image"] != ["10", 0]:
        errors.append("hair pass does not start from the clothing pass's stitched output")

    ckpt = validate_checkpoint(object_info)
    report["checkpoint_notes"] = ckpt
    errors.extend(n for n in ckpt if n.startswith("ERROR"))

    # 2. the plan's own wording has to survive into the prompts.
    for kind, phrase in (("clothing", "深橄榄绿法式复古衬衫"), ("hair", "栗子棕")):
        if phrase not in prompts[kind]:
            errors.append(f"{kind} prompt lost the plan phrase {phrase!r}")
    report["prompt_has_plan_phrases"] = all(
        phrase in prompts[kind]
        for kind, phrase in (("clothing", "深橄榄绿法式复古衬衫"), ("hair", "栗子棕"))
    )
    report["negative_prompt_terms_ok"] = all(t in prompts["negative"] for t in REQUIRED_NEGATIVE_TERMS)

    # 3. files, sizes and coordinates.
    source = json.loads((package / "manifest.json").read_text())["source_image"]
    frame = (source["width"], source["height"])
    sizes = {}
    for name in ("original.jpg", "hair_edit_mask.png", "clothing_edit_mask.png", "protected_identity_mask.png"):
        path = package / name
        if not path.is_file():
            errors.append(f"package is missing {name}")
            continue
        with Image.open(path) as image:
            sizes[name] = image.size
        if sizes[name] != frame:
            errors.append(f"{name} is {sizes[name]}, source photo is {frame}")
    report["image_sizes"] = sizes

    # 4. mask geometry as the node will actually build it.
    if len(sizes) == 4:
        load = lambda n: (np.array(Image.open(package / n).convert("L")) > 127).astype(np.float32)
        hair_raw, clothing_raw = load("hair_edit_mask.png"), load("clothing_edit_mask.png")
        protected = load("protected_identity_mask.png") > 0.5
        from scipy import ndimage
        core = ndimage.binary_erosion(protected, iterations=PROTECTED_CORE_ERODE_PX)
        geometry = {}
        for kind, raw in (("hair", hair_raw), ("clothing", clothing_raw)):
            prepared = _mask_pipeline_np(
                raw,
                blend_pixels=int(CROP_DEFAULTS["mask_blend_pixels"]),
                expand_pixels=int(CROP_DEFAULTS["mask_expand_pixels"]),
                fill_holes=bool(CROP_DEFAULTS["mask_fill_holes"]),
            )
            hard, rim = prepared > 0.5, prepared > 0.02
            box = _crop_box(prepared, int(CROP_DEFAULTS["output_target_width"]),
                            int(CROP_DEFAULTS["output_target_height"]), 32,
                            float(CROP_DEFAULTS["context_from_mask_extend_factor"]), *frame)
            geometry[kind] = {
                "raw_pixels": int((raw > 0.5).sum()),
                "painted_pixels": int(hard.sum()),
                "soft_ring_pixels": int((rim & ~hard).sum()),
                "painted_over_protected": int((hard & protected).sum()),
                "painted_over_facial_core": int((hard & core).sum()),
                "crop": box,
                "scale_x": round(box["target_w"] / box["w"], 4),
                "scale_y": round(box["target_h"] / box["h"], 4),
            }
            if int((hard & core).sum()):
                errors.append(f"{kind}: the painted mask reaches the facial core "
                              f"({int((hard & core).sum())} px inside protected-by-{PROTECTED_CORE_ERODE_PX}px)")
            if abs(geometry[kind]["scale_x"] / geometry[kind]["scale_y"] - 1) > 0.01:
                errors.append(f"{kind}: crop is rescaled non-uniformly "
                              f"({geometry[kind]['scale_x']} x {geometry[kind]['scale_y']})")
        report["mask_geometry"] = geometry

    report["errors"] = errors
    return errors, report


def _post(client: httpx.Client, path: str, key: str, payload: dict) -> dict:
    response = client.post(path, json=payload, headers={"Authorization": f"Bearer {key}"})
    response.raise_for_status()
    body = response.json()
    if body.get("code") not in (0, None):
        raise SystemExit(f"{path} -> code={body.get('code')} msg={body.get('msg')} data={body.get('data')}")
    return body


def upload(client: httpx.Client, key: str, path: Path) -> str:
    with path.open("rb") as handle:
        response = client.post(
            UPLOAD_PATH,
            data={"apiKey": key},
            files={"file": (path.name, handle, "application/octet-stream")},
        )
    response.raise_for_status()
    body = response.json()
    data = body.get("data") or {}
    name = data.get("fileName") or data.get("filename")
    if body.get("code") not in (0, None) or not name:
        raise SystemExit(f"upload {path.name} failed: code={body.get('code')} keys={sorted(data)}")
    print(f"  uploaded {path.name} ({path.stat().st_size} B) -> {name}")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True, help="local-edit-package directory")
    parser.add_argument("--out", type=Path, default=Path("artifacts/p29-fix-run"))
    parser.add_argument("--compare-to", type=Path, default=Path("artifacts/p29-run/workflow.json"),
                        help="previous workflow to diff the node/connection changes against")
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".runninghub_key")
    parser.add_argument("--object-info", type=Path, default=Path("evidence/p29-live-run/object_info_live.json"))
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--instance-type", default="default")
    # ``workflowId`` is a required (non-null) field on /task/openapi/create even
    # when the graph is supplied inline via ``workflow``; the API documents that
    # ``workflow`` takes precedence over it. Without it the call is rejected with
    # code=301 "must not be null" before any task is created.
    parser.add_argument("--workflow-id", default="1")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="build + validate only (default)")
    parser.add_argument("--upload", action="store_true", help="upload the three input files")
    parser.add_argument("--submit", action="store_true", help="create the one paid task")
    parser.add_argument("--i-understand-this-costs-money", action="store_true")
    args = parser.parse_args(argv)

    package = args.package
    bundle = json.loads((package / "prompt_bundle.json").read_text())
    prompts = build_prompts(bundle)
    for key, label in (("glasses_clause_dropped", GLASSES_CLAUSE),
                       ("necklace_clause_dropped", NECKLACE_CLAUSE),
                       ("current_hair_clause_dropped", CURRENT_HAIR_CLAUSE)):
        if not prompts[key]:
            print(f"WARNING: clause {label!r} not found in the plan prompt; nothing dropped",
                  file=sys.stderr)

    staged = {
        "original": "original.jpg",
        "clothing_mask": "clothing_edit_mask.png",
        "hair_mask": "hair_edit_mask.png",
    }
    for name in staged.values():
        if not (package / name).is_file():
            raise SystemExit(f"package is missing {name}")

    args.out.mkdir(parents=True, exist_ok=True)
    workflow = build_workflow(package, {k: f"__UPLOAD__{k}" for k in staged}, prompts, args.seed)

    object_info = None
    if args.object_info.is_file():
        object_info = json.loads(args.object_info.read_text())
        print(f"validating against {args.object_info} ({len(object_info)} installed nodes)")
    else:
        print(f"WARNING: {args.object_info} not found; structural checks only", file=sys.stderr)
    errors = validate_workflow(workflow, object_info)
    if errors:
        for err in errors:
            print(f"  INVALID  {err}", file=sys.stderr)
        return 1
    print(f"workflow OK: {len(workflow)} nodes against the live node table")

    gate_errors, gate = collect_gate_checks(workflow, object_info, prompts, bundle, package)
    gate["validate_workflow_errors"] = errors
    if args.compare_to.is_file():
        gate["workflow_diff_vs"] = str(args.compare_to)
        gate["workflow_diff"] = _workflow_diff(json.loads(args.compare_to.read_text()), workflow)
    (args.out / "gate_report.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"style {gate['style_id']} ({gate['style_name']}), denoise {gate['denoise']}")
    for note in gate["checkpoint_notes"]:
        print(f"  checkpoint: {note}")
    for kind, geom in gate.get("mask_geometry", {}).items():
        print(f"  {kind:8} painted={geom['painted_pixels']:>7} ring={geom['soft_ring_pixels']:>6} "
              f"over_protected={geom['painted_over_protected']:>5} "
              f"over_facial_core={geom['painted_over_facial_core']} "
              f"crop={geom['crop']['w']}x{geom['crop']['h']} -> "
              f"{geom['crop']['target_w']}x{geom['crop']['target_h']} "
              f"(scale {geom['scale_x']},{geom['scale_y']})")
    if gate.get("workflow_diff"):
        print("  workflow diff vs previous run:")
        for line in gate["workflow_diff"]:
            print(f"    {line}")
    if gate_errors:
        for err in gate_errors:
            print(f"  GATE FAILED  {err}", file=sys.stderr)
        print(f"gate report -> {args.out / 'gate_report.json'}", file=sys.stderr)
        return 1
    print(f"gate passed -> {args.out / 'gate_report.json'}")

    if not (args.upload or args.submit):
        (args.out / "workflow.json").write_text(json.dumps(workflow, indent=2))
        print(f"dry run complete -> {args.out / 'workflow.json'}")
        return 0

    key = _read_key(args.key_file)
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)
    print("uploading inputs")
    filenames = {k: upload(client, key, package / v) for k, v in staged.items()}
    workflow = build_workflow(package, filenames, prompts, args.seed)
    errors = validate_workflow(workflow, object_info)
    if errors:
        for err in errors:
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
    created = _post(client, CREATE_PATH, key, payload)
    task_id = created["data"]["taskId"]
    print(f"task created: {task_id} status={created['data'].get('taskStatus')}")
    if created["data"].get("promptTips"):
        print(f"promptTips: {created['data']['promptTips']}")
    (args.out / "task.json").write_text(json.dumps({"taskId": task_id, "create": created}, indent=2))

    deadline = time.time() + args.poll_seconds
    status = "QUEUED"
    while time.time() < deadline:
        time.sleep(6)
        status = _post(client, STATUS_PATH, key, {"apiKey": key, "taskId": task_id})["data"]
        print(f"  {status}")
        if status in {"SUCCESS", "FAILED"}:
            break
    if status != "SUCCESS":
        print(f"task did not succeed (status={status}); no result downloaded", file=sys.stderr)
        return 1

    outputs = _post(client, OUTPUTS_PATH, key, {"apiKey": key, "taskId": task_id})["data"]
    (args.out / "outputs.json").write_text(json.dumps(outputs, indent=2))
    for index, item in enumerate(outputs):
        url = item.get("fileUrl")
        if not url:
            continue
        suffix = Path(url.split("?")[0]).suffix or ".png"
        target = args.out / f"result_{index}{suffix}"
        target.write_bytes(client.get(url).content)
        print(f"  result -> {target} ({target.stat().st_size} B) consumeCoins={item.get('consumeCoins')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
