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

# Mask geometry decisions (see evidence/P2.9-runninghub-run.md for the rationale):
#   * mask_blend_pixels=4 -- the blend ring is the only place generated pixels may
#     touch the protected identity mask, whose closest edit pixel is 1 px away.
#     4 px keeps that ring small while still feathering the stitch seam.
#   * mask_expand_pixels=0 -- the SCHP masks already cover the target silhouette.
#   * mask_fill_holes=True -- fills the 283 px of speckle holes inside the clothing
#     mask. Safe here because neither edit mask encloses the face (0 enclosed holes).
#   * grow_mask_by=0 on VAEEncodeForInpaint -- no second, silent mask dilation.
CROP_DEFAULTS: dict[str, object] = {
    "downscale_algorithm": "bilinear",
    "upscale_algorithm": "bicubic",
    "preresize": False,
    "preresize_mode": "ensure minimum resolution",
    "preresize_min_width": 1024,
    "preresize_min_height": 1024,
    "preresize_max_width": 16384,
    "preresize_max_height": 16384,
    "mask_fill_holes": True,
    "mask_expand_pixels": 0,
    "mask_invert": False,
    "mask_blend_pixels": 4,
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
    "denoise": 1.0,
}

# The plan's Accessories clause asks for new browline glasses, but this round's
# acceptance list pins "glasses unchanged" and the glasses sit inside the
# protected identity mask, so the clause is dropped from the prompt instead of
# contradicting the identity constraints in the same conditioning.
GLASSES_CLAUSE = "replace the current round thin metal-frame glasses"

# English rendering of prompt_bundle.negative_prompt (which is Chinese; CLIP for
# SDXL is English-trained) plus two stitch-seam terms.
NEGATIVE_PROMPT = (
    "plastic airbrushed skin, blurred face, reshaped or replaced facial features, "
    "a different person's face, obvious age drift, changing the number of people, "
    "a broken or damaged background, extra fingers or limbs, text, watermark, "
    "visible seam, hard cut-out edge"
)


def _read_key(key_file: Path) -> str:
    key = key_file.read_text().strip()
    if len(key) < 16:
        raise SystemExit(f"credential file {key_file} does not look like an API key")
    return key


def _strip_glasses_clause(clothing_prompt: str) -> tuple[str, bool]:
    idx = clothing_prompt.find(GLASSES_CLAUSE)
    if idx < 0:
        return clothing_prompt, False
    return clothing_prompt[:idx].strip(), True


def build_prompts(bundle: dict) -> dict:
    clothing, dropped = _strip_glasses_clause(bundle["clothing_prompt"])
    identity = bundle["identity_constraints"]["prompt_text_en"]
    return {
        "hair": f"{bundle['hair_prompt']}\n{identity}",
        "clothing": f"{clothing}\n{identity}",
        "negative": NEGATIVE_PROMPT,
        "glasses_clause_dropped": dropped,
    }


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
        "7": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {"pixels": ["4", 1], "vae": ["3", 2], "mask": ["4", 2], "grow_mask_by": 0},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed,
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                **SAMPLER,
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
            "class_type": "VAEEncodeForInpaint",
            "inputs": {"pixels": ["12", 1], "vae": ["3", 2], "mask": ["12", 2], "grow_mask_by": 0},
        },
        "16": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed + 1,
                "positive": ["13", 0],
                "negative": ["14", 0],
                "latent_image": ["15", 0],
                **SAMPLER,
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
    parser.add_argument("--out", type=Path, default=Path("artifacts/p29-run"))
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
    if not prompts["glasses_clause_dropped"]:
        print("WARNING: glasses clause not found in clothing_prompt; prompt kept verbatim", file=sys.stderr)

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
    print(f"workflow OK: {len(workflow)} nodes, {len(prompts['clothing'].splitlines())} prompt blocks")

    if not (args.upload or args.submit):
        (args.out / "workflow.template.json").write_text(json.dumps(workflow, indent=2))
        print(f"dry run complete -> {args.out / 'workflow.template.json'}")
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
