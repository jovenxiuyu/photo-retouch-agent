import assert from "node:assert/strict";
import test from "node:test";

import { createCanvas } from "../js/engine/core/canvas.mjs";
import {
  adjustmentAmount,
  clamp,
} from "../js/engine/core/math.mjs";
import { applyFeatherMask } from "../js/engine/core/mask.mjs";
import {
  drawAxisWarp,
  mapWarpPosition,
} from "../js/engine/geometry/warp.mjs";

test("normalizes adjustment values in the engine math layer", () => {
  assert.equal(clamp(120, 0, 100), 100);
  assert.equal(adjustmentAmount({ smooth: "45" }, "smooth"), 45);
  assert.equal(adjustmentAmount({ smooth: -10 }, "smooth"), 0);
  assert.equal(adjustmentAmount({ smooth: "invalid" }, "smooth"), 0);
});

test("creates canvases through an injectable document boundary", () => {
  const canvas = {};
  const documentRef = {
    createElement(tag) {
      assert.equal(tag, "canvas");
      return canvas;
    },
  };

  assert.equal(createCanvas(640, 960, documentRef), canvas);
  assert.deepEqual(canvas, { width: 640, height: 960 });
});

test("applies a feathered destination mask to a region", () => {
  const calls = [];
  const gradient = {
    addColorStop(...args) {
      calls.push(["stop", ...args]);
    },
  };
  const context = {
    save() {
      calls.push(["save"]);
    },
    restore() {
      calls.push(["restore"]);
    },
    translate(...args) {
      calls.push(["translate", ...args]);
    },
    scale(...args) {
      calls.push(["scale", ...args]);
    },
    createRadialGradient() {
      return gradient;
    },
    fillRect(...args) {
      calls.push(["fillRect", ...args]);
    },
  };

  applyFeatherMask(
    { getContext: () => context },
    { x: 10, y: 20, width: 80, height: 100 },
    0.7,
    0.5,
  );

  assert.equal(context.globalCompositeOperation, "destination-in");
  assert.deepEqual(calls[1], ["translate", 50, 70]);
  assert.deepEqual(calls[2], ["scale", 40, 50]);
  assert.deepEqual(calls.at(-2), ["fillRect", -1, -1, 2, 2]);
});

test("geometry warp preserves edges and emits deterministic slices", () => {
  const draws = [];
  const context = {
    drawImage(...args) {
      draws.push(args);
    },
  };
  const region = { x: 10, y: 20, width: 120, height: 180 };

  drawAxisWarp(context, {}, region, 0.92, "x");

  assert.equal(draws.length, 48);
  assert.equal(draws[0][5], region.x);
  assert.ok(Math.abs(mapWarpPosition(1, 0.92) - 1) < Number.EPSILON);
});
