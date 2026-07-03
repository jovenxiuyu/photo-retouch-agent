import { createCanvas } from "./engine/core/canvas.mjs";
import {
  drawBodyRetouch,
  drawEyeRetouch,
  drawFaceRetouch,
  drawFeatureRetouch,
  drawMakeupRetouch,
  drawSculptLight,
  drawSkinRetouch,
  drawToneRetouch,
} from "./engine/effects/index.mjs";
import {
  adjustmentAmount as amount,
  clamp,
} from "./engine/core/math.mjs";
import { mapWarpPosition } from "./engine/geometry/warp.mjs";

export { mapWarpPosition };

export function createRenderRecipe(values = {}) {
  const brightness =
    1 +
    (amount(values, "whiten") +
      amount(values, "skinTone") * 0.18 +
      amount(values, "brightness")) /
      720;
  const saturation =
    1 + amount(values, "saturation") / 900;
  const contrast =
    1 + amount(values, "contrast") / 850;
  const warmth =
    amount(values, "warmth") / 1300;
  const faceSlimming =
    (amount(values, "slimFace") + amount(values, "smallFace") * 0.75) /
    2100;
  return {
    brightness: clamp(brightness, 0.8, 1.35),
    saturation: clamp(saturation, 0.75, 1.45),
    contrast: clamp(contrast, 0.8, 1.4),
    warmth: clamp(warmth, 0, 0.18),
    skinToneStrength: amount(values, "skinTone") / 100,
    whiteningStrength: amount(values, "whiten") / 100,
    smoothingOpacity: clamp(amount(values, "smooth") / 240, 0, 0.42),
    smoothingRadius: clamp(
      1.2 + amount(values, "smooth") * 0.075,
      1.2,
      9,
    ),
    acneOpacity: clamp(amount(values, "acne") / 260, 0, 0.38),
    acneRadius: clamp(1 + amount(values, "acne") * 0.065, 1, 7.5),
    wrinkleOpacity: clamp(amount(values, "wrinkle") / 250, 0, 0.4),
    wrinkleRadius: clamp(
      0.8 + amount(values, "wrinkle") * 0.045,
      0.8,
      5.3,
    ),
    faceScaleX: clamp(1 - faceSlimming, 0.92, 1),
    faceScaleY: clamp(1 - amount(values, "smallFace") / 2500, 0.96, 1),
    jawScaleX: clamp(1 - amount(values, "jawline") / 1600, 0.935, 1),
    cheekScaleX: clamp(1 - amount(values, "cheekbone") / 2000, 0.95, 1),
    chinScaleY: clamp(1 + amount(values, "chin") / 1800, 1, 1.06),
    foreheadScaleY: clamp(
      1 - amount(values, "forehead") / 2000,
      0.95,
      1,
    ),
    doubleChinScaleY: clamp(
      1 - amount(values, "doubleChin") / 1800,
      0.94,
      1,
    ),
    bellyScaleX: clamp(1 - amount(values, "slimBelly") / 1000, 0.9, 1),
    waistScaleX: clamp(1 - amount(values, "slimWaist") / 820, 0.875, 1),
    armScaleX: clamp(1 - amount(values, "slimArms") / 1200, 0.915, 1),
    legScaleX: clamp(1 - amount(values, "slimLegs") / 1100, 0.905, 1),
    legScaleY: clamp(1 + amount(values, "longLegs") / 1000, 1, 1.1),
    shoulderScaleX: clamp(
      1 + amount(values, "shoulders") / 1250,
      1,
      1.08,
    ),
    hipScaleX: clamp(1 + amount(values, "hips") / 1400, 1, 1.075),
    hipScaleY: clamp(1 - amount(values, "hips") / 2500, 0.96, 1),
    headRatioScale: clamp(
      1 - amount(values, "headRatio") / 1600,
      0.935,
      1,
    ),
    eyeScale: clamp(1 + amount(values, "enlargeEyes") / 550, 1, 1.18),
    eyeBagStrength: amount(values, "eyeBags") / 100,
    darkCircleStrength: amount(values, "darkCircles") / 100,
    irisStrength: amount(values, "iris") / 100,
    brightEyeStrength: amount(values, "brightEyes") / 100,
    eyeGlow: clamp(
      amount(values, "brightEyes") / 185 +
        amount(values, "iris") / 1000,
      0,
      0.65,
    ),
    eyeDistanceScale: clamp(
      1 + amount(values, "eyeDistance") / 1800,
      1,
      1.06,
    ),
    eyeAngle: clamp(
      (amount(values, "eyeAngle") / 100) * (Math.PI / 45),
      0,
      Math.PI / 45,
    ),
    noseBridgeStrength: amount(values, "noseBridge") / 100,
    noseScaleX: clamp(1 - amount(values, "slimNose") / 1300, 0.92, 1),
    teethStrength: amount(values, "teeth") / 100,
    smileAngle: clamp(
      (amount(values, "smile") / 100) * (Math.PI / 45),
      0,
      Math.PI / 45,
    ),
    lipScaleY: clamp(1 + amount(values, "lipVolume") / 600, 1, 1.17),
    lipColorStrength: amount(values, "lipColor") / 100,
    browStrength: amount(values, "brows") / 100,
    foundationStrength: amount(values, "foundation") / 100,
    blushStrength: amount(values, "blush") / 100,
    contourStrength: amount(values, "contour") / 100,
    eyebrowMakeupStrength: amount(values, "eyebrowMakeup") / 100,
    eyelinerStrength: amount(values, "eyeliner") / 100,
    lipstickStrength: amount(values, "lipstick") / 100,
    toneBrightnessStrength: amount(values, "brightness") / 100,
    toneContrastStrength: amount(values, "contrast") / 100,
    toneWarmthStrength: amount(values, "warmth") / 100,
    toneSaturationStrength: amount(values, "saturation") / 100,
    clarityStrength: amount(values, "clarity") / 100,
    backgroundStrength: amount(values, "background") / 100,
    sculpt: clamp(amount(values, "sculpt") / 140, 0, 0.72),
  };
}

export function buildCanvasFilter(recipe) {
  return [
    `brightness(${recipe.brightness.toFixed(3)})`,
    `saturate(${recipe.saturation.toFixed(3)})`,
    `contrast(${recipe.contrast.toFixed(3)})`,
    `sepia(${recipe.warmth.toFixed(3)})`,
  ].join(" ");
}

export function activeAdjustmentCount(values = {}) {
  return Object.values(values).filter((value) => Number(value) > 0).length;
}

export function renderLocalRetouch(canvas, image, values = {}) {
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!width || !height) return null;

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  const context = canvas.getContext("2d", { alpha: false });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  const source = createCanvas(width, height);
  source.getContext("2d").drawImage(image, 0, 0, width, height);
  const recipe = createRenderRecipe(values);

  context.save();
  context.filter = buildCanvasFilter(recipe);
  context.drawImage(source, 0, 0, width, height);
  context.restore();

  drawToneRetouch(context, width, height, recipe);
  drawSkinRetouch(context, width, height, recipe);
  drawFaceRetouch(context, width, height, recipe);
  drawBodyRetouch(context, width, height, recipe);
  drawFeatureRetouch(context, width, height, recipe);
  drawEyeRetouch(context, width, height, recipe);
  drawMakeupRetouch(context, width, height, recipe);
  drawSculptLight(context, width, height, recipe);

  return recipe;
}
