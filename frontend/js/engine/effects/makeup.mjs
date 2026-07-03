import { createCanvas } from "../core/canvas.mjs";
import { drawFilteredRegion } from "../core/filtering.mjs";
import { drawBrows } from "./features.mjs";

function drawBlush(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  [
    { x: width * 0.37, y: height * 0.43 },
    { x: width * 0.63, y: height * 0.43 },
  ].forEach((point) => {
    const blush = context.createRadialGradient(
      point.x,
      point.y,
      0,
      point.x,
      point.y,
      width * 0.095,
    );
    blush.addColorStop(0, `rgba(231, 104, 116, ${strength * 0.4})`);
    blush.addColorStop(0.58, `rgba(222, 118, 126, ${strength * 0.2})`);
    blush.addColorStop(1, "rgba(222, 118, 126, 0)");
    context.fillStyle = blush;
    context.fillRect(
      point.x - width * 0.11,
      point.y - height * 0.075,
      width * 0.22,
      height * 0.15,
    );
  });
  context.restore();
}

function drawMakeupContour(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  [
    { x: width * 0.31, y: height * 0.43 },
    { x: width * 0.69, y: height * 0.43 },
  ].forEach((point) => {
    const shadow = context.createRadialGradient(
      point.x,
      point.y,
      0,
      point.x,
      point.y,
      width * 0.14,
    );
    shadow.addColorStop(0, `rgba(78, 48, 43, ${strength * 0.22})`);
    shadow.addColorStop(1, "rgba(78, 48, 43, 0)");
    context.fillStyle = shadow;
    context.fillRect(
      point.x - width * 0.15,
      point.y - height * 0.13,
      width * 0.3,
      height * 0.26,
    );
  });

  const centerLight = context.createRadialGradient(
    width * 0.5,
    height * 0.39,
    0,
    width * 0.5,
    height * 0.39,
    width * 0.16,
  );
  centerLight.addColorStop(
    0,
    `rgba(255, 236, 218, ${strength * 0.16})`,
  );
  centerLight.addColorStop(1, "rgba(255, 236, 218, 0)");
  context.fillStyle = centerLight;
  context.fillRect(
    width * 0.34,
    height * 0.24,
    width * 0.32,
    height * 0.32,
  );
  context.restore();
}

function drawEyeliner(context, source, width, height, strength) {
  if (strength <= 0) return;

  [
    {
      x: width * 0.36,
      y: height * 0.335,
      width: width * 0.14,
      height: height * 0.06,
    },
    {
      x: width * 0.5,
      y: height * 0.335,
      width: width * 0.14,
      height: height * 0.06,
    },
  ].forEach((region) => {
    drawFilteredRegion(
      context,
      source,
      region,
      `brightness(${(1 - strength * 0.24).toFixed(3)}) contrast(${(
        1 +
        strength * 0.42
      ).toFixed(3)})`,
      strength * 0.48,
      0.46,
    );
  });
}

function drawLipstick(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const lipstick = context.createRadialGradient(
    width * 0.5,
    height * 0.525,
    0,
    width * 0.5,
    height * 0.525,
    width * 0.105,
  );
  lipstick.addColorStop(0, `rgba(184, 42, 68, ${strength * 0.72})`);
  lipstick.addColorStop(0.62, `rgba(143, 31, 55, ${strength * 0.42})`);
  lipstick.addColorStop(1, "rgba(143, 31, 55, 0)");
  context.fillStyle = lipstick;
  context.beginPath();
  context.ellipse(
    width * 0.5,
    height * 0.525,
    width * 0.105,
    height * 0.031,
    0,
    0,
    Math.PI * 2,
  );
  context.fill();
  context.restore();
}

export function drawMakeupRetouch(context, width, height, recipe) {
  const makeupSource = createCanvas(width, height);
  makeupSource.getContext("2d").drawImage(context.canvas, 0, 0);
  drawFilteredRegion(
    context,
    makeupSource,
    {
      x: width * 0.27,
      y: height * 0.2,
      width: width * 0.46,
      height: height * 0.38,
    },
    `blur(${(recipe.foundationStrength * 2.2).toFixed(
      1,
    )}px) brightness(${(1 + recipe.foundationStrength * 0.055).toFixed(
      3,
    )}) saturate(${(1 + recipe.foundationStrength * 0.045).toFixed(
      3,
    )}) sepia(${(recipe.foundationStrength * 0.025).toFixed(3)})`,
    recipe.foundationStrength * 0.34,
    0.7,
  );
  drawBlush(context, width, height, recipe.blushStrength);
  drawMakeupContour(context, width, height, recipe.contourStrength);

  const eyebrowSource = createCanvas(width, height);
  eyebrowSource.getContext("2d").drawImage(context.canvas, 0, 0);
  drawBrows(
    context,
    eyebrowSource,
    width,
    height,
    recipe.eyebrowMakeupStrength * 0.9,
  );
  drawEyeliner(
    context,
    eyebrowSource,
    width,
    height,
    recipe.eyelinerStrength,
  );
  drawLipstick(context, width, height, recipe.lipstickStrength);
}
