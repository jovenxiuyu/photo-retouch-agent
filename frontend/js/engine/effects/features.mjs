import { createCanvas } from "../core/canvas.mjs";
import { drawFilteredRegion } from "../core/filtering.mjs";
import {
  drawCurrentWarp,
  drawTransformedRegion,
} from "../geometry/warp.mjs";

function drawNoseBridge(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const highlight = context.createLinearGradient(
    width * 0.5,
    height * 0.31,
    width * 0.5,
    height * 0.5,
  );
  highlight.addColorStop(0, "rgba(255, 244, 226, 0)");
  highlight.addColorStop(
    0.45,
    `rgba(255, 244, 226, ${strength * 0.32})`,
  );
  highlight.addColorStop(1, "rgba(255, 244, 226, 0)");
  context.strokeStyle = highlight;
  context.lineWidth = width * 0.018;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(width * 0.5, height * 0.32);
  context.lineTo(width * 0.5, height * 0.49);
  context.stroke();

  context.strokeStyle = `rgba(76, 48, 39, ${strength * 0.09})`;
  context.lineWidth = width * 0.012;
  [
    [width * 0.475, width * 0.485],
    [width * 0.525, width * 0.515],
  ].forEach(([startX, endX]) => {
    context.beginPath();
    context.moveTo(startX, height * 0.34);
    context.lineTo(endX, height * 0.49);
    context.stroke();
  });
  context.restore();
}

function drawLipColor(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const lipGradient = context.createRadialGradient(
    width * 0.5,
    height * 0.525,
    0,
    width * 0.5,
    height * 0.525,
    width * 0.11,
  );
  lipGradient.addColorStop(
    0,
    `rgba(188, 48, 72, ${strength * 0.52})`,
  );
  lipGradient.addColorStop(
    0.58,
    `rgba(164, 42, 63, ${strength * 0.3})`,
  );
  lipGradient.addColorStop(1, "rgba(164, 42, 63, 0)");
  context.fillStyle = lipGradient;
  context.beginPath();
  context.ellipse(
    width * 0.5,
    height * 0.525,
    width * 0.105,
    height * 0.032,
    0,
    0,
    Math.PI * 2,
  );
  context.fill();
  context.restore();
}

export function drawBrows(context, source, width, height, strength) {
  if (strength <= 0) return;

  [
    {
      x: width * 0.34,
      y: height * 0.285,
      width: width * 0.15,
      height: height * 0.065,
    },
    {
      x: width * 0.51,
      y: height * 0.285,
      width: width * 0.15,
      height: height * 0.065,
    },
  ].forEach((region) => {
    drawFilteredRegion(
      context,
      source,
      region,
      `brightness(${(1 - strength * 0.18).toFixed(3)}) contrast(${(
        1 +
        strength * 0.32
      ).toFixed(3)})`,
      strength * 0.58,
      0.5,
    );
  });
}

export function drawFeatureRetouch(context, width, height, recipe) {
  drawCurrentWarp(
    context,
    {
      x: width * 0.41,
      y: height * 0.31,
      width: width * 0.18,
      height: height * 0.23,
    },
    recipe.noseScaleX,
    1,
  );

  if (recipe.lipScaleY > 1) {
    const lipSource = createCanvas(width, height);
    lipSource.getContext("2d").drawImage(context.canvas, 0, 0);
    drawTransformedRegion(
      context,
      lipSource,
      {
        x: width * 0.39,
        y: height * 0.475,
        width: width * 0.22,
        height: height * 0.105,
      },
      {
        scaleX: 1 + (recipe.lipScaleY - 1) * 0.2,
        scaleY: recipe.lipScaleY,
      },
    );
  }

  if (recipe.smileAngle > 0) {
    const smileSource = createCanvas(width, height);
    smileSource.getContext("2d").drawImage(context.canvas, 0, 0);
    [
      {
        x: width * 0.385,
        y: height * 0.48,
        width: width * 0.125,
        height: height * 0.1,
        rotation: recipe.smileAngle,
      },
      {
        x: width * 0.49,
        y: height * 0.48,
        width: width * 0.125,
        height: height * 0.1,
        rotation: -recipe.smileAngle,
      },
    ].forEach((region) => {
      drawTransformedRegion(context, smileSource, region, {
        rotation: region.rotation,
      });
    });
  }

  const featureSource = createCanvas(width, height);
  featureSource.getContext("2d").drawImage(context.canvas, 0, 0);
  drawFilteredRegion(
    context,
    featureSource,
    {
      x: width * 0.42,
      y: height * 0.505,
      width: width * 0.16,
      height: height * 0.055,
    },
    `brightness(${(1 + recipe.teethStrength * 0.24).toFixed(
      3,
    )}) saturate(${(1 - recipe.teethStrength * 0.55).toFixed(3)})`,
    recipe.teethStrength * 0.52,
    0.48,
  );

  drawNoseBridge(context, width, height, recipe.noseBridgeStrength);
  drawLipColor(context, width, height, recipe.lipColorStrength);
  drawBrows(context, featureSource, width, height, recipe.browStrength);
}
