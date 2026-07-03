import { createCanvas } from "../core/canvas.mjs";
import { drawFilteredRegion } from "../core/filtering.mjs";
import {
  drawCurrentWarp,
  drawTransformedRegion,
} from "../geometry/warp.mjs";

function drawEyeGlow(context, width, height, recipe) {
  if (recipe.eyeGlow <= 0) return;

  const eyes = [
    { x: width * 0.43, y: height * 0.365 },
    { x: width * 0.57, y: height * 0.365 },
  ];
  context.save();
  context.globalCompositeOperation = "screen";
  eyes.forEach((eye) => {
    const radius = width * 0.055 * recipe.eyeScale;
    const gradient = context.createRadialGradient(
      eye.x,
      eye.y,
      0,
      eye.x,
      eye.y,
      radius,
    );
    gradient.addColorStop(0, `rgba(240, 247, 248, ${recipe.eyeGlow * 0.2})`);
    gradient.addColorStop(0.35, `rgba(220, 234, 236, ${recipe.eyeGlow * 0.1})`);
    gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
    context.fillStyle = gradient;
    context.beginPath();
    context.ellipse(
      eye.x,
      eye.y,
      radius,
      radius * 0.52,
      0,
      0,
      Math.PI * 2,
    );
    context.fill();
  });
  context.restore();
}

export function drawEyeRetouch(context, width, height, recipe) {
  const eyeRegions = [
    {
      x: width * 0.345,
      y: height * 0.315,
      width: width * 0.17,
      height: height * 0.105,
      angle: recipe.eyeAngle,
    },
    {
      x: width * 0.485,
      y: height * 0.315,
      width: width * 0.17,
      height: height * 0.105,
      angle: -recipe.eyeAngle,
    },
  ];

  if (recipe.eyeScale > 1 || recipe.eyeAngle > 0) {
    const eyeSource = createCanvas(width, height);
    eyeSource.getContext("2d").drawImage(context.canvas, 0, 0);
    eyeRegions.forEach((region) => {
      drawTransformedRegion(context, eyeSource, region, {
        scaleX: recipe.eyeScale,
        scaleY: 1 + (recipe.eyeScale - 1) * 0.7,
        rotation: region.angle,
      });
    });
  }

  drawCurrentWarp(
    context,
    {
      x: width * 0.27,
      y: height * 0.29,
      width: width * 0.46,
      height: height * 0.17,
    },
    recipe.eyeDistanceScale,
    1,
  );

  const detailSource = createCanvas(width, height);
  detailSource.getContext("2d").drawImage(context.canvas, 0, 0);
  const underEyeRegions = [
    {
      x: width * 0.33,
      y: height * 0.355,
      width: width * 0.19,
      height: height * 0.105,
    },
    {
      x: width * 0.48,
      y: height * 0.355,
      width: width * 0.19,
      height: height * 0.105,
    },
  ];
  underEyeRegions.forEach((region) => {
    drawFilteredRegion(
      context,
      detailSource,
      region,
      `blur(${(1 + recipe.eyeBagStrength * 4).toFixed(
        1,
      )}px) brightness(${(1 + recipe.eyeBagStrength * 0.09).toFixed(3)})`,
      recipe.eyeBagStrength * 0.36,
      0.55,
    );
    drawFilteredRegion(
      context,
      detailSource,
      region,
      `brightness(${(1 + recipe.darkCircleStrength * 0.16).toFixed(
        3,
      )}) saturate(${(1 - recipe.darkCircleStrength * 0.06).toFixed(3)})`,
      recipe.darkCircleStrength * 0.38,
      0.52,
    );
  });

  const irisRegions = [
    {
      x: width * 0.375,
      y: height * 0.338,
      width: width * 0.11,
      height: height * 0.055,
    },
    {
      x: width * 0.515,
      y: height * 0.338,
      width: width * 0.11,
      height: height * 0.055,
    },
  ];
  irisRegions.forEach((region) => {
    drawFilteredRegion(
      context,
      detailSource,
      region,
      `contrast(${(1 + recipe.irisStrength * 0.28).toFixed(
        3,
      )}) saturate(${(1 + recipe.irisStrength * 0.2).toFixed(
        3,
      )}) brightness(${(1 + recipe.irisStrength * 0.035).toFixed(3)})`,
      recipe.irisStrength * 0.55,
      0.45,
    );
  });

  drawEyeGlow(context, width, height, recipe);
}
