import { createCanvas } from "../core/canvas.mjs";
import { drawFilteredRegion } from "../core/filtering.mjs";

export function drawSkinRetouch(context, width, height, recipe) {
  const skinSource = createCanvas(width, height);
  skinSource.getContext("2d").drawImage(context.canvas, 0, 0);
  drawFilteredRegion(
    context,
    skinSource,
    {
      x: width * 0.27,
      y: height * 0.2,
      width: width * 0.46,
      height: height * 0.38,
    },
    `brightness(${(1 + recipe.skinToneStrength * 0.06).toFixed(3)}) saturate(${(
      1 +
      recipe.skinToneStrength * 0.1
    ).toFixed(3)}) sepia(${(recipe.skinToneStrength * 0.035).toFixed(3)})`,
    recipe.skinToneStrength * 0.52,
    0.72,
  );
  drawFilteredRegion(
    context,
    skinSource,
    {
      x: width * 0.29,
      y: height * 0.22,
      width: width * 0.42,
      height: height * 0.34,
    },
    `blur(${recipe.smoothingRadius.toFixed(1)}px)`,
    recipe.smoothingOpacity,
  );
  drawFilteredRegion(
    context,
    skinSource,
    {
      x: width * 0.3,
      y: height * 0.25,
      width: width * 0.4,
      height: height * 0.31,
    },
    `blur(${recipe.acneRadius.toFixed(1)}px) contrast(0.985)`,
    recipe.acneOpacity,
    0.62,
  );
  const wrinkleRegions = [
    {
      x: width * 0.32,
      y: height * 0.21,
      width: width * 0.36,
      height: height * 0.15,
    },
    {
      x: width * 0.31,
      y: height * 0.33,
      width: width * 0.18,
      height: height * 0.1,
    },
    {
      x: width * 0.51,
      y: height * 0.33,
      width: width * 0.18,
      height: height * 0.1,
    },
  ];
  wrinkleRegions.forEach((region) => {
    drawFilteredRegion(
      context,
      skinSource,
      region,
      `blur(${recipe.wrinkleRadius.toFixed(1)}px)`,
      recipe.wrinkleOpacity,
      0.58,
    );
  });
}

export function drawSculptLight(context, width, height, recipe) {
  if (recipe.sculpt <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const highlight = context.createRadialGradient(
    width * 0.48,
    height * 0.38,
    0,
    width * 0.48,
    height * 0.38,
    width * 0.2,
  );
  highlight.addColorStop(0, `rgba(255, 244, 229, ${recipe.sculpt * 0.2})`);
  highlight.addColorStop(1, "rgba(255, 244, 229, 0)");
  context.fillStyle = highlight;
  context.fillRect(width * 0.25, height * 0.18, width * 0.5, height * 0.48);

  const shadowOpacity = recipe.sculpt * 0.16;
  [
    { x: width * 0.34, y: height * 0.41 },
    { x: width * 0.66, y: height * 0.41 },
  ].forEach((point) => {
    const shadow = context.createRadialGradient(
      point.x,
      point.y,
      0,
      point.x,
      point.y,
      width * 0.13,
    );
    shadow.addColorStop(0, `rgba(78, 48, 38, ${shadowOpacity})`);
    shadow.addColorStop(1, "rgba(78, 48, 38, 0)");
    context.fillStyle = shadow;
    context.fillRect(
      point.x - width * 0.15,
      point.y - height * 0.11,
      width * 0.3,
      height * 0.22,
    );
  });
  context.restore();
}
