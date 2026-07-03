import { createCanvas } from "../core/canvas.mjs";
import { drawFilteredRegion } from "../core/filtering.mjs";

export function drawToneRetouch(context, width, height, recipe) {
  if (recipe.clarityStrength > 0) {
    const claritySource = createCanvas(width, height);
    claritySource.getContext("2d").drawImage(context.canvas, 0, 0);
    const clarity = createCanvas(width, height);
    const clarityContext = clarity.getContext("2d");
    clarityContext.filter = `contrast(${(
      1 +
      recipe.clarityStrength * 0.34
    ).toFixed(3)}) saturate(${(
      1 +
      recipe.clarityStrength * 0.08
    ).toFixed(3)})`;
    clarityContext.drawImage(claritySource, 0, 0);
    context.save();
    context.globalAlpha = recipe.clarityStrength * 0.34;
    context.drawImage(clarity, 0, 0);
    context.restore();
  }

  if (recipe.backgroundStrength <= 0) return;

  const foregroundSource = createCanvas(width, height);
  foregroundSource.getContext("2d").drawImage(context.canvas, 0, 0);
  const softenedBackground = createCanvas(width, height);
  const backgroundContext = softenedBackground.getContext("2d");
  backgroundContext.filter = `blur(${(
    1 +
    recipe.backgroundStrength * 9
  ).toFixed(1)}px) saturate(${(
    1 -
    recipe.backgroundStrength * 0.08
  ).toFixed(3)})`;
  backgroundContext.drawImage(foregroundSource, 0, 0);

  context.save();
  context.globalAlpha = recipe.backgroundStrength * 0.72;
  context.drawImage(softenedBackground, 0, 0);
  context.restore();

  drawFilteredRegion(
    context,
    foregroundSource,
    {
      x: width * 0.16,
      y: height * 0.07,
      width: width * 0.68,
      height: height * 0.62,
    },
    "none",
    recipe.backgroundStrength * 0.96,
    0.76,
  );
  drawFilteredRegion(
    context,
    foregroundSource,
    {
      x: width * 0.08,
      y: height * 0.43,
      width: width * 0.84,
      height: height * 0.72,
    },
    "none",
    recipe.backgroundStrength * 0.9,
    0.72,
  );
}
