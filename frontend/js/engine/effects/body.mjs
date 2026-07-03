import { drawCurrentWarp } from "../geometry/warp.mjs";

export function drawBodyRetouch(context, width, height, recipe) {
  drawCurrentWarp(
    context,
    {
      x: width * 0.14,
      y: height * 0.06,
      width: width * 0.72,
      height: height * 0.58,
    },
    recipe.headRatioScale,
    recipe.headRatioScale,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.08,
      y: height * 0.46,
      width: width * 0.84,
      height: height * 0.23,
    },
    recipe.shoulderScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.17,
      y: height * 0.54,
      width: width * 0.66,
      height: height * 0.35,
    },
    recipe.bellyScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.2,
      y: height * 0.62,
      width: width * 0.6,
      height: height * 0.27,
    },
    recipe.waistScaleX,
    1,
  );
  [
    {
      x: width * 0.035,
      y: height * 0.52,
      width: width * 0.28,
      height: height * 0.46,
    },
    {
      x: width * 0.685,
      y: height * 0.52,
      width: width * 0.28,
      height: height * 0.46,
    },
  ].forEach((region) => {
    drawCurrentWarp(context, region, recipe.armScaleX, 1);
  });
  drawCurrentWarp(
    context,
    {
      x: width * 0.16,
      y: height * 0.72,
      width: width * 0.68,
      height: height * 0.25,
    },
    recipe.hipScaleX,
    recipe.hipScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.15,
      y: height * 0.73,
      width: width * 0.7,
      height: height * 0.36,
    },
    recipe.legScaleX,
    recipe.legScaleY,
  );
}
