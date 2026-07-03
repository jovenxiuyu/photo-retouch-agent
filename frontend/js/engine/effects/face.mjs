import { drawCurrentWarp } from "../geometry/warp.mjs";

export function drawFaceRetouch(context, width, height, recipe) {
  drawCurrentWarp(
    context,
    {
      x: width * 0.16,
      y: height * 0.09,
      width: width * 0.68,
      height: height * 0.59,
    },
    recipe.faceScaleX,
    recipe.faceScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.22,
      y: height * 0.29,
      width: width * 0.56,
      height: height * 0.26,
    },
    recipe.cheekScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.24,
      y: height * 0.37,
      width: width * 0.52,
      height: height * 0.25,
    },
    recipe.jawScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.27,
      y: height * 0.14,
      width: width * 0.46,
      height: height * 0.22,
    },
    1,
    recipe.foreheadScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.34,
      y: height * 0.45,
      width: width * 0.32,
      height: height * 0.18,
    },
    1,
    recipe.chinScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.29,
      y: height * 0.5,
      width: width * 0.42,
      height: height * 0.18,
    },
    1,
    recipe.doubleChinScaleY,
  );
}
