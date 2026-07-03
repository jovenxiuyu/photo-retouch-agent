import { createCanvas } from "../core/canvas.mjs";
import { clamp } from "../core/math.mjs";
import { applyFeatherMask } from "../core/mask.mjs";

export function mapWarpPosition(position, scale) {
  const normalized = clamp(position, 0, 1) * 2 - 1;
  const warped =
    normalized *
    (scale + (1 - scale) * normalized * normalized);
  return (warped + 1) / 2;
}

export function drawAxisWarp(context, source, region, scale, axis) {
  const horizontal = axis === "x";
  const dimension = horizontal ? region.width : region.height;
  const sliceCount = clamp(Math.ceil(dimension / 6), 48, 96);

  for (let index = 0; index < sliceCount; index += 1) {
    const start = index / sliceCount;
    const end = (index + 1) / sliceCount;
    const sourceStart =
      (horizontal ? region.x : region.y) + dimension * start;
    const sourceEnd =
      (horizontal ? region.x : region.y) + dimension * end;
    const destinationStart =
      (horizontal ? region.x : region.y) +
      dimension * mapWarpPosition(start, scale);
    const destinationEnd =
      (horizontal ? region.x : region.y) +
      dimension * mapWarpPosition(end, scale);

    if (horizontal) {
      context.drawImage(
        source,
        sourceStart,
        region.y,
        sourceEnd - sourceStart,
        region.height,
        destinationStart,
        region.y,
        destinationEnd - destinationStart + 0.6,
        region.height,
      );
    } else {
      context.drawImage(
        source,
        region.x,
        sourceStart,
        region.width,
        sourceEnd - sourceStart,
        region.x,
        destinationStart,
        region.width,
        destinationEnd - destinationStart + 0.6,
      );
    }
  }
}

export function drawScaledRegion(
  context,
  source,
  region,
  scaleX,
  scaleY,
) {
  if (Math.abs(scaleX - 1) < 0.001 && Math.abs(scaleY - 1) < 0.001) return;

  let warped = source;

  if (Math.abs(scaleX - 1) >= 0.001) {
    const horizontalWarp = createCanvas(source.width, source.height);
    drawAxisWarp(
      horizontalWarp.getContext("2d"),
      source,
      region,
      scaleX,
      "x",
    );
    warped = horizontalWarp;
  }

  if (Math.abs(scaleY - 1) >= 0.001) {
    const verticalWarp = createCanvas(source.width, source.height);
    drawAxisWarp(
      verticalWarp.getContext("2d"),
      warped,
      region,
      scaleY,
      "y",
    );
    warped = verticalWarp;
  }

  applyFeatherMask(warped, region);
  context.drawImage(warped, 0, 0);
}

export function drawCurrentWarp(context, region, scaleX, scaleY) {
  if (Math.abs(scaleX - 1) < 0.001 && Math.abs(scaleY - 1) < 0.001) return;

  const snapshot = createCanvas(context.canvas.width, context.canvas.height);
  snapshot.getContext("2d").drawImage(context.canvas, 0, 0);
  drawScaledRegion(context, snapshot, region, scaleX, scaleY);
}

export function drawTransformedRegion(
  context,
  source,
  region,
  {
    scaleX = 1,
    scaleY = 1,
    rotation = 0,
    opacity = 1,
  } = {},
) {
  if (
    Math.abs(scaleX - 1) < 0.001 &&
    Math.abs(scaleY - 1) < 0.001 &&
    Math.abs(rotation) < 0.001
  ) {
    return;
  }

  const layer = createCanvas(source.width, source.height);
  const layerContext = layer.getContext("2d");
  const centerX = region.x + region.width / 2;
  const centerY = region.y + region.height / 2;
  layerContext.save();
  layerContext.translate(centerX, centerY);
  layerContext.rotate(rotation);
  layerContext.scale(scaleX, scaleY);
  layerContext.drawImage(
    source,
    region.x,
    region.y,
    region.width,
    region.height,
    -region.width / 2,
    -region.height / 2,
    region.width,
    region.height,
  );
  layerContext.restore();
  applyFeatherMask(layer, region, 0.62, opacity);
  context.drawImage(layer, 0, 0);
}
