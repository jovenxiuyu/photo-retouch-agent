import { createCanvas } from "./canvas.mjs";
import { applyFeatherMask } from "./mask.mjs";

export function drawFilteredRegion(
  context,
  source,
  region,
  filter,
  opacity,
  featherStart = 0.68,
) {
  if (opacity <= 0) return;

  const filtered = createCanvas(source.width, source.height);
  const filteredContext = filtered.getContext("2d");
  filteredContext.filter = filter;
  filteredContext.drawImage(source, 0, 0);
  applyFeatherMask(filtered, region, featherStart, opacity);
  context.drawImage(filtered, 0, 0);
}
