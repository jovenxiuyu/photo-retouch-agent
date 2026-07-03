export function applyFeatherMask(
  canvas,
  region,
  featherStart = 0.82,
  opacity = 1,
) {
  const context = canvas.getContext("2d");
  const radiusX = region.width / 2;
  const radiusY = region.height / 2;

  context.save();
  context.globalCompositeOperation = "destination-in";
  context.translate(region.x + radiusX, region.y + radiusY);
  context.scale(radiusX, radiusY);
  const gradient = context.createRadialGradient(0, 0, 0, 0, 0, 1);
  gradient.addColorStop(0, `rgba(0, 0, 0, ${opacity})`);
  gradient.addColorStop(featherStart, `rgba(0, 0, 0, ${opacity})`);
  gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
  context.fillStyle = gradient;
  context.fillRect(-1, -1, 2, 2);
  context.restore();
}
