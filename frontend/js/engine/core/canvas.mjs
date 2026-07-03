export function createCanvas(
  width,
  height,
  documentRef = globalThis.document,
) {
  if (!documentRef?.createElement) {
    throw new Error("Canvas creation requires a document.");
  }
  const canvas = documentRef.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}
