export const clamp = (value, min, max) =>
  Math.min(max, Math.max(min, value));

export const adjustmentAmount = (values, key) =>
  clamp(Number(values[key]) || 0, 0, 100);
