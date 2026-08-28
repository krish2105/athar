/** Number formatting. Kept in one place so a figure reads the same everywhere. */

export const brl = (value: number, digits = 0) =>
  new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "BRL",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);

export const aed = (value: number, digits = 0) =>
  `AED ${new Intl.NumberFormat("en-GB", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value)}`;

export const count = (value: number, digits = 0) =>
  new Intl.NumberFormat("en-GB", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);

export const pct = (value: number, digits = 1) =>
  `${(value * 100).toFixed(digits)}%`;

/** Signed percentage, for an error or a gap where direction is the point. */
export const signedPct = (value: number, digits = 0) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(digits)}%`;

export const ratio = (value: number, digits = 2) => `${value.toFixed(digits)}×`;

export const roi = (value: number, digits = 2) => value.toFixed(digits);
