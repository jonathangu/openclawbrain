import { createHash } from "node:crypto";

const sortedEntries = (value) => Object.entries(value).sort(([left], [right]) => left.localeCompare(right));

function canonicalize(value) {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  const result = {};
  for (const [key, child] of sortedEntries(value)) {
    const canonicalChild = canonicalize(child);
    if (canonicalChild !== undefined) {
      result[key] = canonicalChild;
    }
  }
  return result;
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

export function checksumJsonPayload(value) {
  return `sha256:${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}
