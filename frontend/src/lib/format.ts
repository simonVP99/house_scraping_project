/**
 * Unified currency, percentage, and label formatting across the Ghent RE UI.
 * All prices are formatted consistently in EUR (€425,000 / €425k / €3,420/m²).
 */

export function formatEur(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `€${Math.round(value).toLocaleString("en-US")}`;
}

export function formatEurCompact(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1_000_000) {
    return `${sign}€${(abs / 1_000_000).toFixed(2)}M`;
  }
  if (abs >= 1_000) {
    return `${sign}€${Math.round(abs / 1_000)}k`;
  }
  return `${sign}€${Math.round(abs)}`;
}

export function formatEurSigned(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const rounded = Math.round(value);
  const prefix = rounded > 0 ? "+" : rounded < 0 ? "-" : "";
  return `${prefix}€${Math.abs(rounded).toLocaleString("en-US")}`;
}

export function formatPsm(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `€${Math.round(value).toLocaleString("en-US")}/m²`;
}

export function formatPctSigned(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

export function formatCondition(code: string | null | undefined): string {
  if (!code) return "—";
  const map: Record<string, string> = {
    AS_NEW: "As new",
    JUST_RENOVATED: "Just renovated",
    GOOD: "Good",
    TO_BE_DONE_UP: "To refresh",
    TO_RENOVATE: "To renovate",
    TO_RESTORE: "To restore",
    FIRST_TIME_USED: "First-time use",
  };
  return map[code] ?? code.replace(/_/g, " ").toLowerCase();
}

export function formatNeighbourhood(name: string | null | undefined, displayName?: string | null): string {
  if (displayName) return displayName;
  if (!name) return "Ghent";
  return name
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .trim();
}
