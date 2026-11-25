import { useMemo, useState } from "react";

export type SortDir = "asc" | "desc";

export interface TableSortState<K extends string> {
  key: K;
  dir: SortDir;
}

/**
 * Lightweight column-sort hook for plain client-side tables.
 * Handles numbers, strings (locale-aware), booleans, and puts null/undefined last.
 */
export function useTableSort<T, K extends string = Extract<keyof T, string>>(
  rows: readonly T[],
  initialKey: NoInfer<K>,
  initialDir: SortDir = "desc",
  accessors?: Partial<Record<K, (row: T) => string | number | boolean | null | undefined>>,
) {
  const [sort, setSort] = useState<TableSortState<K>>({ key: initialKey, dir: initialDir });

  const toggle = (key: K, defaultDir: SortDir = "desc") => {
    setSort((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key, dir: defaultDir },
    );
  };

  const sorted = useMemo(() => {
    const out = [...rows];
    const getter = accessors?.[sort.key];
    const dirMul = sort.dir === "asc" ? 1 : -1;

    out.sort((a, b) => {
      const av = getter ? getter(a) : (a as Record<string, unknown>)[sort.key];
      const bv = getter ? getter(b) : (b as Record<string, unknown>)[sort.key];

      const aNull = av == null || av === "" || (typeof av === "number" && !Number.isFinite(av));
      const bNull = bv == null || bv === "" || (typeof bv === "number" && !Number.isFinite(bv));
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;

      if (typeof av === "number" && typeof bv === "number") {
        return (av - bv) * dirMul;
      }
      if (typeof av === "boolean" && typeof bv === "boolean") {
        return (Number(av) - Number(bv)) * dirMul;
      }
      return String(av).localeCompare(String(bv), undefined, { numeric: true, sensitivity: "base" }) * dirMul;
    });
    return out;
  }, [rows, sort, accessors]);

  return { sorted, sort, toggle };
}

/** Subtle directional glyph for sortable table column headers. */
export function sortGlyph(active: boolean, dir: SortDir): string {
  if (!active) return "↕";
  return dir === "asc" ? "↑" : "↓";
}
