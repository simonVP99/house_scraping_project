import { Fragment, useMemo, useState } from "react";
import type { FieldProfile } from "../types/eda";
import { sortGlyph, type SortDir } from "../hooks/useTableSort";

interface Props {
  fields: FieldProfile[];
}

const KIND_ORDER = ["numeric", "categorical", "boolean", "text", "date", "geo"];

function fmt(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function compact(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return fmt(n, abs < 10 ? 2 : 0);
}

// Null-rate colouring: a field that is mostly "not declared" is a data finding,
// not an error, but it must be visually impossible to miss.
function nullTone(pct: number): string {
  if (pct >= 95) return "bg-destructive/15 text-destructive border-destructive/30";
  if (pct >= 50) return "bg-amber-500/15 text-amber-600 border-amber-500/30";
  if (pct >= 25) return "bg-yellow-500/10 text-yellow-700 border-yellow-500/25";
  return "bg-emerald-500/10 text-emerald-700 border-emerald-500/25";
}

export default function EdaFieldTable({ fields }: Props) {
  const [kind, setKind] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"name" | "kind" | "non_null" | "null" | "unique" | "signal">("null");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggleSort = (col: typeof sortBy, defaultDir: SortDir = "desc") => {
    if (sortBy === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortBy(col);
      setSortDir(defaultDir);
    }
  };

  const kinds = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const f of fields) counts[f.kind] = (counts[f.kind] ?? 0) + 1;
    return counts;
  }, [fields]);

  const rows = useMemo(() => {
    let out = fields;
    if (kind !== "all") out = out.filter((f) => f.kind === kind);
    if (search.trim()) {
      const q = search.toLowerCase();
      out = out.filter((f) => f.name.toLowerCase().includes(q));
    }
    const sorted = [...out];
    const mul = sortDir === "asc" ? 1 : -1;
    if (sortBy === "name") sorted.sort((a, b) => a.name.localeCompare(b.name) * mul);
    else if (sortBy === "kind") sorted.sort((a, b) => a.kind.localeCompare(b.kind) * mul);
    else if (sortBy === "non_null") sorted.sort((a, b) => (a.non_null - b.non_null) * mul);
    else if (sortBy === "null") sorted.sort((a, b) => (a.null_pct - b.null_pct) * mul);
    else if (sortBy === "unique") sorted.sort((a, b) => ((a.n_unique ?? 0) - (b.n_unique ?? 0)) * mul);
    else
      sorted.sort(
        (a, b) =>
          (Math.abs((a.stats?.std ?? 0) / (a.stats?.mean || 1)) -
            Math.abs((b.stats?.std ?? 0) / (b.stats?.mean || 1))) *
          mul,
      );
    return sorted;
  }, [fields, kind, search, sortBy, sortDir]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter fields..."
          className="h-9 w-56 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-primary/40"
        />
        <div className="flex flex-wrap gap-1">
          <button
            onClick={() => setKind("all")}
            className={`px-2.5 py-1 text-xs rounded-md border transition-colors ${
              kind === "all"
                ? "bg-primary text-primary-foreground border-primary"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            all {fields.length}
          </button>
          {KIND_ORDER.filter((k) => kinds[k]).map((k) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              className={`px-2.5 py-1 text-xs rounded-md border transition-colors ${
                kind === k
                  ? "bg-primary text-primary-foreground border-primary"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {k} {kinds[k]}
            </button>
          ))}
        </div>
        <select
          value={sortBy}
          onChange={(e) => {
            const v = e.target.value as typeof sortBy;
            setSortBy(v);
            setSortDir(v === "name" || v === "kind" ? "asc" : "desc");
          }}
          className="h-9 rounded-md border border-border bg-background px-2 text-sm ml-auto"
        >
          <option value="null">sort: null rate</option>
          <option value="name">sort: name</option>
          <option value="kind">sort: kind</option>
          <option value="non_null">sort: declared</option>
          <option value="unique">sort: cardinality</option>
          <option value="signal">sort: variability</option>
        </select>
      </div>

      <div className="text-xs text-muted-foreground">
        {rows.length} fields shown · click a row to inspect
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm border-collapse">
          <thead className="bg-muted/50">
            <tr className="text-left">
              {(
                [
                  ["name", "Field", "left", "asc"],
                  ["kind", "Kind", "left", "asc"],
                  ["non_null", "Declared", "right", "desc"],
                  ["null", "Null %", "right", "desc"],
                  ["unique", "Unique", "right", "desc"],
                ] as const
              ).map(([key, label, align, defDir]) => (
                <th
                  key={key}
                  onClick={() => toggleSort(key, defDir)}
                  className={`px-3 py-2 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap ${
                    align === "right" ? "text-right" : ""
                  }`}
                >
                  {label}{" "}
                  <span className={`ml-0.5 text-[10px] ${sortBy === key ? "text-primary" : "opacity-35"}`}>
                    {sortGlyph(sortBy === key, sortDir)}
                  </span>
                </th>
              ))}
              <th className="px-3 py-2 font-medium text-right">Summary</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((f) => {
              const open = expanded === f.name;
              return (
                <Fragment key={f.name}>
                  <tr
                    onClick={() => setExpanded(open ? null : f.name)}
                    className="border-t border-border cursor-pointer hover:bg-muted/40 transition-colors"
                  >
                    <td className="px-3 py-1.5 font-mono text-xs">
                      <span className="text-foreground">{f.name}</span>
                      {f.is_leaky && (
                        <span className="ml-2 rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] font-sans font-medium text-destructive">
                          target-derived
                        </span>
                      )}
                      {f.flags.includes("near_empty") && !f.is_leaky && (
                        <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-[10px] font-sans text-muted-foreground">
                          near-empty
                        </span>
                      )}
                      {f.flags.includes("constant") && (
                        <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-[10px] font-sans text-muted-foreground">
                          constant
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-1.5 text-xs text-muted-foreground">
                      {f.kind}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-xs">
                      {fmt(f.non_null)}
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      <span
                        className={`inline-block rounded border px-1.5 py-0.5 text-[11px] tabular-nums ${nullTone(
                          f.null_pct,
                        )}`}
                      >
                        {f.null_pct.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-xs text-muted-foreground">
                      {fmt(f.n_unique)}
                    </td>
                    <td className="px-3 py-1.5 text-right text-xs text-muted-foreground whitespace-nowrap">
                      {f.stats
                        ? `med ${compact(f.stats.median)} · ${compact(
                            f.stats.min,
                          )}–${compact(f.stats.max)}`
                        : f.values?.length
                          ? `${f.values.length}${f.n_unique && f.n_unique > f.values.length ? "+" : ""} values`
                          : "—"}
                    </td>
                  </tr>
                  {open && (
                    <tr className="border-t border-border bg-muted/20">
                      <td colSpan={6} className="px-4 py-3">
                        <FieldDetail field={f} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FieldDetail({ field }: { field: FieldProfile }) {
  if (field.stats) {
    const s = field.stats;
    const rows: [string, string][] = [
      ["count", fmt(s.count)],
      ["mean", compact(s.mean)],
      ["std", compact(s.std)],
      ["min", fmt(s.min, 0)],
      ["p01", fmt(s.p01, 0)],
      ["p25", fmt(s.p25, 0)],
      ["median", fmt(s.median, 0)],
      ["p75", fmt(s.p75, 0)],
      ["p99", fmt(s.p99, 0)],
      ["max", fmt(s.max, 0)],
      ["zeros", fmt(s.zeros)],
      ["negatives", fmt(s.negatives)],
      ["IQR×3 outliers", fmt(s.outliers_iqr3 ?? 0)],
    ];
    return (
      <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-xs sm:grid-cols-4">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2 border-b border-border/50 py-0.5">
            <span className="text-muted-foreground">{k}</span>
            <span className="tabular-nums font-medium">{v}</span>
          </div>
        ))}
      </div>
    );
  }

  if (field.length) {
    const l = field.length;
    return (
      <div className="space-y-1 text-xs">
        <div className="grid grid-cols-2 gap-x-8 sm:grid-cols-4">
          {(
            [
              ["count", fmt(l.count)],
              ["mean chars", compact(l.mean)],
              ["median chars", compact(l.median)],
              ["max chars", fmt(l.max)],
              ["rows at 2500 cap", fmt(l.at_2500_cap)],
              ["placeholder '-'", fmt(l.placeholder_dash)],
            ] as [string, string][]
          ).map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2 border-b border-border/50 py-0.5">
              <span className="text-muted-foreground">{k}</span>
              <span className="tabular-nums font-medium">{v}</span>
            </div>
          ))}
        </div>
        <div className="text-muted-foreground">
          Free text — truncated to 80 chars in the value list below.
        </div>
        <ValueList field={field} />
      </div>
    );
  }

  return <ValueList field={field} />;
}

function ValueList({ field }: { field: FieldProfile }) {
  if (!field.values?.length) {
    return <div className="text-xs text-muted-foreground">No declared values.</div>;
  }
  const max = Math.max(...field.values.map((v) => v.pct));
  return (
    <div className="space-y-1 text-xs">
      <div className="text-muted-foreground mb-1">
        Top values (share of declared: {fmt(field.non_null)} rows)
      </div>
      {field.values.map((v, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-56 truncate font-mono text-[11px]" title={String(v.value)}>
            {String(v.value)}
          </span>
          <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
            <div
              className="h-full bg-primary/60"
              style={{ width: `${max ? (v.pct / max) * 100 : 0}%` }}
            />
          </div>
          <span className="w-24 text-right tabular-nums text-muted-foreground">
            {fmt(v.count)} · {v.pct.toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}
