import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LineChart,
  Line,
  Cell,
} from "recharts";
import { useEdaReport } from "../hooks/useEdaReport";
import { useTableSort, sortGlyph } from "../hooks/useTableSort";
import EdaFieldTable from "../components/EdaFieldTable";
import EdaSpatialMap from "../components/EdaSpatialMap";
import type { EdaReport } from "../types/eda";

const TABS = [
  "overview",
  "fields",
  "missingness",
  "target",
  "spatial",
  "temporal",
  "integrity",
] as const;
type Tab = (typeof TABS)[number];

// Recharts needs concrete colours: theme tokens here are oklch CSS custom
// properties (--color-primary), which an SVG fill attribute cannot resolve --
// passing "hsl(var(--primary))" silently renders black. Matches the hex
// constants already used in PriceTrendChart.
const C_PRIMARY = "#9a4a2a";   // warm terracotta (--color-primary)
const C_DANGER = "#b3402e";
const C_WARN = "#d08a2c";

function eur(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return `€${Math.round(n).toLocaleString()}`;
}

function compact(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

export default function EdaPage({ embedded = false }: { embedded?: boolean }) {
  const { data, isLoading, error } = useEdaReport();
  const [tab, setTab] = useState<Tab>("overview");

  if (isLoading) {
    return <div className="text-muted-foreground py-8">Loading full dataset EDA report...</div>;
  }
  if (error) {
    return (
      <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-6 text-sm">
        <div className="font-medium text-destructive mb-1">No EDA report available</div>
        <div className="text-muted-foreground">{(error as Error).message}</div>
        <div className="mt-3 font-mono text-xs text-muted-foreground">
          .venv-api/bin/python src/eda_report.py --input outputs/property_data_full.parquet
        </div>
      </div>
    );
  }
  if (!data) return null;

  return (
    <div className={embedded ? "space-y-6" : "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6"}>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          {!embedded && (
            <h1 className="font-display text-3xl font-bold tracking-tight">
              Dataset EDA
            </h1>
          )}
          <p className="text-sm text-muted-foreground">
            {data.dataset.rows.toLocaleString()} scraped snapshots ·{" "}
            {data.dataset.columns} fields · {data.dataset.priced_rows.toLocaleString()}{" "}
            usable prices
          </p>
        </div>
        <div className="text-xs text-muted-foreground text-right">
          <div>generated {new Date(data.generated_at).toLocaleString()}</div>
          <div className="font-mono">label: {data.label}</div>
        </div>
      </div>

      <div className="flex flex-wrap gap-1 border-b border-border pb-px">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-2 text-sm font-medium rounded-t-md border-b-2 transition-colors ${
              tab === t
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "overview" && <Overview data={data} />}
      {tab === "fields" && <EdaFieldTable fields={data.fields} />}
      {tab === "missingness" && <Missingness data={data} />}
      {tab === "target" && <TargetSection data={data} />}
      {tab === "spatial" && <SpatialSection data={data} />}
      {tab === "temporal" && <TemporalSection data={data} />}
      {tab === "integrity" && <IntegritySection data={data} />}
    </div>
  );
}

function Overview({ data }: { data: EdaReport }) {
  const kinds: Record<string, number> = {};
  for (const f of data.fields) kinds[f.kind] = (kinds[f.kind] ?? 0) + 1;
  const nearEmpty = data.fields.filter((f) => f.null_pct >= 95).length;
  const bands = Object.entries(data.missingness.bands);

  const top = data.correlations
    .filter((c) => !c.is_leaky && c.field !== "price_old_value")
    .slice(0, 14);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Listings" value={data.dataset.rows.toLocaleString()} />
        <Stat label="Fields" value={String(data.dataset.columns)} sub={`${nearEmpty} near-empty`} />
        <Stat
          label="Unique properties"
          value={(data.duplicates.unique_property_ids ?? 0).toLocaleString()}
          sub={`${data.duplicates.rows_per_property ?? 0} snapshots each`}
        />
        <Stat
          label="Scrape days"
          value={String(data.temporal.distinct_days ?? 0)}
          sub={`${data.temporal.min} → ${data.temporal.max}`}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Field types">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={Object.entries(kinds).map(([k, v]) => ({ kind: k, count: v }))}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
              <XAxis dataKey="kind" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Bar dataKey="count" fill={C_PRIMARY} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Null-rate distribution">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={bands.map(([band, count]) => ({ band, count }))}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
              <XAxis dataKey="band" fontSize={10} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {bands.map(([band], i) => (
                  <Cell
                    key={band}
                    fill={i >= 5 ? C_DANGER : i >= 3 ? C_WARN : C_PRIMARY}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="Strongest relationships with price (target-derived columns excluded)">
        <div className="space-y-1.5">
          {top.map((c) => {
            const max = Math.max(...top.map((x) => Math.abs(x.spearman ?? x.pearson ?? 0)));
            const v = c.spearman ?? c.pearson ?? 0;
            return (
              <div key={c.field} className="flex items-center gap-3 text-xs">
                <span className="w-52 truncate font-mono" title={c.field}>
                  {c.field}
                </span>
                <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                  <div
                    className={`h-full ${v >= 0 ? "bg-primary/70" : "bg-destructive/60"}`}
                    style={{ width: `${max ? (Math.abs(v) / max) * 100 : 0}%` }}
                  />
                </div>
                <span className="w-14 text-right tabular-nums text-muted-foreground">
                  {v.toFixed(3)}
                </span>
                <span className="w-20 text-right tabular-nums text-muted-foreground/70">
                  n={c.n.toLocaleString()}
                </span>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}

function Missingness({ data }: { data: EdaReport }) {
  const { sorted: rows, sort, toggle } = useTableSort(
    data.missingness.fields,
    "null_pct",
    "desc",
  );
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-4 text-sm">
        <div className="font-medium mb-1">Null here means &ldquo;not declared&rdquo;, not &ldquo;absent&rdquo;.</div>
        <p className="text-muted-foreground">
          Immoweb sends <code className="font-mono text-xs">null</code> for any optional
          field the agency left blank — and reveals some fields only when they are
          true. {data.missingness.rows_with_no_nulls === 0
            ? "Not a single row in this dataset has all fields filled."
            : `${data.missingness.rows_with_no_nulls} rows are fully populated.`}{" "}
          Imputing these away would destroy the signal, so they are reported instead.
        </p>
      </div>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr className="text-left">
              <th
                onClick={() => toggle("field", "asc")}
                className="px-3 py-2 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap"
              >
                Field{" "}
                <span className={`ml-0.5 text-[10px] ${sort.key === "field" ? "text-primary" : "opacity-35"}`}>
                  {sortGlyph(sort.key === "field", sort.dir)}
                </span>
              </th>
              <th
                onClick={() => toggle("null_pct", "desc")}
                className="px-3 py-2 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap"
              >
                Null rate{" "}
                <span className={`ml-0.5 text-[10px] ${sort.key === "null_pct" ? "text-primary" : "opacity-35"}`}>
                  {sortGlyph(sort.key === "null_pct", sort.dir)}
                </span>
              </th>
              <th
                onClick={() => toggle("null_count", "desc")}
                className="px-3 py-2 font-medium text-right cursor-pointer select-none hover:text-foreground whitespace-nowrap"
              >
                Missing rows{" "}
                <span className={`ml-0.5 text-[10px] ${sort.key === "null_count" ? "text-primary" : "opacity-35"}`}>
                  {sortGlyph(sort.key === "null_count", sort.dir)}
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.field} className="border-t border-border">
                <td className="px-3 py-1.5 font-mono text-xs">{r.field}</td>
                <td className="px-3 py-1.5">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-64 rounded bg-muted overflow-hidden">
                      <div
                        className={`h-full ${
                          r.null_pct >= 75
                            ? "bg-destructive/60"
                            : r.null_pct >= 25
                              ? "bg-amber-500/70"
                              : "bg-emerald-500/60"
                        }`}
                        style={{ width: `${r.null_pct}%` }}
                      />
                    </div>
                    <span className="tabular-nums text-xs text-muted-foreground">
                      {r.null_pct.toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums text-xs text-muted-foreground">
                  {r.null_count.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TargetSection({ data }: { data: EdaReport }) {
  const [group, setGroup] = useState<string>(
    Object.keys(data.target_by_group).find((k) => k === "property_subtype") ??
      Object.keys(data.target_by_group)[0] ??
      "",
  );
  const rows = data.target_by_group[group] ?? [];
  const { sorted: corr, sort: corrSort, toggle: toggleCorr } = useTableSort(
    data.correlations,
    "spearman",
    "desc",
  );

  return (
    <div className="space-y-6">
      <Panel title="Price by group">
        <select
          value={group}
          onChange={(e) => setGroup(e.target.value)}
          className="mb-3 h-9 rounded-md border border-border bg-background px-2 text-sm"
        >
          {Object.keys(data.target_by_group).map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
        <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 22)}>
          <BarChart data={rows.slice(0, 30)} layout="vertical" margin={{ left: 130 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
            <XAxis type="number" fontSize={10} tickFormatter={compact} />
            <YAxis type="category" dataKey="value" fontSize={10} width={125} />
            <Tooltip formatter={(v) => eur(typeof v === "number" ? v : null)} />
            <Bar dataKey="median" fill={C_PRIMARY} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="All correlations with price">
        <div className="max-h-[420px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="bg-muted/50 sticky top-0">
              <tr className="text-left">
                {(
                  [
                    ["field", "Field", "left", "asc"],
                    ["spearman", "Spearman", "right", "desc"],
                    ["pearson", "Pearson", "right", "desc"],
                    ["n", "n", "right", "desc"],
                  ] as const
                ).map(([key, label, align, defDir]) => (
                  <th
                    key={key}
                    onClick={() => toggleCorr(key, defDir)}
                    className={`px-2 py-1.5 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap ${
                      align === "right" ? "text-right" : ""
                    }`}
                  >
                    {label}{" "}
                    <span className={`ml-0.5 text-[10px] ${corrSort.key === key ? "text-primary" : "opacity-35"}`}>
                      {sortGlyph(corrSort.key === key, corrSort.dir)}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {corr.map((c) => (
                <tr key={c.field} className={`border-t border-border ${c.is_leaky ? "opacity-50" : ""}`}>
                  <td className="px-2 py-1 font-mono">
                    {c.field}
                    {c.is_leaky && (
                      <span className="ml-1.5 rounded bg-destructive/15 px-1 text-[10px] font-sans text-destructive">
                        leaky
                      </span>
                    )}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">{c.spearman?.toFixed(3) ?? "—"}</td>
                  <td className="px-2 py-1 text-right tabular-nums">{c.pearson?.toFixed(3) ?? "—"}</td>
                  <td className="px-2 py-1 text-right tabular-nums text-muted-foreground">
                    {c.n.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function SpatialSection({ data }: { data: EdaReport }) {
  const [metric, setMetric] = useState<"median_price" | "median_price_sqm" | "n">(
    "median_price",
  );
  const spatial = data.spatial;
  const { sorted: topPostal, sort: postalSort, toggle: togglePostal } = useTableSort(
    spatial.postal_codes,
    "median_price",
    "desc",
  );

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Listings with coords" value={spatial.n_with_coords.toLocaleString()} />
        <Stat label="Priced with coords" value={spatial.n_priced_with_coords.toLocaleString()} />
        <Stat
          label="Grid cells"
          value={String(spatial.cells.length)}
          sub={`${(spatial.grid_size_deg * 111).toFixed(2)} km cells`}
        />
      </div>

      <Panel
        title="Price geography"
        right={
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as typeof metric)}
            className="h-8 rounded-md border border-border bg-background px-2 text-xs"
          >
            <option value="median_price">median price</option>
            <option value="median_price_sqm">median €/m²</option>
            <option value="n">listing density</option>
          </select>
        }
      >
        <EdaSpatialMap cells={spatial.cells} metric={metric} bbox={spatial.bbox} />
        <p className="mt-2 text-xs text-muted-foreground">
          {spatial.cells.length} cells with ≥3 priced listings. Colour is a quantile
          scale (2nd–98th percentile) so outlier cells don&apos;t flatten the city into
          one shade. Marker size scales with listing count.
        </p>
      </Panel>

      <Panel title="Median price by postal code">
        <div className="max-h-[320px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 sticky top-0">
              <tr className="text-left">
                {(
                  [
                    ["postal_code", "Postal", "left", "asc"],
                    ["median_price", "Median", "right", "desc"],
                    ["n", "n", "right", "desc"],
                  ] as const
                ).map(([key, label, align, defDir]) => (
                  <th
                    key={key}
                    onClick={() => togglePostal(key, defDir)}
                    className={`px-2 py-1.5 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap ${
                      align === "right" ? "text-right" : ""
                    }`}
                  >
                    {label}{" "}
                    <span className={`ml-0.5 text-[10px] ${postalSort.key === key ? "text-primary" : "opacity-35"}`}>
                      {sortGlyph(postalSort.key === key, postalSort.dir)}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {topPostal.map((p) => (
                <tr key={String(p.postal_code)} className="border-t border-border">
                  <td className="px-2 py-1 font-mono text-xs">{p.postal_code}</td>
                  <td className="px-2 py-1 text-right tabular-nums">{eur(p.median_price)}</td>
                  <td className="px-2 py-1 text-right tabular-nums text-muted-foreground">
                    {p.n.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function TemporalSection({ data }: { data: EdaReport }) {
  const monthly = data.temporal.monthly;
  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-border bg-card p-4 text-sm">
        <span className="font-medium">{data.temporal.distinct_days}</span> distinct
        scrape days between <span className="font-mono">{data.temporal.min}</span> and{" "}
        <span className="font-mono">{data.temporal.max}</span>. The gaps are real
        collection pauses, not missing data.
      </div>
      <Panel title="Listings collected per month">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={monthly}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
            <XAxis dataKey="month" fontSize={10} angle={-45} textAnchor="end" height={70} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" fill={C_PRIMARY} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>
      <Panel title="Cumulative collection">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart
            data={monthly.reduce<{ month: string; cum: number }[]>((acc, m) => {
              const prev = acc.length ? acc[acc.length - 1].cum : 0;
              acc.push({ month: m.month, cum: prev + m.count });
              return acc;
            }, [])}
          >
            <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
            <XAxis dataKey="month" fontSize={10} angle={-45} textAnchor="end" height={70} />
            <YAxis fontSize={11} tickFormatter={compact} />
            <Tooltip />
            <Line type="monotone" dataKey="cum" stroke={C_PRIMARY} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  );
}

function IntegritySection({ data }: { data: EdaReport }) {
  const d = data.duplicates;
  const multi = d.address_groups_with_multiple_ids ?? 0;
  const groups = d.address_groups ?? 0;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Rows" value={(d.rows ?? 0).toLocaleString()} />
        <Stat label="Unique property IDs" value={(d.unique_property_ids ?? 0).toLocaleString()} />
        <Stat
          label="Snapshots per property"
          value={String(d.rows_per_property ?? "—")}
          sub="same listing, re-scraped"
        />
        <Stat
          label="Address groups"
          value={groups.toLocaleString()}
          sub={`${multi.toLocaleString()} hold >1 property ID`}
        />
      </div>

      <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-4 text-sm">
        <div className="font-medium mb-1">Relisting leakage</div>
        <p className="text-muted-foreground">
          {groups ? ((multi / groups) * 100).toFixed(1) : 0}% of address groups
          ({(d.ids_in_multi_id_groups ?? 0).toLocaleString()} property IDs) appear under
          more than one <code className="font-mono text-xs">property_id</code>. These are
          relistings of the same home, so a random train/test split puts the same
          property on both sides and inflates accuracy. Group-aware splitting is
          required — group by address, not by ID.
        </p>
      </div>

      <Panel title="Fields flagged as target-derived">
        <div className="space-y-1 text-xs">
          {data.fields.filter((f) => f.is_leaky).map((f) => (
            <div key={f.name} className="flex justify-between border-b border-border/50 py-1">
              <span className="font-mono">{f.name}</span>
              <span className="text-muted-foreground">
                declared {f.non_null.toLocaleString()} · {f.null_pct.toFixed(1)}% null
              </span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          These are derived from the price itself. They must never be predictors.
        </p>
      </Panel>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 font-display text-2xl font-bold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

function Panel({
  title,
  right,
  children,
}: {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="font-display text-sm font-semibold">{title}</h2>
        {right}
      </div>
      {children}
    </div>
  );
}
