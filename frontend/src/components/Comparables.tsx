import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import type { ComparablesData, Differentiator } from "../types/prediction";
import { formatEur, formatEurCompact, formatPsm, formatCondition, formatNeighbourhood } from "../lib/format";

interface Props {
  comparables: ComparablesData;
  askingPrice: number | null;
  predictedPrice?: number | null;
  livableArea?: number | null;
}

export default function Comparables({ comparables, askingPrice, predictedPrice, livableArea }: Props) {
  const {
    properties,
    median_price_k,
    median_price_per_sqm,
    price_per_sqm_distribution,
    differentiators,
    neighbourhood,
  } = comparables;

  const predictedPsm = predictedPrice && livableArea ? Math.round(predictedPrice / livableArea) : null;
  const askingPsm = askingPrice && livableArea ? Math.round(askingPrice / livableArea) : comparables.subject_price_per_sqm;
  const medianEur = median_price_k * 1000;

  return (
    <section aria-label="Market context and comparables" className="rounded-xl border border-border bg-card p-5 space-y-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h3 className="font-display text-lg font-semibold">Comparable Sales Context</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            {properties.length} nearest similar properties in {formatNeighbourhood(neighbourhood)} · median {formatEur(medianEur)}
          </p>
        </div>
        {median_price_per_sqm != null && (
          <span className="text-xs font-medium text-muted-foreground tabular-nums">
            Comp median: <strong className="text-foreground">{formatPsm(median_price_per_sqm)}</strong>
          </span>
        )}
      </div>

      {/* Price/m² distribution strip showing Fair Value AND Asking Price */}
      {median_price_per_sqm != null && price_per_sqm_distribution.length > 0 && (
        <PricePerSqmScale
          predictedPsm={predictedPsm}
          askingPsm={askingPsm}
          medianPsm={median_price_per_sqm}
          distribution={price_per_sqm_distribution}
        />
      )}

      {/* Key differences vs comparables */}
      {differentiators.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2.5">
            Key Differences vs Comparables
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {differentiators.map((d, i) => (
              <DifferentiatorPill key={i} diff={d} />
            ))}
          </div>
        </div>
      )}

      {/* Comparable table */}
      <details className="group" open={properties.length <= 6}>
        <summary className="text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground cursor-pointer select-none flex items-center gap-1.5">
          <span className="text-[10px] transition-transform group-open:rotate-90">{"\u25B6"}</span>
          Comparable properties ({properties.length})
        </summary>
        <div className="mt-3 rounded-lg border border-border overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Price</TableHead>
                <TableHead className="text-xs">€/m²</TableHead>
                <TableHead className="text-xs">Area</TableHead>
                <TableHead className="text-xs">Beds</TableHead>
                <TableHead className="text-xs">Condition</TableHead>
                <TableHead className="text-xs">EPC</TableHead>
                <TableHead className="text-xs">Age</TableHead>
                <TableHead className="text-xs text-right">Listing</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {properties.map((p, i) => (
                <TableRow key={p.property_id || i}>
                  <TableCell className="font-semibold text-sm tabular-nums">
                    {formatEur(p.price_k * 1000)}
                  </TableCell>
                  <TableCell className="text-sm tabular-nums text-muted-foreground">
                    {p.price_per_sqm ? formatPsm(p.price_per_sqm) : "—"}
                  </TableCell>
                  <TableCell className="text-sm tabular-nums">
                    {p.livable_area ? `${Math.round(p.livable_area)} m²` : "—"}
                  </TableCell>
                  <TableCell className="text-sm tabular-nums">{p.bedrooms ?? "—"}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{formatCondition(p.condition)}</TableCell>
                  <TableCell className="text-xs font-medium">{p.energy_label ?? "—"}</TableCell>
                  <TableCell className="text-sm tabular-nums text-muted-foreground">
                    {p.age != null ? `${Math.round(p.age)} yr` : "—"}
                  </TableCell>
                  <TableCell className="text-xs text-right">
                    {p.url ? (
                      <a
                        href={p.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary hover:underline font-medium"
                      >
                        Immoweb ↗
                      </a>
                    ) : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </details>
    </section>
  );
}

function PricePerSqmScale({
  predictedPsm,
  askingPsm,
  medianPsm,
  distribution,
}: {
  predictedPsm: number | null;
  askingPsm: number | null;
  medianPsm: number;
  distribution: number[];
}) {
  const all = [
    ...distribution,
    ...(predictedPsm != null ? [predictedPsm] : []),
    ...(askingPsm != null ? [askingPsm] : []),
  ];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const pad = Math.max((max - min) * 0.06, 150);
  const lo = min - pad;
  const hi = max + pad;
  const span = Math.max(hi - lo, 1);

  const pct = (v: number) => Math.max(2, Math.min(98, ((v - lo) / span) * 100));

  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-muted-foreground">
        {predictedPsm != null && (
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-primary inline-block" />
            Fair value: <strong className="text-foreground tabular-nums">{formatPsm(predictedPsm)}</strong>
          </span>
        )}
        {askingPsm != null && (
          <span className="inline-flex items-center gap-1.5">
            <span
              className={`w-2.5 h-2.5 rounded-full inline-block ${
                askingPsm <= (predictedPsm ?? medianPsm) ? "bg-positive" : "bg-negative"
              }`}
            />
            Asking: <strong className="text-foreground tabular-nums">{formatPsm(askingPsm)}</strong>
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-foreground/25 inline-block" />
          12 comps (median {formatPsm(medianPsm)})
        </span>
      </div>

      <div className="relative h-7 bg-muted/60 rounded-full border border-border/60">
        {distribution.map((v, i) => (
          <div
            key={i}
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-foreground/20"
            style={{ left: `${pct(v)}%` }}
            title={formatPsm(v)}
          />
        ))}
        <div
          className="absolute top-1 bottom-1 w-0.5 bg-foreground/35 -translate-x-1/2"
          style={{ left: `${pct(medianPsm)}%` }}
          title={`Comp median ${formatPsm(medianPsm)}`}
        />
        {predictedPsm != null && (
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full bg-primary border-2 border-card shadow-sm z-10"
            style={{ left: `${pct(predictedPsm)}%` }}
            title={`Fair value ${formatPsm(predictedPsm)}`}
          />
        )}
        {askingPsm != null && (
          <div
            className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full border-2 border-card shadow-sm z-20 ${
              askingPsm <= (predictedPsm ?? medianPsm) ? "bg-positive" : "bg-negative"
            }`}
            style={{ left: `${pct(askingPsm)}%` }}
            title={`Asking ${formatPsm(askingPsm)}`}
          />
        )}
      </div>

      <div className="flex justify-between text-[10px] text-muted-foreground tabular-nums">
        <span>{formatEurCompact(lo)}/m²</span>
        <span>{formatEurCompact((lo + hi) / 2)}/m²</span>
        <span>{formatEurCompact(hi)}/m²</span>
      </div>
    </div>
  );
}

function DifferentiatorPill({ diff }: { diff: Differentiator }) {
  const isPositive = diff.impact === "positive";
  return (
    <div
      className={`flex items-start gap-2 rounded-lg border px-3 py-2.5 text-xs ${
        isPositive ? "border-positive/20 bg-positive/5" : "border-warning/25 bg-warning/5"
      }`}
    >
      <span className={`font-bold mt-0.5 ${isPositive ? "text-positive" : "text-warning"}`}>
        {isPositive ? "\u2191" : "\u2193"}
      </span>
      <div>
        <span className="font-semibold text-foreground">{diff.feature}: {String(diff.subject)}</span>
        <p className="text-muted-foreground mt-0.5">
          {diff.direction} than comp avg ({String(diff.comparable_avg)})
        </p>
      </div>
    </div>
  );
}
