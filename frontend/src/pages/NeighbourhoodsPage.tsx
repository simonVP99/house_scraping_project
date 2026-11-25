import { useState } from "react";
import { useStats } from "../hooks/useStats";
import { useNeighbourhoods } from "../hooks/useNeighbourhoods";
import KpiStrip from "../components/KpiStrip";
import NeighbourhoodMap, { type MapMetric } from "../components/NeighbourhoodMap";
import ComparisonTable from "../components/ComparisonTable";
import NeighbourhoodDetail from "../components/NeighbourhoodDetail";

const METRIC_OPTIONS: { value: MapMetric; label: string }[] = [
  { value: "median_price_sqm", label: "Median €/m²" },
  { value: "median_price_eur", label: "Median Price" },
  { value: "median_delta_pct", label: "Phase-7 Fair Δ" },
  { value: "pct_underpriced", label: "% Below Fair" },
  { value: "avg_listing_duration", label: "Avg Stay" },
  { value: "count", label: "Volume" },
];

export default function NeighbourhoodsPage() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data: neighbourhoods, isLoading: nbhLoading } = useNeighbourhoods();
  const [metric, setMetric] = useState<MapMetric>("median_price_sqm");
  const [selected, setSelected] = useState<string | null>("Binnenstad");

  const activeName =
    selected && neighbourhoods?.some((n) => n.name === selected)
      ? selected
      : neighbourhoods?.[0]?.name ?? null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-widest text-primary">
          Ghent Spatial Analytics · Phase-7 OOF Corpus
        </p>
        <h1 className="font-display text-3xl md:text-4xl font-bold tracking-tight mt-1">
          Neighbourhood Market Explorer
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          All 25 mapped Ghent wijken powered by the clean 14,599-property dataset and out-of-fold Phase-7 fair-value scores
        </p>
      </div>

      {/* KPI Strip */}
      {statsLoading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-24 rounded-xl bg-muted animate-warm-pulse" />
          ))}
        </div>
      ) : (
        stats && <KpiStrip stats={stats} />
      )}

      {/* Metric selector */}
      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Map metric">
        {METRIC_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => setMetric(opt.value)}
            className={`px-3 py-1.5 text-xs font-medium rounded-full border transition-colors cursor-pointer ${
              metric === opt.value
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-card text-muted-foreground border-border hover:text-foreground hover:bg-muted/60"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Side-by-side Map + Instant Neighbourhood Detail Inspector */}
      {nbhLoading ? (
        <div className="h-[500px] rounded-xl bg-muted animate-warm-pulse" />
      ) : (
        neighbourhoods && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-7 lg:sticky lg:top-20">
              <NeighbourhoodMap
                neighbourhoods={neighbourhoods}
                metric={metric}
                selected={activeName}
                onSelect={setSelected}
              />
            </div>
            <div className="lg:col-span-5">
              {activeName && <NeighbourhoodDetail name={activeName} />}
            </div>
          </div>
        )
      )}

      {/* Full sortable comparison table */}
      {neighbourhoods && (
        <ComparisonTable
          neighbourhoods={neighbourhoods}
          selected={activeName}
          onSelect={(name) => {
            setSelected(name);
            window.scrollTo({ top: 180, behavior: "smooth" });
          }}
        />
      )}
    </div>
  );
}
