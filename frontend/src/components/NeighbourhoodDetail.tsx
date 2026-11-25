import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNeighbourhoodDetail } from "../hooks/useNeighbourhoods";
import { useLiveListings, useRefreshListings } from "../hooks/useLiveListings";
import PriceTrendChart from "./PriceTrendChart";
import LiveListingsTable from "./LiveListingsTable";
import {
  formatEurCompact,
  formatEur,
  formatPsm,
  formatPctSigned,
  formatCondition,
  formatNeighbourhood,
} from "../lib/format";

interface Props {
  name: string;
}

export default function NeighbourhoodDetail({ name }: Props) {
  const { data, isLoading, error } = useNeighbourhoodDetail(name);
  const { data: liveListings, isLoading: listingsLoading } = useLiveListings(name, true);
  const refreshMutation = useRefreshListings();

  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 space-y-4">
        <div className="h-7 w-48 bg-muted rounded animate-warm-pulse" />
        <div className="grid grid-cols-3 gap-3">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-16 bg-muted/60 rounded-lg animate-warm-pulse" />
          ))}
        </div>
        <div className="h-52 bg-muted/50 rounded-lg animate-warm-pulse" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-sm text-negative">
        Failed to load details for {formatNeighbourhood(name)}.
      </div>
    );
  }

  const { stats, trend, recent_scored } = data;
  const label = formatNeighbourhood(data.name, data.display_name ?? stats.display_name);
  const medEur = stats.median_price_eur ?? stats.median_price_k * 1000;
  const minEur = stats.min_price_eur ?? stats.min_price_k * 1000;
  const maxEur = stats.max_price_eur ?? stats.max_price_k * 1000;

  return (
    <div className="space-y-4">
      {/* Unified Light Editorial Profile Card (no jarring black box) */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-primary">
              Selected Neighbourhood
            </p>
            <h2 className="font-display text-2xl font-bold mt-0.5">{label}</h2>
            <p className="text-xs text-muted-foreground mt-0.5 tabular-nums">
              Range: {formatEurCompact(minEur)} – {formatEurCompact(maxEur)}
              {stats.pct_houses != null ? ` · ${stats.pct_houses.toFixed(0)}% houses` : ""}
            </p>
          </div>
          <span className="text-xs font-semibold tabular-nums bg-muted px-2.5 py-1 rounded-full text-foreground">
            {stats.count.toLocaleString()} homes
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
          <MiniStat label="Median Price" value={formatEurCompact(medEur)} />
          <MiniStat
            label="Median €/m²"
            value={stats.median_price_sqm ? formatPsm(stats.median_price_sqm) : "—"}
          />
          <MiniStat label="Median Area" value={`${Math.round(stats.median_area ?? stats.avg_area)} m²`} />
          <MiniStat label="Avg Age" value={stats.avg_age != null ? `${Math.round(stats.avg_age)} yr` : "—"} />
          <MiniStat label="Top Condition" value={formatCondition(stats.top_condition)} />
          <MiniStat label="Top EPC" value={stats.top_energy_label ?? "—"} />
        </div>
      </div>

      {/* Continuous monthly price trend */}
      {trend.length > 0 && <PriceTrendChart data={trend} neighbourhoodName={label} />}

      {/* Live Listings OR Recent Phase-7 Scored Highlights */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h3 className="font-display text-base font-semibold">
              {liveListings && liveListings.length > 0
                ? `Live Scraped Listings (${liveListings.length})`
                : `Recent Phase-7 Value Highlights (${recent_scored?.length ?? 0})`}
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              {liveListings && liveListings.length > 0
                ? "Listings scraped in the last 7 days, ranked by fair-value gap"
                : "Latest corpus properties in this neighbourhood ranked by Phase-7 OOF value gap"}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refreshMutation.mutate()}
            disabled={refreshMutation.isPending}
          >
            <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${refreshMutation.isPending ? "animate-spin" : ""}`} />
            {refreshMutation.isPending ? "Scanning..." : "Scan Live"}
          </Button>
        </div>

        {listingsLoading ? (
          <div className="h-24 bg-muted/50 rounded-lg animate-warm-pulse" />
        ) : liveListings && liveListings.length > 0 ? (
          <LiveListingsTable listings={liveListings} />
        ) : recent_scored && recent_scored.length > 0 ? (
          <div className="divide-y divide-border/60 border border-border rounded-lg overflow-hidden">
            {recent_scored.slice(0, 8).map((item) => {
              const isDeal = item.delta_pct > 0;
              return (
                <div
                  key={item.property_id}
                  className="px-3.5 py-2.5 flex items-center justify-between gap-3 text-xs hover:bg-muted/40 transition-colors"
                >
                  <div className="min-w-0">
                    <p className="font-medium text-foreground">
                      {item.property_type || "Property"}
                      {item.livable_area ? ` · ${item.livable_area} m²` : ""}
                      {item.bedrooms != null ? ` · ${item.bedrooms} bed` : ""}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      {[
                        item.condition ? formatCondition(item.condition) : null,
                        item.energy_label ? `EPC ${item.energy_label}` : null,
                        item.date,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </p>
                  </div>
                  <div className="text-right shrink-0 tabular-nums">
                    <p className="font-semibold text-foreground">
                      Ask {formatEur(item.asking_eur)} → Fair {formatEur(item.predicted_eur)}
                    </p>
                    <p className={`font-bold ${isDeal ? "text-positive" : "text-negative"}`}>
                      {formatPctSigned(item.delta_pct)}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground py-4 text-center">
            No scored listings available for {label}.
          </p>
        )}
      </div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-muted/45 border border-border/50 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold text-foreground tabular-nums mt-0.5 truncate">{value}</p>
    </div>
  );
}
