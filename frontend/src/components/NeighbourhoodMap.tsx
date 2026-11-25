import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import type { NeighbourhoodSummary } from "../types/neighbourhood";
import { formatEurCompact, formatPsm, formatPctSigned, formatNeighbourhood } from "../lib/format";
import "leaflet/dist/leaflet.css";

export type MapMetric =
  | "median_price_sqm"
  | "median_price_eur"
  | "avg_listing_duration"
  | "count"
  | "pct_underpriced"
  | "median_delta_pct";

interface Props {
  neighbourhoods: NeighbourhoodSummary[];
  metric: MapMetric;
  selected: string | null;
  onSelect: (name: string) => void;
}

const TERRACOTTA_RAMP = [
  "#f4e6d8",
  "#e7c7ac",
  "#d8a37c",
  "#c2794f",
  "#a65432",
  "#7d351d",
];

function getMetricValue(n: NeighbourhoodSummary, metric: MapMetric): number | null {
  if (metric === "median_price_eur") {
    return n.median_price_eur ?? n.median_price_k * 1000;
  }
  const val = n[metric];
  return typeof val === "number" && Number.isFinite(val) ? val : null;
}

function formatMetricValue(v: number | null, metric: MapMetric): string {
  if (v == null) return "—";
  switch (metric) {
    case "median_price_sqm":
      return formatPsm(v);
    case "median_price_eur":
      return formatEurCompact(v);
    case "avg_listing_duration":
      return `${v.toFixed(0)} days`;
    case "count":
      return `${v.toLocaleString()} listings`;
    case "pct_underpriced":
      return `${v.toFixed(1)}%`;
    case "median_delta_pct":
      return formatPctSigned(v);
  }
}

function rampColor(ratio: number, diverging: boolean): string {
  const r = Math.max(0, Math.min(1, ratio));
  if (diverging) {
    // Negative (overpriced) warm red -> neutral stone -> Positive (underpriced) forest green
    const div = ["#b3402e", "#d47965", "#e8ded3", "#8cb592", "#2f7d4d"];
    return div[Math.min(div.length - 1, Math.floor(r * div.length))];
  }
  return TERRACOTTA_RAMP[Math.min(TERRACOTTA_RAMP.length - 1, Math.floor(r * TERRACOTTA_RAMP.length))];
}

export default function NeighbourhoodMap({ neighbourhoods, metric, selected, onSelect }: Props) {
  const values = neighbourhoods
    .map((n) => getMetricValue(n, metric))
    .filter((v): v is number => v != null);

  const diverging = metric === "median_delta_pct";
  const minVal = values.length ? Math.min(...values) : 0;
  const maxVal = values.length ? Math.max(...values) : 1;
  const span = maxVal - minVal || 1;
  const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 0.1);

  const toRatio = (v: number) =>
    diverging ? (v + absMax) / (2 * absMax) : (v - minVal) / span;

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="h-[480px] relative">
        <MapContainer
          center={[51.0543, 3.7174]}
          zoom={11.5}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {neighbourhoods.map((n) => {
            if (!n.geojson) return null;
            const val = getMetricValue(n, metric);
            const fill = val != null ? rampColor(toRatio(val), diverging) : "#e2dcd5";
            const isSelected = selected === n.name;
            const label = formatNeighbourhood(n.name, n.display_name);

            return (
              <GeoJSON
                key={`${n.name}-${metric}-${isSelected}`}
                data={n.geojson}
                style={{
                  fillColor: fill,
                  fillOpacity: isSelected ? 0.85 : 0.64,
                  color: isSelected ? "#1f1814" : "#4a3b32",
                  weight: isSelected ? 2.8 : 1.1,
                }}
                onEachFeature={(_feature, layer) => {
                  layer.bindTooltip(
                    `<div style="font-size:12px"><strong>${label}</strong><br/>` +
                    `${formatMetricValue(val, metric)} · ${n.count.toLocaleString()} homes</div>`,
                    { sticky: true },
                  );
                  layer.on("click", () => onSelect(n.name));
                }}
              />
            );
          })}
        </MapContainer>
      </div>

      {/* Color Scale Legend */}
      <div className="px-4 py-2.5 border-t border-border bg-muted/30 flex items-center justify-between gap-4 text-[11px] text-muted-foreground tabular-nums">
        <span>{diverging ? formatPctSigned(-absMax) : formatMetricValue(minVal, metric)}</span>
        <div className="flex-1 flex h-2 rounded-full overflow-hidden max-w-xs border border-border/50">
          {(diverging
            ? ["#b3402e", "#d47965", "#e8ded3", "#8cb592", "#2f7d4d"]
            : TERRACOTTA_RAMP
          ).map((hex) => (
            <div key={hex} className="flex-1" style={{ backgroundColor: hex }} />
          ))}
        </div>
        <span>{diverging ? formatPctSigned(absMax) : formatMetricValue(maxVal, metric)}</span>
      </div>
    </div>
  );
}
