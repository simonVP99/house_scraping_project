import { useState, useMemo } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import type { GeoJsonObject } from "geojson";
import type { NeighbourhoodSummary } from "../types/neighbourhood";
import "leaflet/dist/leaflet.css";

interface Props {
  neighbourhoods: NeighbourhoodSummary[];
  onSelect: (name: string) => void;
}

type MetricKey = "median_price_sqm" | "avg_listing_duration" | "count" | "pct_underpriced" | "median_delta_pct";

const METRICS: { key: MetricKey; label: string; unit: string }[] = [
  { key: "median_price_sqm", label: "Median Price/m\u00B2", unit: "\u20AC/m\u00B2" },
  { key: "avg_listing_duration", label: "Avg Duration", unit: "days" },
  { key: "count", label: "Listings", unit: "" },
  { key: "pct_underpriced", label: "% Underpriced", unit: "%" },
  { key: "median_delta_pct", label: "Median Delta", unit: "%" },
];

function interpolateColor(value: number, min: number, max: number): string {
  const t = max === min ? 0.5 : (value - min) / (max - min);
  const r = Math.round(255 * (1 - t) + 59 * t);
  const g = Math.round(255 * (1 - t) + 130 * t);
  const b = Math.round(178 * (1 - t) + 246 * t);
  return `rgb(${r},${g},${b})`;
}

export default function NeighbourhoodMap({ neighbourhoods, onSelect }: Props) {
  const [metric, setMetric] = useState<MetricKey>("median_price_sqm");

  const { min, max } = useMemo(() => {
    const values = neighbourhoods
      .map((n) => n[metric])
      .filter((v): v is number => v != null);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [neighbourhoods, metric]);

  return (
    <div className="space-y-3">
      <div className="flex gap-2 flex-wrap">
        {METRICS.map((m) => (
          <button
            key={m.key}
            onClick={() => setMetric(m.key)}
            className={`px-3 py-1.5 text-sm rounded-md border ${
              metric === m.key
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-slate-700 border-slate-200 hover:border-slate-300"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="rounded-lg overflow-hidden border" style={{ height: 500 }}>
        <MapContainer
          center={[51.054, 3.725]}
          zoom={13}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />
          {neighbourhoods.map((nbh) => {
            if (!nbh.geojson) return null;
            const value = nbh[metric];
            const color = value != null ? interpolateColor(value, min, max) : "#ccc";

            return (
              <GeoJSON
                key={`${nbh.name}-${metric}`}
                data={nbh.geojson as GeoJsonObject}
                style={() => ({
                  fillColor: color,
                  color: "#333",
                  weight: 1.5,
                  fillOpacity: 0.65,
                })}
                eventHandlers={{
                  click: () => onSelect(nbh.name),
                }}
                onEachFeature={(_feature, layer) => {
                  const unit = METRICS.find((m) => m.key === metric)?.unit ?? "";
                  layer.bindTooltip(
                    `<b>${nbh.name}</b><br/>${value?.toFixed(1) ?? "N/A"} ${unit}`
                  );
                }}
              />
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}
