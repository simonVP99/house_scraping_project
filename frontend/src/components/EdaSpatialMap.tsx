import { useMemo } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import type { SpatialCell } from "../types/eda";
import "leaflet/dist/leaflet.css";

interface Props {
  cells: SpatialCell[];
  metric: "median_price" | "median_price_sqm" | "n";
  bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
}

// Viridis-ish ramp: perceptually ordered and colour-blind safe, so "darker =
// more expensive" survives greyscale printing and CVD.
const RAMP = [
  "#440154", "#46327e", "#365c8d", "#277f8e", "#1fa187",
  "#4ac16d", "#a0da39", "#fde725",
];

function colourFor(t: number): string {
  const idx = Math.min(RAMP.length - 1, Math.max(0, Math.floor(t * RAMP.length)));
  return RAMP[idx];
}

export default function EdaSpatialMap({ cells, metric, bbox }: Props) {
  const { bandFor, centre } = useMemo(() => {
    const values = cells
      .map((c) => c[metric])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
      .sort((a, b) => a - b);
    // Rank-based (quantile) colouring, not min/max.  Property prices are
    // right-skewed -- median 435k against a 2.6M max -- so a linear ramp put
    // 73% of cells in the bottom two bands and the city rendered uniformly
    // dark.  Ranking gives every band an equal share of cells, which is what
    // makes spatial *pattern* readable.
    const n = values.length;
    const lookup = new Map<number, number>();
    values.forEach((v, i) => {
      if (!lookup.has(v)) lookup.set(v, n > 1 ? i / (n - 1) : 0.5);
    });
    return {
      bandFor: (v: number) => lookup.get(v) ?? 0.5,
      centre: {
        lat: (bbox.lat_min + bbox.lat_max) / 2,
        lon: (bbox.lon_min + bbox.lon_max) / 2,
      },
    };
  }, [cells, metric, bbox]);

  return (
    <MapContainer
      center={[centre.lat, centre.lon]}
      zoom={12}
      style={{ height: "520px", width: "100%" }}
      className="rounded-lg border border-border"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {cells.map((cell) => {
        const value = cell[metric];
        const t = typeof value === "number" ? bandFor(value) : 0.5;
        return (
          <CircleMarker
            key={`${cell.gx}-${cell.gy}`}
            center={[cell.lat, cell.lon]}
            radius={Math.min(13, 2.5 + Math.sqrt(cell.n) / 2.5)}
            pathOptions={{
              color: colourFor(Math.max(0, Math.min(1, t))),
              fillColor: colourFor(Math.max(0, Math.min(1, t))),
              fillOpacity: 0.72,
              weight: 1,
            }}
          >
            <Tooltip>
              <div className="text-xs leading-relaxed">
                <div className="font-semibold">
                  {typeof cell.median_price === "number"
                    ? `€${Math.round(cell.median_price).toLocaleString()}`
                    : "—"}{" "}
                  median
                </div>
                <div>n = {cell.n.toLocaleString()} listings</div>
                {cell.median_price_sqm ? (
                  <div>
                    €{Math.round(cell.median_price_sqm).toLocaleString()}/m²
                  </div>
                ) : null}
              </div>
            </Tooltip>          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
