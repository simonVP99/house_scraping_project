import { useEffect, useState } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import { Compass, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAffordability } from "../hooks/useInsights";
import type { AffordabilityNeighbourhood, AffordabilityQuery } from "../types/insights";
import { formatEur, formatEurCompact, formatEurSigned, formatPsm } from "../lib/format";
import "leaflet/dist/leaflet.css";

const DEFAULT_QUERY: AffordabilityQuery = {
  budget_eur: 400_000,
  property_type: "HOUSE",
  livable_area: 140,
  bedrooms: 3,
  condition: "JUST_RENOVATED",
  condition_mode: "min",
  energy_label: "B",
  epc_mode: "min",
  year_built: 1965,
  has_garden: true,
  has_terrace: true,
};

function tierColor(tier: AffordabilityNeighbourhood["tier"]): string {
  switch (tier) {
    case "comfortable":
      return "#2f7d4d"; // forest green
    case "realistic":
      return "#78a65a"; // sage green
    case "stretch":
      return "#d08a2c"; // amber
    case "out_of_reach":
      return "#b3402e"; // warm terracotta red
  }
}

function tierBadge(tier: AffordabilityNeighbourhood["tier"]): { label: string; cls: string } {
  switch (tier) {
    case "comfortable":
      return { label: "Comfortable", cls: "bg-positive/10 text-positive border-positive/25" };
    case "realistic":
      return { label: "Realistic", cls: "bg-positive/5 text-positive/90 border-positive/20" };
    case "stretch":
      return { label: "Stretch", cls: "bg-warning/10 text-warning border-warning/30" };
    case "out_of_reach":
      return { label: "Above budget", cls: "bg-negative/10 text-negative border-negative/25" };
  }
}

export default function AffordabilityPage() {
  const [query, setQuery] = useState<AffordabilityQuery>(DEFAULT_QUERY);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const mutation = useAffordability();

  // Evaluate on initial mount so the user sees live Phase-7 results immediately
  useEffect(() => {
    mutation.mutate(DEFAULT_QUERY);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const data = mutation.data;
  const rows = data?.neighbourhoods ?? [];
  const selected = rows.find((r) => r.name === selectedName) ?? rows[0] ?? null;

  const comfortableCount = rows.filter((r) => r.tier === "comfortable" || r.tier === "realistic").length;
  const stretchCount = rows.filter((r) => r.tier === "stretch").length;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate(query);
  };

  const selectClass =
    "flex h-9 w-full rounded-md border border-input bg-background px-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-primary">
            Reverse Valuation Explorer
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-bold tracking-tight mt-1">
            Where Can I Afford in Ghent?
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Set your budget and target home specs — Phase-7 Lean-5 scores all 25 neighbourhoods live and computes conformal probability{" "}
            <span className="font-mono text-xs">P(fair value ≤ budget)</span>.
          </p>
        </div>

        {data && (
          <div className="flex flex-wrap items-center gap-2 text-xs tabular-nums">
            <span className="px-3 py-1 rounded-full bg-positive/10 text-positive border border-positive/25 font-semibold">
              {comfortableCount} realistic / comfortable
            </span>
            <span className="px-3 py-1 rounded-full bg-warning/10 text-warning border border-warning/25 font-semibold">
              {stretchCount} stretch
            </span>
          </div>
        )}
      </div>

      {/* Controls Bar */}
      <form onSubmit={handleSubmit} className="rounded-xl border border-border bg-card p-5 space-y-4">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <SlidersHorizontal className="w-3.5 h-3.5 text-primary" />
            Target Home & Budget
          </div>
          <Button type="submit" size="sm" disabled={mutation.isPending}>
            {mutation.isPending ? "Scoring 25 neighbourhoods..." : "Update Map"}
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
          {/* Budget slider */}
          <div className="md:col-span-4">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <label htmlFor="budget-slider" className="font-medium text-muted-foreground">
                Max Budget
              </label>
              <span className="font-display text-lg font-bold text-primary tabular-nums">
                {formatEur(query.budget_eur)}
              </span>
            </div>
            <input
              id="budget-slider"
              type="range"
              min={175_000}
              max={900_000}
              step={10_000}
              value={query.budget_eur}
              onChange={(e) => setQuery((q) => ({ ...q, budget_eur: Number(e.target.value) }))}
              className="w-full accent-[var(--color-primary)] cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground tabular-nums">
              <span>€175k</span>
              <span>€450k</span>
              <span>€900k</span>
            </div>
          </div>

          <div className="md:col-span-2">
            <label htmlFor="aff-type" className="block text-xs font-medium text-muted-foreground mb-1">
              Type
            </label>
            <select
              id="aff-type"
              value={query.property_type}
              onChange={(e) => {
                const pt = e.target.value as "HOUSE" | "APARTMENT";
                setQuery((q) => ({
                  ...q,
                  property_type: pt,
                  livable_area: pt === "HOUSE" ? Math.max(q.livable_area, 120) : Math.min(q.livable_area, 110),
                  bedrooms: pt === "HOUSE" ? 3 : 2,
                  has_garden: pt === "HOUSE",
                }));
              }}
              className={selectClass}
            >
              <option value="HOUSE">House</option>
              <option value="APARTMENT">Apartment</option>
            </select>
          </div>

          <div className="md:col-span-2">
            <label htmlFor="aff-area" className="block text-xs font-medium text-muted-foreground mb-1">
              Area ({query.livable_area} m²)
            </label>
            <input
              id="aff-area"
              type="range"
              min={45}
              max={280}
              step={5}
              value={query.livable_area}
              onChange={(e) => setQuery((q) => ({ ...q, livable_area: Number(e.target.value) }))}
              className="w-full accent-[var(--color-primary)] cursor-pointer mt-2"
            />
          </div>

          <div className="md:col-span-2">
            <label htmlFor="aff-cond" className="block text-xs font-medium text-muted-foreground mb-1">
              Min Condition
            </label>
            <select
              id="aff-cond"
              value={query.condition}
              onChange={(e) => setQuery((q) => ({ ...q, condition: e.target.value, condition_mode: "min" }))}
              className={selectClass}
            >
              <option value="AS_NEW">As new only</option>
              <option value="JUST_RENOVATED">Just renovated +</option>
              <option value="GOOD">Good +</option>
              <option value="TO_BE_DONE_UP">To refresh +</option>
              <option value="TO_RENOVATE">To renovate +</option>
              <option value="ANY">Any condition</option>
            </select>
          </div>

          <div className="md:col-span-2">
            <label htmlFor="aff-epc" className="block text-xs font-medium text-muted-foreground mb-1">
              Min EPC Label
            </label>
            <select
              id="aff-epc"
              value={query.energy_label}
              onChange={(e) => setQuery((q) => ({ ...q, energy_label: e.target.value, epc_mode: "min" }))}
              className={selectClass}
            >
              <option value="A">EPC A+ (A / A+ / A++)</option>
              <option value="B">EPC B+ (B or better)</option>
              <option value="C">EPC C+ (C or better)</option>
              <option value="D">EPC D+ (no reno rule)</option>
              <option value="E">EPC E+</option>
              <option value="ANY">Any EPC (A–F)</option>
            </select>
          </div>
        </div>
      </form>

      {mutation.error && (
        <div className="rounded-xl border border-negative/30 bg-negative/5 p-4 text-sm text-negative">
          {mutation.error.message}
        </div>
      )}

      {/* Map + Sticky Neighbourhood Inspector Side-by-Side */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-7 rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="font-semibold text-foreground flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-primary" />
              Ghent Conformal Affordability Map
            </span>
            <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: "#2f7d4d" }} /> ≥70% Comfortable
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: "#78a65a" }} /> 40–69% Realistic
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: "#d08a2c" }} /> 18–39% Stretch
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: "#b3402e" }} /> &lt;18% Above
              </span>
            </div>
          </div>

          <div className="h-[520px] relative">
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
              {rows.map((n) => {
                if (!n.geojson) return null;
                const isSelected = selected?.name === n.name;
                return (
                  <GeoJSON
                    key={`${n.name}-${n.prob_affordable}-${isSelected}`}
                    data={n.geojson}
                    style={{
                      fillColor: tierColor(n.tier),
                      fillOpacity: isSelected ? 0.8 : 0.58,
                      color: isSelected ? "#1f1814" : "#3a2e28",
                      weight: isSelected ? 2.8 : 1.1,
                    }}
                    onEachFeature={(_feature, layer) => {
                      layer.bindTooltip(
                        `<div style="font-size:12px"><strong>${n.display_name}</strong><br/>` +
                        `Est. Fair Value: <strong>${formatEur(n.predicted_eur)}</strong><br/>` +
                        `80% Range: ${formatEurCompact(n.low_eur)} – ${formatEurCompact(n.high_eur)}<br/>` +
                        `Affordability: <strong>${n.prob_affordable.toFixed(0)}%</strong></div>`,
                        { sticky: true },
                      );
                      layer.on("click", () => setSelectedName(n.name));
                    }}
                  />
                );
              })}
            </MapContainer>
          </div>
        </div>

        {/* Right column: selected detail + ranked list */}
        <div className="lg:col-span-5 space-y-4">
          {selected && (
            <div className="rounded-xl border border-border bg-card p-5 space-y-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
                    Selected Neighbourhood
                  </p>
                  <h2 className="font-display text-2xl font-bold mt-0.5">{selected.display_name}</h2>
                </div>
                <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${tierBadge(selected.tier).cls}`}>
                  {selected.prob_affordable.toFixed(0)}% · {tierBadge(selected.tier).label}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-lg bg-muted/50 p-3">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Predicted Fair Value</p>
                  <p className="font-display text-xl font-bold tabular-nums mt-0.5">{formatEur(selected.predicted_eur)}</p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">vs Your Budget</p>
                  <p className={`font-display text-xl font-bold tabular-nums mt-0.5 ${selected.budget_gap_eur >= 0 ? "text-positive" : "text-negative"}`}>
                    {formatEurSigned(selected.budget_gap_eur)}
                  </p>
                </div>
              </div>

              <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t border-border/70 tabular-nums">
                {selected.entry_eur != null && selected.entry_eur !== selected.predicted_eur && (
                  <div className="flex justify-between">
                    <span>Entry fair value (min qualifying spec)</span>
                    <span className="font-medium text-foreground">{formatEur(selected.entry_eur)}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span>80% conformal range</span>
                  <span className="font-medium text-foreground">{formatEur(selected.low_eur)} – {formatEur(selected.high_eur)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Neighbourhood median €/m²</span>
                  <span className="font-medium text-foreground">{formatPsm(selected.median_price_sqm)}</span>
                </div>
                {selected.matching_count != null && (
                  <div className="flex justify-between">
                    <span>Matching corpus listings</span>
                    <span className="font-medium text-foreground">{selected.matching_count} of {selected.count}</span>
                  </div>
                )}
                {selected.median_dist_center_km != null && (
                  <div className="flex justify-between">
                    <span>Distance to Ghent center</span>
                    <span className="font-medium text-foreground">{selected.median_dist_center_km.toFixed(1)} km</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Ranked list */}
          <div className="rounded-xl border border-border bg-card overflow-hidden">
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Ranked Neighbourhoods ({rows.length})
              </h3>
              <span className="text-[11px] text-muted-foreground">Click to highlight on map</span>
            </div>
            <div className="max-h-[340px] overflow-y-auto divide-y divide-border/60">
              {rows.map((r) => {
                const active = selected?.name === r.name;
                return (
                  <button
                    key={r.name}
                    type="button"
                    onClick={() => setSelectedName(r.name)}
                    className={`w-full px-4 py-2.5 text-left flex items-center justify-between gap-3 hover:bg-muted/50 transition-colors cursor-pointer ${
                      active ? "bg-primary/5" : ""
                    }`}
                  >
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">{r.display_name}</p>
                      <p className="text-[11px] text-muted-foreground tabular-nums">
                        Fair {formatEurCompact(r.predicted_eur)} · 80% {formatEurCompact(r.low_eur)}–{formatEurCompact(r.high_eur)}
                      </p>
                    </div>
                    <div className="text-right shrink-0">
                      <span className={`text-xs font-bold tabular-nums px-2 py-0.5 rounded-full border ${tierBadge(r.tier).cls}`}>
                        {r.prob_affordable.toFixed(0)}%
                      </span>
                    </div>
                  </button>
                );
              })}
              {mutation.isPending && rows.length === 0 && (
                <div className="p-8 text-center text-sm text-muted-foreground">
                  Scoring 25 Ghent neighbourhoods with Phase-7 Lean-5...
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
