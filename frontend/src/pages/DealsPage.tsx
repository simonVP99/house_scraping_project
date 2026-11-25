import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ExternalLink, RefreshCw, Sparkles, SlidersHorizontal, ArrowDownUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useDeals } from "../hooks/useInsights";
import { useNeighbourhoods } from "../hooks/useNeighbourhoods";
import { useRefreshListings } from "../hooks/useLiveListings";
import type { DealItem, DealsQuery } from "../types/insights";
import {
  formatCondition,
  formatEur,
  formatEurCompact,
  formatEurSigned,
  formatPctSigned,
  formatPsm,
} from "../lib/format";

type SortKey = "delta_pct" | "delta_eur" | "asking_price" | "predicted_price" | "livable_area";

export default function DealsPage() {
  const [neighbourhood, setNeighbourhood] = useState("ALL");
  const [propertyType, setPropertyType] = useState("ALL");
  const [maxPrice, setMaxPrice] = useState<number>(650_000);
  const [minEpc, setMinEpc] = useState("ANY");
  const [minDelta, setMinDelta] = useState<number>(5);
  const [belowBandOnly, setBelowBandOnly] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("delta_pct");
  const [sortDesc, setSortDesc] = useState(true);

  const queryParams: DealsQuery = useMemo(
    () => ({
      neighbourhood,
      property_type: propertyType,
      max_price: maxPrice,
      min_epc: minEpc,
      min_delta_pct: minDelta,
      below_band_only: belowBandOnly,
      limit: 200,
    }),
    [neighbourhood, propertyType, maxPrice, minEpc, minDelta, belowBandOnly],
  );

  const { data, isLoading, error } = useDeals(queryParams);
  const { data: neighbourhoods } = useNeighbourhoods();
  const refresh = useRefreshListings();
  const qc = useQueryClient();

  const deals = useMemo(() => {
    const items = [...(data?.deals ?? [])];
    items.sort((a, b) => {
      const av = Number(a[sortKey] ?? -Infinity);
      const bv = Number(b[sortKey] ?? -Infinity);
      return sortDesc ? bv - av : av - bv;
    });
    return items;
  }, [data, sortKey, sortDesc]);

  const belowBandCount = useMemo(
    () => deals.filter((d) => d.band_position === "below_band").length,
    [deals],
  );

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDesc((d) => !d);
    else {
      setSortKey(key);
      setSortDesc(key !== "asking_price");
    }
  };

  const handleRefresh = () => {
    refresh.mutate(undefined, {
      onSuccess: () => {
        qc.invalidateQueries({ queryKey: ["deals"] });
      },
    });
  };

  const selectClass =
    "flex h-9 w-full rounded-md border border-input bg-background px-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring";

  const isLive = data?.source === "live_scrape";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-primary">
            City-Wide Phase-7 Deal Scanner
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-bold tracking-tight mt-1">
            Ghent Deal Hunter
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Every Ghent listing ranked by agency-blind Phase-7 fair value gap and 80% conformal band position.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {data && (
            <span
              className={`text-xs font-medium px-3 py-1 rounded-full border ${
                isLive
                  ? "bg-positive/10 text-positive border-positive/25"
                  : "bg-muted text-muted-foreground border-border"
              }`}
            >
              {isLive
                ? `Live scrape · ${data.date_used}`
                : `Recent Phase-7 OOF corpus · through ${data.date_used ?? "latest"}`}
            </span>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refresh.isPending}
          >
            <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${refresh.isPending ? "animate-spin" : ""}`} />
            {refresh.isPending ? "Scanning..." : "Scan Live HTML"}
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <SlidersHorizontal className="w-3.5 h-3.5 text-primary" />
            Deal Filters
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs tabular-nums">
            <span className="text-muted-foreground">
              Showing <strong className="text-foreground">{deals.length}</strong>
              {data && data.total > deals.length ? ` of ${data.total}` : ""} listings
            </span>
            <span className="px-2.5 py-0.5 rounded-full bg-positive/10 text-positive border border-positive/25 font-semibold">
              {belowBandCount} below 80% fair band
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 items-end">
          <div>
            <label htmlFor="deal-nbh" className="block text-xs font-medium text-muted-foreground mb-1">
              Neighbourhood
            </label>
            <select
              id="deal-nbh"
              value={neighbourhood}
              onChange={(e) => setNeighbourhood(e.target.value)}
              className={selectClass}
            >
              <option value="ALL">All 25 Ghent wijken</option>
              {(neighbourhoods ?? []).map((n) => (
                <option key={n.name} value={n.name}>
                  {n.display_name || n.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="deal-type" className="block text-xs font-medium text-muted-foreground mb-1">
              Type
            </label>
            <select
              id="deal-type"
              value={propertyType}
              onChange={(e) => setPropertyType(e.target.value)}
              className={selectClass}
            >
              <option value="ALL">Houses & Apartments</option>
              <option value="HOUSE">Houses only</option>
              <option value="APARTMENT">Apartments only</option>
            </select>
          </div>

          <div>
            <label htmlFor="deal-budget" className="block text-xs font-medium text-muted-foreground mb-1">
              Max Ask ({formatEurCompact(maxPrice)})
            </label>
            <input
              id="deal-budget"
              type="range"
              min={200_000}
              max={900_000}
              step={25_000}
              value={maxPrice}
              onChange={(e) => setMaxPrice(Number(e.target.value))}
              className="w-full accent-[var(--color-primary)] cursor-pointer mt-2"
            />
          </div>

          <div>
            <label htmlFor="deal-epc" className="block text-xs font-medium text-muted-foreground mb-1">
              Min EPC Label
            </label>
            <select
              id="deal-epc"
              value={minEpc}
              onChange={(e) => setMinEpc(e.target.value)}
              className={selectClass}
            >
              <option value="ANY">Any EPC</option>
              <option value="A">EPC A+</option>
              <option value="B">EPC B+</option>
              <option value="C">EPC C+</option>
              <option value="D">EPC D+ (no reno rule)</option>
            </select>
          </div>

          <div>
            <label htmlFor="deal-delta" className="block text-xs font-medium text-muted-foreground mb-1">
              Min Discount ({minDelta >= 0 ? `+${minDelta}%` : `${minDelta}%`})
            </label>
            <select
              id="deal-delta"
              value={minDelta}
              onChange={(e) => setMinDelta(Number(e.target.value))}
              className={selectClass}
            >
              <option value={-100}>All listings</option>
              <option value={0}>Below fair value (≥0%)</option>
              <option value={5}>Undervalued ≥5%</option>
              <option value={10}>Strong deal ≥10%</option>
              <option value={15}>Deep discount ≥15%</option>
            </select>
          </div>

          <div className="flex items-center h-9">
            <label className="inline-flex items-center gap-2 text-xs font-medium text-foreground cursor-pointer select-none">
              <input
                type="checkbox"
                checked={belowBandOnly}
                onChange={(e) => setBelowBandOnly(e.target.checked)}
                className="rounded border-input accent-[var(--color-primary)] w-4 h-4"
              />
              <span>Below 80% conformal band only</span>
            </label>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-negative/30 bg-negative/5 p-4 text-sm text-negative">
          {(error as Error).message}
        </div>
      )}

      {/* Results Table */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        {isLoading ? (
          <div className="p-8 space-y-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-12 bg-muted/60 rounded animate-warm-pulse" />
            ))}
          </div>
        ) : deals.length === 0 ? (
          <div className="p-12 text-center space-y-2">
            <Sparkles className="w-6 h-6 text-muted-foreground mx-auto" />
            <p className="text-sm font-medium text-foreground">No listings match these filters</p>
            <p className="text-xs text-muted-foreground">
              Try relaxing the minimum discount, EPC threshold, or unchecking &ldquo;Below 80% conformal band only&rdquo;.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/40">
                  <TableHead>Property</TableHead>
                  <TableHead className="text-right">
                    <button type="button" onClick={() => handleSort("asking_price")} className="inline-flex items-center gap-1 hover:text-foreground cursor-pointer">
                      Asking <ArrowDownUp className="w-3 h-3" />
                    </button>
                  </TableHead>
                  <TableHead className="text-right">
                    <button type="button" onClick={() => handleSort("predicted_price")} className="inline-flex items-center gap-1 hover:text-foreground cursor-pointer">
                      Phase-7 Fair (80% Band) <ArrowDownUp className="w-3 h-3" />
                    </button>
                  </TableHead>
                  <TableHead className="text-right">
                    <button type="button" onClick={() => handleSort("delta_pct")} className="inline-flex items-center gap-1 hover:text-foreground cursor-pointer">
                      Value Gap <ArrowDownUp className="w-3 h-3" />
                    </button>
                  </TableHead>
                  <TableHead className="text-right">
                    <button type="button" onClick={() => handleSort("livable_area")} className="inline-flex items-center gap-1 hover:text-foreground cursor-pointer">
                      Specs <ArrowDownUp className="w-3 h-3" />
                    </button>
                  </TableHead>
                  <TableHead className="text-right">Link</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {deals.map((d: DealItem) => {
                  const psm =
                    d.livable_area && d.livable_area > 0
                      ? Math.round(d.asking_price / d.livable_area)
                      : null;
                  const isGood = d.delta_pct >= 0;
                  return (
                    <TableRow key={`${d.property_id}-${d.date_obtained}`} className="hover:bg-muted/30">
                      <TableCell>
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="text-sm font-medium text-foreground">
                            {d.neighbourhood || d.address}
                          </span>
                          {d.band_position === "below_band" && (
                            <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-positive/10 text-positive border border-positive/25">
                              Below 80% band
                            </span>
                          )}
                          {d.had_price_cut && (
                            <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-warning/10 text-warning border border-warning/30">
                              Price cut
                            </span>
                          )}
                          {d.jepa_dz_quality != null && d.jepa_dz_quality >= 0.45 && (
                            <span
                              title={`Cross-Modal JEPA visual surprise Δz = +${d.jepa_dz_quality.toFixed(2)}σ: photos look sharper than peers with identical specs`}
                              className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-positive/10 text-positive border border-positive/25"
                            >
                              Photos +Δz
                            </span>
                          )}
                          {d.jepa_dz_quality != null && d.jepa_dz_quality <= -0.55 && (
                            <span
                              title={`Cross-Modal JEPA visual surprise Δz = ${d.jepa_dz_quality.toFixed(2)}σ: photos look rougher than specs predict`}
                              className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-negative/10 text-negative border border-negative/25"
                            >
                              Inspect photos
                            </span>
                          )}
                          {d.property_regime && d.property_regime !== "residential" && (
                            <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-muted text-muted-foreground border border-border">
                              {d.property_regime === "mixed_commercial"
                                ? "Mixed-use / Store"
                                : d.property_regime === "multi_unit_investment"
                                  ? "Investment"
                                  : d.property_regime === "live_work"
                                    ? "Live-work"
                                    : d.property_regime.replace("_", " ")}
                            </span>
                          )}
                        </div>
                        <p className="text-[11px] text-muted-foreground mt-0.5">
                          {d.property_type === "APARTMENT" ? "Apartment" : "House"}
                          {d.condition ? ` · ${formatCondition(d.condition)}` : ""}
                          {d.energy_label ? ` · EPC ${d.energy_label}` : ""}
                          {d.date_obtained ? ` · ${d.date_obtained}` : ""}
                        </p>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        <p className="text-sm font-semibold">{formatEur(d.asking_price)}</p>
                        {psm != null && (
                          <p className="text-[11px] text-muted-foreground">{formatPsm(psm)}</p>
                        )}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        <p className="text-sm font-semibold text-foreground">{formatEur(d.predicted_price)}</p>
                        <p className="text-[11px] text-muted-foreground">
                          {formatEurCompact(d.low_eur)} – {formatEurCompact(d.high_eur)}
                        </p>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        <p className={`text-sm font-bold ${isGood ? "text-positive" : "text-negative"}`}>
                          {formatPctSigned(d.delta_pct)}
                        </p>
                        <p className={`text-[11px] ${isGood ? "text-positive/80" : "text-negative/80"}`}>
                          {formatEurSigned(d.delta_eur)}
                        </p>
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-xs text-muted-foreground">
                        <p className="text-foreground font-medium">
                          {d.livable_area ? `${Math.round(d.livable_area)} m²` : "—"}
                        </p>
                        <p>{d.bedrooms != null ? `${Math.round(d.bedrooms)} bed` : ""}</p>
                      </TableCell>
                      <TableCell className="text-right">
                        <a
                          href={d.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
                        >
                          Immoweb <ExternalLink className="w-3 h-3" />
                        </a>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    </div>
  );
}
