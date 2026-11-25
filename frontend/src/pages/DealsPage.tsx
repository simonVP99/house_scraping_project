import { Fragment, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownUp,
  ChevronDown,
  ChevronRight,
  Clock,
  ExternalLink,
  RefreshCw,
  RotateCcw,
  Scale,
  Search,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
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
import { api } from "../services/api";
import DealDossierPanel from "../components/DealDossierPanel";
import CompareModal from "../components/CompareModal";
import type { DealItem, DealsQuery, LiveCheckResult } from "../types/insights";
import {
  formatCondition,
  formatEur,
  formatEurCompact,
  formatEurSigned,
  formatPctSigned,
  formatPsm,
} from "../lib/format";

type SortKey = "delta_pct" | "delta_eur" | "asking_price" | "predicted_price" | "livable_area" | "days_since_last_seen";

const MAX_COMPARE = 4;

function AvailabilityBadge({ deal }: { deal: DealItem }) {
  const status = deal.availability_status;
  const lastSeen = deal.last_seen || deal.date_obtained;
  const days = deal.days_since_last_seen;

  if (status === "active_recent") {
    return (
      <span
        title={`Confirmed active in the latest scrape (${lastSeen ?? "recent"})`}
        className="inline-flex items-center gap-1 whitespace-nowrap text-[10px] font-semibold px-2 py-0.5 rounded-full bg-positive/12 text-positive border border-positive/30"
      >
        <span className="w-1.5 h-1.5 rounded-full bg-positive shrink-0" />
        Active · {lastSeen ?? "latest"}
      </span>
    );
  }
  if (status === "seen_recently") {
    return (
      <span
        title={`Last seen ${days}d before latest scrape (${lastSeen ?? "recent"})`}
        className="inline-flex items-center gap-1 whitespace-nowrap text-[10px] font-semibold px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/25"
      >
        <Clock className="w-2.5 h-2.5 shrink-0" />
        Seen {lastSeen ?? `${days}d ago`}
      </span>
    );
  }
  if (status === "under_option") {
    return (
      <span
        title={`Marked under option as of ${lastSeen ?? "latest snapshot"}`}
        className="inline-flex items-center gap-1 whitespace-nowrap text-[10px] font-semibold px-2 py-0.5 rounded-full bg-warning/15 text-warning border border-warning/35"
      >
        Under option
      </span>
    );
  }
  if (status === "sold_or_rented") {
    return (
      <span className="inline-flex items-center gap-1 whitespace-nowrap text-[10px] font-semibold px-2 py-0.5 rounded-full bg-negative/10 text-negative border border-negative/25">
        Sold / Removed
      </span>
    );
  }
  return (
    <span
      title={`Not seen in the most recent scrape (${deal.latest_scrape_date ?? "latest"}). Last observed ${lastSeen ?? "earlier"}.`}
      className="inline-flex items-center gap-1 whitespace-nowrap text-[10px] font-medium px-2 py-0.5 rounded-full bg-muted text-muted-foreground border border-border"
    >
      Last seen {lastSeen ?? "earlier"}
    </span>
  );
}

export default function DealsPage() {
  const [neighbourhood, setNeighbourhood] = useState("ALL");
  const [propertyType, setPropertyType] = useState("ALL");
  const [maxPredicted, setMaxPredicted] = useState<number>(650_000);
  const [minEpc, setMinEpc] = useState("ANY");
  const [minCondition, setMinCondition] = useState("ANY");
  const [minArea, setMinArea] = useState<number>(0);
  const [minBedrooms, setMinBedrooms] = useState<number>(0);
  const [regime, setRegime] = useState("ALL");
  const [hasGarden, setHasGarden] = useState(false);
  const [hasTerrace, setHasTerrace] = useState(false);
  const [priceCutOnly, setPriceCutOnly] = useState(false);
  const [goodPhotosOnly, setGoodPhotosOnly] = useState(false);
  const [includeArchive, setIncludeArchive] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [minDelta, setMinDelta] = useState<number>(5);
  const [belowBandOnly, setBelowBandOnly] = useState(false);
  const [availableOnly, setAvailableOnly] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("delta_pct");
  const [sortDesc, setSortDesc] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [liveChecks, setLiveChecks] = useState<Record<string, LiveCheckResult>>({});
  const [checkingId, setCheckingId] = useState<string | null>(null);

  // Side-by-side comparison tray (max 4 properties)
  const [compareList, setCompareList] = useState<
    { property_id: string; label: string; asking_price: number }[]
  >([]);
  const [compareOpen, setCompareOpen] = useState(false);

  const queryParams: DealsQuery = useMemo(
    () => ({
      neighbourhood,
      property_type: propertyType,
      max_predicted: maxPredicted,
      min_epc: minEpc,
      min_condition: minCondition,
      min_area: minArea > 0 ? minArea : undefined,
      min_bedrooms: minBedrooms > 0 ? minBedrooms : undefined,
      regime,
      has_garden: hasGarden,
      has_terrace: hasTerrace,
      price_cut_only: priceCutOnly,
      min_jepa_dz: goodPhotosOnly ? 0.25 : undefined,
      search: searchText.trim() || undefined,
      include_archive: includeArchive,
      min_delta_pct: minDelta,
      below_band_only: belowBandOnly,
      available_only: availableOnly,
      limit: 200,
    }),
    [
      neighbourhood,
      propertyType,
      maxPredicted,
      minEpc,
      minCondition,
      minArea,
      minBedrooms,
      regime,
      hasGarden,
      hasTerrace,
      priceCutOnly,
      goodPhotosOnly,
      searchText,
      includeArchive,
      minDelta,
      belowBandOnly,
      availableOnly,
    ],
  );

  const resetFilters = () => {
    setNeighbourhood("ALL");
    setPropertyType("ALL");
    setMaxPredicted(650_000);
    setMinEpc("ANY");
    setMinCondition("ANY");
    setMinArea(0);
    setMinBedrooms(0);
    setRegime("ALL");
    setHasGarden(false);
    setHasTerrace(false);
    setPriceCutOnly(false);
    setGoodPhotosOnly(false);
    setIncludeArchive(false);
    setSearchText("");
    setMinDelta(5);
    setBelowBandOnly(false);
    setAvailableOnly(false);
  };

  const { data, isLoading, error } = useDeals(queryParams);
  const { data: neighbourhoods } = useNeighbourhoods();
  const refresh = useRefreshListings();
  const qc = useQueryClient();

  const deals = useMemo(() => {
    const items = [...(data?.deals ?? [])];
    items.sort((a, b) => {
      const av = Number(a[sortKey] ?? (sortKey === "days_since_last_seen" ? 9999 : -Infinity));
      const bv = Number(b[sortKey] ?? (sortKey === "days_since_last_seen" ? 9999 : -Infinity));
      return sortDesc ? bv - av : av - bv;
    });
    return items;
  }, [data, sortKey, sortDesc]);

  const belowBandCount = useMemo(
    () => deals.filter((d) => d.band_position === "below_band").length,
    [deals],
  );
  const activeRecentCount = useMemo(
    () => deals.filter((d) => d.availability_status === "active_recent" || d.availability_status === "seen_recently").length,
    [deals],
  );

  const comparedIds = useMemo(() => new Set(compareList.map((c) => c.property_id)), [compareList]);

  const toggleCompare = (deal: DealItem) => {
    setCompareList((prev) => {
      if (prev.some((c) => c.property_id === deal.property_id)) {
        return prev.filter((c) => c.property_id !== deal.property_id);
      }
      if (prev.length >= MAX_COMPARE) return prev;
      return [
        ...prev,
        {
          property_id: deal.property_id,
          label: deal.neighbourhood || deal.address || `#${deal.property_id}`,
          asking_price: deal.asking_price,
        },
      ];
    });
  };

  const removeCompare = (pid: string) => {
    setCompareList((prev) => {
      const next = prev.filter((c) => c.property_id !== pid);
      if (next.length === 0) setCompareOpen(false);
      return next;
    });
  };

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDesc((d) => !d);
    else {
      setSortKey(key);
      setSortDesc(key !== "asking_price" && key !== "days_since_last_seen");
    }
  };

  const handleRefresh = () => {
    refresh.mutate(undefined, {
      onSuccess: () => {
        qc.invalidateQueries({ queryKey: ["deals"] });
      },
    });
  };

  const handleCheckLive = async (pid: string) => {
    setCheckingId(pid);
    try {
      const res = await api.checkDealLive(pid);
      setLiveChecks((prev) => ({ ...prev, [pid]: res }));
    } catch {
      setLiveChecks((prev) => ({
        ...prev,
        [pid]: {
          property_id: pid,
          live_status: "error",
          label: "Could not reach Immoweb — Open Link",
          url: `https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/${pid}`,
        },
      }));
    } finally {
      setCheckingId((cur) => (cur === pid ? null : cur));
    }
  };

  const selectClass =
    "flex h-9 w-full rounded-md border border-input bg-background px-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring";

  const isLive = data?.source === "live_scrape";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-24 space-y-6">
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
            Click any house for its complete dossier (photos, specs, JEPA audit, Flanders environment &amp; history), or tick up to 4 to compare side by side.
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
                : data.source === "corpus_full"
                  ? `Full local corpus · through ${data.date_used ?? "latest"}`
                  : `Local corpus (90d) · through ${data.date_used ?? "latest"}`}
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
          <div className="flex flex-wrap items-center gap-2.5 text-xs tabular-nums">
            <span className="text-muted-foreground">
              Showing <strong className="text-foreground">{deals.length}</strong>
              {data && data.total > deals.length ? ` of ${data.total}` : ""} listings
            </span>
            <span className="px-2.5 py-0.5 rounded-full bg-positive/10 text-positive border border-positive/25 font-semibold">
              {activeRecentCount} recently active
            </span>
            <span className="px-2.5 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/25 font-semibold">
              {belowBandCount} below 80% fair band
            </span>
            <button
              type="button"
              onClick={resetFilters}
              className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" /> Reset
            </button>
          </div>
        </div>

        {/* Row 1: Core valuation & location filters */}
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
              Max Predicted ({formatEurCompact(maxPredicted)})
            </label>
            <input
              id="deal-budget"
              type="range"
              min={200_000}
              max={1_000_000}
              step={25_000}
              value={maxPredicted}
              onChange={(e) => setMaxPredicted(Number(e.target.value))}
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
            <label htmlFor="deal-cond" className="block text-xs font-medium text-muted-foreground mb-1">
              Min Condition
            </label>
            <select
              id="deal-cond"
              value={minCondition}
              onChange={(e) => setMinCondition(e.target.value)}
              className={selectClass}
            >
              <option value="ANY">Any condition</option>
              <option value="AS_NEW">As new only</option>
              <option value="JUST_RENOVATED">Just renovated +</option>
              <option value="GOOD">Good +</option>
              <option value="TO_BE_DONE_UP">To refresh +</option>
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
        </div>

        {/* Row 2: Size, Bedrooms, Regime & Local Search */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 items-end pt-3 border-t border-border/60">
          <div>
            <label htmlFor="deal-area" className="block text-xs font-medium text-muted-foreground mb-1">
              Min Area
            </label>
            <select
              id="deal-area"
              value={minArea}
              onChange={(e) => setMinArea(Number(e.target.value))}
              className={selectClass}
            >
              <option value={0}>Any size</option>
              <option value={80}>≥ 80 m²</option>
              <option value={110}>≥ 110 m²</option>
              <option value={140}>≥ 140 m²</option>
              <option value={180}>≥ 180 m²</option>
              <option value={220}>≥ 220 m²</option>
            </select>
          </div>

          <div>
            <label htmlFor="deal-beds" className="block text-xs font-medium text-muted-foreground mb-1">
              Min Bedrooms
            </label>
            <select
              id="deal-beds"
              value={minBedrooms}
              onChange={(e) => setMinBedrooms(Number(e.target.value))}
              className={selectClass}
            >
              <option value={0}>Any</option>
              <option value={1}>1+ bed</option>
              <option value={2}>2+ beds</option>
              <option value={3}>3+ beds</option>
              <option value={4}>4+ beds</option>
            </select>
          </div>

          <div>
            <label htmlFor="deal-regime" className="block text-xs font-medium text-muted-foreground mb-1">
              Property Regime
            </label>
            <select
              id="deal-regime"
              value={regime}
              onChange={(e) => setRegime(e.target.value)}
              className={selectClass}
            >
              <option value="ALL">All regimes</option>
              <option value="residential">Pure residential only</option>
              <option value="mixed_commercial">Mixed-use / Store</option>
              <option value="live_work">Live-work / Practice</option>
              <option value="multi_unit_investment">Multi-unit / Investment</option>
              <option value="rented_yield">Rented / Yield pitch</option>
              <option value="non_standard">All non-standard</option>
            </select>
          </div>

          <div className="sm:col-span-2 lg:col-span-3">
            <label htmlFor="deal-search" className="block text-xs font-medium text-muted-foreground mb-1">
              Search Local Corpus (street, agency, title, or Immoweb #ID)
            </label>
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-muted-foreground absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                id="deal-search"
                type="text"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="e.g. Kortrijksesteenweg, Trevi, tuin, 21868762..."
                className={`${selectClass} pl-8`}
              />
            </div>
          </div>
        </div>

        {/* Row 3: Quick Toggle Chips (all evaluated locally) */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pt-2 border-t border-border/60 text-xs">
          <label className="inline-flex items-center gap-1.5 font-medium text-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={availableOnly}
              onChange={(e) => setAvailableOnly(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
            />
            <span>Recently active only</span>
          </label>
          <label className="inline-flex items-center gap-1.5 font-medium text-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={belowBandOnly}
              onChange={(e) => setBelowBandOnly(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
            />
            <span>Below 80% fair band</span>
          </label>
          <label className="inline-flex items-center gap-1.5 font-medium text-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={hasGarden}
              onChange={(e) => setHasGarden(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
            />
            <span>Garden</span>
          </label>
          <label className="inline-flex items-center gap-1.5 font-medium text-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={hasTerrace}
              onChange={(e) => setHasTerrace(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
            />
            <span>Terrace</span>
          </label>
          <label className="inline-flex items-center gap-1.5 font-medium text-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={priceCutOnly}
              onChange={(e) => setPriceCutOnly(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
            />
            <span>Price cut observed</span>
          </label>
          <label className="inline-flex items-center gap-1.5 font-medium text-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={goodPhotosOnly}
              onChange={(e) => setGoodPhotosOnly(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
            />
            <span>JEPA Photos +Δz (≥+0.25σ)</span>
          </label>
          <label
            title="Search across all 13,439 normal Ghent homes in our local corpus (2024–2026) — even older/offline listings — for benchmarking & comparison"
            className="inline-flex items-center gap-1.5 font-medium text-primary cursor-pointer select-none ml-auto"
          >
            <input
              type="checkbox"
              checked={includeArchive}
              onChange={(e) => setIncludeArchive(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
            />
            <span>Include full local archive (13.4k homes)</span>
          </label>
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
              Try raising Max Predicted, relaxing the minimum discount / EPC threshold, or unchecking &ldquo;Recently active only&rdquo;.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/40">
                  <TableHead className="w-10 text-center" title="Select up to 4 homes to compare side by side">
                    <Scale className="w-3.5 h-3.5 mx-auto text-muted-foreground" />
                  </TableHead>
                  <TableHead className="w-7" />
                  <TableHead>Property</TableHead>
                  <TableHead>
                    <button type="button" onClick={() => handleSort("days_since_last_seen")} className="inline-flex items-center gap-1 hover:text-foreground cursor-pointer">
                      Availability <ArrowDownUp className="w-3 h-3" />
                    </button>
                  </TableHead>
                  <TableHead className="text-right">
                    <button type="button" onClick={() => handleSort("asking_price")} className="inline-flex items-center gap-1 hover:text-foreground cursor-pointer">
                      Asking <ArrowDownUp className="w-3 h-3" />
                    </button>
                  </TableHead>
                  <TableHead className="text-right">
                    <button type="button" onClick={() => handleSort("predicted_price")} className="inline-flex items-center gap-1 hover:text-foreground cursor-pointer">
                      Phase-7 Fair <ArrowDownUp className="w-3 h-3" />
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
                  const rowKey = `${d.property_id}-${d.date_obtained ?? ""}`;
                  const isExpanded = expandedId === rowKey;
                  const isCompared = comparedIds.has(d.property_id);
                  const canAddCompare = compareList.length < MAX_COMPARE;
                  const psm =
                    d.livable_area && d.livable_area > 0
                      ? Math.round(d.asking_price / d.livable_area)
                      : null;
                  const isGood = d.delta_pct >= 0;
                  return (
                    <Fragment key={rowKey}>
                      <TableRow
                        onClick={() => setExpandedId(isExpanded ? null : rowKey)}
                        aria-expanded={isExpanded}
                        className={`cursor-pointer transition-colors ${
                          isExpanded
                            ? "bg-primary/6 hover:bg-primary/8"
                            : isCompared
                              ? "bg-positive/5 hover:bg-positive/8"
                              : "hover:bg-muted/35"
                        }`}
                      >
                        <TableCell
                          className="text-center pr-0"
                          onClick={(e) => {
                            e.stopPropagation();
                          }}
                        >
                          <input
                            type="checkbox"
                            checked={isCompared}
                            disabled={!isCompared && !canAddCompare}
                            onChange={() => toggleCompare(d)}
                            aria-label={`Select ${d.address} for comparison`}
                            title={
                              isCompared
                                ? "Remove from comparison"
                                : canAddCompare
                                  ? "Add to side-by-side comparison"
                                  : "Maximum 4 properties selected"
                            }
                            className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5 cursor-pointer disabled:opacity-40"
                          />
                        </TableCell>
                        <TableCell className="px-1 text-muted-foreground">
                          {isExpanded ? (
                            <ChevronDown className="w-4 h-4 text-primary" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                        </TableCell>
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
                            {d.has_renders && (
                              <span
                                title={`CGI architectural renders / virtual staging detected (${d.vis_pct_any_render ?? "?"}% of photos) — visual fair value may reflect post-renovation renders`}
                                className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-warning/15 text-warning border border-warning/35"
                              >
                                Renders / Staged
                              </span>
                            )}
                            {d.has_condition_mislabel && (
                              <span
                                title="Immoweb condition tag says As New / Renovated / First-time use, but the description says 'te renoveren'"
                                className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-negative/10 text-negative border border-negative/25"
                              >
                                Condition mismatch
                              </span>
                            )}
                            {d.has_text_under_option && (
                              <span
                                title="Description text mentions 'optie' / 'bod aanvaard' / 'compromis'"
                                className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-warning/10 text-warning border border-warning/30"
                              >
                                Text: Option
                              </span>
                            )}
                            {!d.has_renders && d.jepa_dz_quality != null && d.jepa_dz_quality >= 0.45 && (
                              <span
                                title={`Cross-Modal JEPA visual surprise Δz = +${d.jepa_dz_quality.toFixed(2)}σ`}
                                className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-positive/10 text-positive border border-positive/25"
                              >
                                Photos +Δz
                              </span>
                            )}
                            {d.jepa_dz_quality != null && d.jepa_dz_quality <= -0.55 && (
                              <span
                                title={`Cross-Modal JEPA visual surprise Δz = ${d.jepa_dz_quality.toFixed(2)}σ`}
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
                          <p className="text-[11px] text-muted-foreground mt-0.5 truncate max-w-md">
                            {d.property_type === "APARTMENT" ? "Apartment" : "House"}
                            {d.condition ? ` · ${formatCondition(d.condition)}` : ""}
                            {d.energy_label ? ` · EPC ${d.energy_label}` : ""}
                            {d.address && d.neighbourhood && !d.address.startsWith(d.neighbourhood)
                              ? ` · ${d.address}`
                              : ""}
                          </p>
                        </TableCell>
                        <TableCell>
                          <AvailabilityBadge deal={d} />
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
                            onClick={(e) => e.stopPropagation()}
                            className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
                          >
                            Immoweb <ExternalLink className="w-3 h-3" />
                          </a>
                        </TableCell>
                      </TableRow>
                      {isExpanded && (
                        <TableRow className="hover:bg-transparent">
                          <TableCell colSpan={9} className="p-0">
                            <DealDossierPanel
                              deal={d}
                              liveCheck={liveChecks[d.property_id]}
                              checking={checkingId === d.property_id}
                              isCompared={isCompared}
                              canAddCompare={canAddCompare}
                              onCheckLive={() => handleCheckLive(d.property_id)}
                              onToggleCompare={() => toggleCompare(d)}
                            />
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      {/* Floating Side-by-Side Comparison Dock */}
      {compareList.length > 0 && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-40 w-[min(94vw,56rem)] rounded-2xl border border-primary/30 bg-card/95 backdrop-blur shadow-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2 min-w-0">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-primary">
              <Scale className="w-4 h-4" />
              Compare ({compareList.length}/{MAX_COMPARE})
            </span>
            {compareList.map((c) => (
              <span
                key={c.property_id}
                className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-muted border border-border text-foreground max-w-[190px]"
              >
                <span className="truncate">{c.label}</span>
                <span className="text-[10px] text-muted-foreground tabular-nums">
                  {formatEurCompact(c.asking_price)}
                </span>
                <button
                  type="button"
                  onClick={() => removeCompare(c.property_id)}
                  className="text-muted-foreground hover:text-foreground cursor-pointer"
                  aria-label={`Remove ${c.label}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                setCompareList([]);
                setCompareOpen(false);
              }}
            >
              Clear
            </Button>
            <Button
              type="button"
              size="sm"
              disabled={compareList.length < 2}
              onClick={() => setCompareOpen(true)}
            >
              <Scale className="w-3.5 h-3.5 mr-1.5" />
              {compareList.length < 2 ? "Pick 1 more to compare" : `Compare ${compareList.length} Homes`}
            </Button>
          </div>
        </div>
      )}

      {/* Side-by-Side Comparison Modal */}
      {compareOpen && compareList.length > 0 && (
        <CompareModal
          propertyIds={compareList.map((c) => c.property_id)}
          onRemove={removeCompare}
          onClear={() => {
            setCompareList([]);
            setCompareOpen(false);
          }}
          onClose={() => setCompareOpen(false)}
        />
      )}
    </div>
  );
}
