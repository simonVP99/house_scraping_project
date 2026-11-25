import { Fragment, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownUp,
  Calendar,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  ExternalLink,
  Home,
  MapPin,
  Radio,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
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

function DealQuickOverview({
  deal,
  liveCheck,
  checking,
  onCheckLive,
}: {
  deal: DealItem;
  liveCheck?: LiveCheckResult;
  checking: boolean;
  onCheckLive: () => void;
}) {
  const askPsm = deal.livable_area && deal.livable_area > 0 ? Math.round(deal.asking_price / deal.livable_area) : null;
  const fairPsm = deal.livable_area && deal.livable_area > 0 ? Math.round(deal.predicted_price / deal.livable_area) : null;

  const amenities = [
    deal.has_garden
      ? deal.garden_surface && deal.garden_surface > 0
        ? `Garden (${Math.round(deal.garden_surface)} m²)`
        : "Garden"
      : null,
    deal.has_terrace
      ? deal.terrace_surface && deal.terrace_surface > 0
        ? `Terrace (${Math.round(deal.terrace_surface)} m²)`
        : "Terrace"
      : null,
    deal.has_photovoltaic_panels ? "Solar panels" : null,
    deal.has_heat_pump ? "Heat pump" : null,
    deal.has_lift ? "Elevator" : null,
  ].filter(Boolean) as string[];

  const regimeLabel =
    deal.property_regime === "mixed_commercial"
      ? "Mixed-Use / Store (Handelspand)"
      : deal.property_regime === "multi_unit_investment"
        ? "Multi-Unit / Opbrengsteigendom"
        : deal.property_regime === "live_work"
          ? "Live-Work / Practice (Praktijk)"
          : deal.property_regime === "rented_yield"
            ? "Currently Rented / Yield Pitch"
            : deal.property_regime === "student_kot"
              ? "Student Kot"
              : deal.property_regime === "service_flat"
                ? "Service Flat"
                : null;

  // Compact visual gauge coordinates for asking vs 80% conformal band
  const spanMin = Math.min(deal.low_eur, deal.asking_price) * 0.94;
  const spanMax = Math.max(deal.high_eur, deal.asking_price) * 1.06;
  const toPct = (v: number) => Math.max(2, Math.min(98, ((v - spanMin) / Math.max(spanMax - spanMin, 1)) * 100));

  return (
    <div className="bg-muted/25 border-t border-border/70 px-5 py-4 space-y-4">
      {/* Top row: Title, street address, availability & actions */}
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-3">
        <div className="space-y-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1 text-xs font-semibold text-foreground">
              <MapPin className="w-3.5 h-3.5 text-primary shrink-0" />
              {deal.address}
            </span>
            {deal.dist_from_center != null && (
              <span className="text-[11px] text-muted-foreground tabular-nums">
                · {deal.dist_from_center.toFixed(1)} km from Korenmarkt
              </span>
            )}
            {deal.agency && (
              <span className="text-[11px] text-muted-foreground">
                · Agency: <strong className="text-foreground font-medium">{deal.agency}</strong>
              </span>
            )}
          </div>
          {deal.title && (
            <p className="text-sm font-medium text-foreground leading-snug">{deal.title}</p>
          )}
          {deal.description_snippet && (
            <p className="text-xs text-muted-foreground leading-relaxed max-w-4xl">
              {deal.description_snippet}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onCheckLive}
            disabled={checking}
            className="h-8 text-xs"
          >
            <Radio className={`w-3.5 h-3.5 mr-1.5 ${checking ? "animate-pulse text-primary" : ""}`} />
            {checking ? "Checking Immoweb..." : "Check Live Status"}
          </Button>
          <a
            href={deal.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 h-8 px-3 rounded-md bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Open Immoweb #{deal.property_id}
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>

      {/* Live check result banner (if user clicked Check Live Status) */}
      {liveCheck && (
        <div
          className={`rounded-lg px-3.5 py-2 text-xs flex items-center justify-between gap-2 border ${
            liveCheck.live_status === "online"
              ? "bg-positive/10 border-positive/30 text-positive"
              : liveCheck.live_status === "offline" || liveCheck.live_status === "sold_or_rented"
                ? "bg-negative/10 border-negative/30 text-negative"
                : "bg-warning/10 border-warning/30 text-warning"
          }`}
        >
          <span className="inline-flex items-center gap-1.5 font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
            Live Immoweb check: {liveCheck.label}
          </span>
          <a href={liveCheck.url} target="_blank" rel="noopener noreferrer" className="underline font-medium">
            Verify on page
          </a>
        </div>
      )}

      {/* 3-Column Quick Overview Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Column 1: Specs & Amenities */}
        <div className="rounded-lg border border-border bg-card p-3.5 space-y-2 text-xs">
          <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
            <Home className="w-3 h-3 text-primary" />
            Property Specs
          </div>
          <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums">
            <dt className="text-muted-foreground">Type</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.property_subtype ? deal.property_subtype.replace(/_/g, " ") : deal.property_type}
            </dd>
            <dt className="text-muted-foreground">Livable / Plot</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.livable_area ? `${Math.round(deal.livable_area)} m²` : "—"}
              {deal.property_size && deal.property_size > 0 ? ` / ${Math.round(deal.property_size)} m²` : ""}
            </dd>
            <dt className="text-muted-foreground">Beds / Baths</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.bedrooms != null ? Math.round(deal.bedrooms) : "—"} /{" "}
              {deal.bathrooms != null ? Math.round(deal.bathrooms) : "—"}
            </dd>
            <dt className="text-muted-foreground">Condition / Year</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.condition ? formatCondition(deal.condition) : "—"}
              {deal.year_built ? ` · ${Math.round(deal.year_built)}` : ""}
            </dd>
            <dt className="text-muted-foreground">EPC</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.energy_label ?? "—"}
              {deal.primary_energy_consumption != null
                ? ` (${Math.round(deal.primary_energy_consumption)} kWh/m²)`
                : ""}
            </dd>
            {deal.cadastral_income != null && deal.cadastral_income > 0 && (
              <>
                <dt className="text-muted-foreground">Cadastral (KI)</dt>
                <dd className="text-right font-medium text-foreground">€{Math.round(deal.cadastral_income)}</dd>
              </>
            )}
          </dl>
          {amenities.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-1 border-t border-border/60">
              {amenities.map((a) => (
                <span key={a} className="px-1.5 py-0.5 rounded bg-muted text-[10px] text-foreground">
                  {a}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Column 2: Valuation & Photo Honesty */}
        <div className="rounded-lg border border-border bg-card p-3.5 space-y-2.5 text-xs">
          <div className="flex items-center justify-between">
            <span className="font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
              Phase-7 Valuation &amp; JEPA Signal
            </span>
            <span className={`font-bold tabular-nums ${deal.delta_pct >= 0 ? "text-positive" : "text-negative"}`}>
              {formatPctSigned(deal.delta_pct)}
            </span>
          </div>

          {/* Mini 80% conformal band bar */}
          <div className="space-y-1 pt-0.5">
            <div className="relative h-4 rounded-full bg-muted/70 overflow-hidden">
              <div
                className="absolute top-0 bottom-0 bg-primary/20 border-x border-primary/50"
                style={{
                  left: `${toPct(deal.low_eur)}%`,
                  width: `${Math.max(4, toPct(deal.high_eur) - toPct(deal.low_eur))}%`,
                }}
              />
              <div
                title={`Phase-7 fair value: ${formatEur(deal.predicted_price)}`}
                className="absolute top-0 bottom-0 w-0.5 bg-primary"
                style={{ left: `${toPct(deal.predicted_price)}%` }}
              />
              <div
                title={`Asking price: ${formatEur(deal.asking_price)}`}
                className={`absolute top-1 bottom-1 w-2 -ml-1 rounded-full ${
                  deal.delta_pct >= 0 ? "bg-positive" : "bg-negative"
                }`}
                style={{ left: `${toPct(deal.asking_price)}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground tabular-nums">
              <span>80% low: {formatEurCompact(deal.low_eur)}</span>
              <span>Fair: {formatEurCompact(deal.predicted_price)}</span>
              <span>high: {formatEurCompact(deal.high_eur)}</span>
            </div>
          </div>

          <dl className="grid grid-cols-2 gap-y-1 gap-x-3 tabular-nums pt-1 border-t border-border/60">
            <dt className="text-muted-foreground">Ask vs Fair €/m²</dt>
            <dd className="text-right font-medium text-foreground">
              {askPsm ? formatPsm(askPsm) : "—"} → {fairPsm ? formatPsm(fairPsm) : "—"}
            </dd>
            {deal.vision_delta_pct != null && (
              <>
                <dt className="text-muted-foreground">Photo price shift</dt>
                <dd
                  className={`text-right font-medium ${
                    deal.vision_delta_pct >= 2
                      ? "text-positive"
                      : deal.vision_delta_pct <= -2
                        ? "text-negative"
                        : "text-foreground"
                  }`}
                >
                  {formatPctSigned(deal.vision_delta_pct)}
                  {deal.vision_delta_eur != null ? ` (${formatEurSigned(deal.vision_delta_eur)})` : ""}
                </dd>
              </>
            )}
            {deal.jepa_dz_quality != null && (
              <>
                <dt className="text-muted-foreground">JEPA Visual Δz</dt>
                <dd
                  className={`text-right font-semibold ${
                    deal.jepa_dz_quality >= 0.45
                      ? "text-positive"
                      : deal.jepa_dz_quality <= -0.55
                        ? "text-negative"
                        : "text-foreground"
                  }`}
                >
                  {deal.jepa_dz_quality > 0 ? "+" : ""}{deal.jepa_dz_quality.toFixed(2)}σ
                </dd>
              </>
            )}
            {deal.jepa_energy_sym != null && (
              <>
                <dt className="text-muted-foreground">Contradiction Energy</dt>
                <dd className="text-right font-medium text-foreground">{deal.jepa_energy_sym.toFixed(2)}</dd>
              </>
            )}
          </dl>
          {regimeLabel && (
            <p className="text-[11px] rounded bg-warning/10 border border-warning/30 text-warning px-2 py-1 font-medium">
              Regime note: {regimeLabel} (wider valuation spread than pure residential)
            </p>
          )}
        </div>

        {/* Column 3: Market History & Availability */}
        <div className="rounded-lg border border-border bg-card p-3.5 space-y-2 text-xs">
          <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
            <Calendar className="w-3 h-3 text-primary" />
            Availability &amp; Market History
          </div>
          <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums">
            <dt className="text-muted-foreground">Status</dt>
            <dd className="text-right">
              <AvailabilityBadge deal={deal} />
            </dd>
            <dt className="text-muted-foreground">First seen</dt>
            <dd className="text-right font-medium text-foreground">{deal.first_seen || deal.date_obtained || "—"}</dd>
            <dt className="text-muted-foreground">Last seen</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.last_seen || deal.date_obtained || "—"}
              {deal.days_since_last_seen != null && deal.days_since_last_seen <= 3 && (
                <span className="text-positive ml-1">(latest)</span>
              )}
            </dd>
            <dt className="text-muted-foreground">Days tracked</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.listing_duration_days != null ? `${Math.round(deal.listing_duration_days)}d` : "—"}
              {deal.n_snapshots != null ? ` (${Math.round(deal.n_snapshots)} scrapes)` : ""}
            </dd>
            <dt className="text-muted-foreground">Price cut</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.had_price_cut ? (
                <span className="text-warning font-semibold">
                  Yes
                  {deal.price_cut_eur && deal.price_cut_eur > 0
                    ? ` (−${formatEurCompact(deal.price_cut_eur)})`
                    : ""}
                </span>
              ) : (
                "None observed"
              )}
            </dd>
            <dt className="text-muted-foreground">Under option</dt>
            <dd className="text-right font-medium text-foreground">
              {deal.is_under_option ? <span className="text-warning font-semibold">Yes</span> : "No"}
            </dd>
          </dl>
          <p className="text-[11px] text-muted-foreground pt-1 border-t border-border/60 leading-snug">
            {deal.availability_status === "active_recent"
              ? `Confirmed online in our latest scrape (${deal.latest_scrape_date ?? "recent"}).`
              : deal.availability_status === "seen_recently"
                ? `Seen within 14 days of our latest scrape (${deal.latest_scrape_date ?? "recent"}). Use "Check Live Status" to verify.`
                : deal.availability_status === "under_option"
                  ? "Seller marked this listing under option on the last scrape."
                  : `Last observed ${deal.last_seen ?? "earlier"} — not present in the ${deal.latest_scrape_date ?? "latest"} scrape, so it may be sold or relisted.`}
          </p>
        </div>
      </div>
    </div>
  );
}

export default function DealsPage() {
  const [neighbourhood, setNeighbourhood] = useState("ALL");
  const [propertyType, setPropertyType] = useState("ALL");
  const [maxPrice, setMaxPrice] = useState<number>(650_000);
  const [minEpc, setMinEpc] = useState("ANY");
  const [minDelta, setMinDelta] = useState<number>(5);
  const [belowBandOnly, setBelowBandOnly] = useState(false);
  const [availableOnly, setAvailableOnly] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("delta_pct");
  const [sortDesc, setSortDesc] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [liveChecks, setLiveChecks] = useState<Record<string, LiveCheckResult>>({});
  const [checkingId, setCheckingId] = useState<string | null>(null);

  const queryParams: DealsQuery = useMemo(
    () => ({
      neighbourhood,
      property_type: propertyType,
      max_price: maxPrice,
      min_epc: minEpc,
      min_delta_pct: minDelta,
      below_band_only: belowBandOnly,
      available_only: availableOnly,
      limit: 200,
    }),
    [neighbourhood, propertyType, maxPrice, minEpc, minDelta, belowBandOnly, availableOnly],
  );

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
            Click any listing to inspect its specs, photo honesty (JEPA Δz), market stay, and recent availability.
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
                : `Latest scrape · ${data.date_used ?? "latest"}`}
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

          <div className="flex flex-col justify-center gap-1.5">
            <label className="inline-flex items-center gap-2 text-xs font-medium text-foreground cursor-pointer select-none">
              <input
                type="checkbox"
                checked={availableOnly}
                onChange={(e) => setAvailableOnly(e.target.checked)}
                className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
              />
              <span>Recently active only</span>
            </label>
            <label className="inline-flex items-center gap-2 text-xs font-medium text-foreground cursor-pointer select-none">
              <input
                type="checkbox"
                checked={belowBandOnly}
                onChange={(e) => setBelowBandOnly(e.target.checked)}
                className="rounded border-input accent-[var(--color-primary)] w-3.5 h-3.5"
              />
              <span>Below 80% fair band</span>
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
              Try relaxing the minimum discount, EPC threshold, or unchecking &ldquo;Recently active only&rdquo;.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/40">
                  <TableHead className="w-8" />
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
                          isExpanded ? "bg-primary/5 hover:bg-primary/8" : "hover:bg-muted/35"
                        }`}
                      >
                        <TableCell className="pr-0 text-muted-foreground">
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
                            {d.jepa_dz_quality != null && d.jepa_dz_quality >= 0.45 && (
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
                          <TableCell colSpan={8} className="p-0">
                            <DealQuickOverview
                              deal={d}
                              liveCheck={liveChecks[d.property_id]}
                              checking={checkingId === d.property_id}
                              onCheckLive={() => handleCheckLive(d.property_id)}
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
    </div>
  );
}
