import { useState } from "react";
import {
  Building2,
  Calendar,
  CheckCircle2,
  ExternalLink,
  FileText,
  Leaf,
  MapPin,
  Radio,
  Scale,
  ShieldCheck,
  Sparkles,
  Trees,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePropertyDossier } from "../hooks/useInsights";
import type { DealItem, LiveCheckResult, PropertyDossier } from "../types/insights";
import {
  formatCondition,
  formatEur,
  formatEurCompact,
  formatEurSigned,
  formatPctSigned,
  formatPsm,
} from "../lib/format";

function yesNo(v: boolean | null | undefined, yes = "Yes", no = "No") {
  if (v === true) return yes;
  if (v === false) return no;
  return "—";
}

function fmtMeters(m: number | null | undefined) {
  if (m == null) return "—";
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`;
}

function fmtVal(v: string | number | null | undefined, suffix = "") {
  if (v == null || v === "") return "—";
  return `${v}${suffix}`;
}

function DossierBody({
  dossier,
  liveCheck,
  checking,
  isCompared,
  canAddCompare,
  onCheckLive,
  onToggleCompare,
}: {
  dossier: PropertyDossier;
  liveCheck?: LiveCheckResult;
  checking: boolean;
  isCompared: boolean;
  canAddCompare: boolean;
  onCheckLive: () => void;
  onToggleCompare: () => void;
}) {
  const [showFullDesc, setShowFullDesc] = useState(false);
  const v = dossier.valuation;
  const s = dossier.structure;
  const e = dossier.energy_compliance;
  const f = dossier.financial;
  const env = dossier.environment;
  const av = dossier.availability;

  const spanMin = Math.min(v.low_80_eur ?? 0, v.asking_eur ?? 0) * 0.94;
  const spanMax = Math.max(v.high_80_eur ?? 1, v.asking_eur ?? 1) * 1.06;
  const toPct = (x: number) => Math.max(2, Math.min(98, ((x - spanMin) / Math.max(spanMax - spanMin, 1)) * 100));

  const desc = dossier.description || "";
  const shortDesc = desc.length > 420 && !showFullDesc ? `${desc.slice(0, 420)}…` : desc;

  return (
    <div className="bg-muted/25 border-t border-border/70 px-5 py-5 space-y-5">
      {/* Header + Actions */}
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div className="space-y-1.5 min-w-0">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="inline-flex items-center gap-1 font-semibold text-foreground">
              <MapPin className="w-3.5 h-3.5 text-primary shrink-0" />
              {dossier.address}
            </span>
            {dossier.neighbourhood && (
              <span className="px-2 py-0.5 rounded bg-primary/10 text-primary font-medium">
                {dossier.neighbourhood}
              </span>
            )}
            {dossier.dist_from_center_km != null && (
              <span className="text-muted-foreground tabular-nums">
                · {dossier.dist_from_center_km.toFixed(1)} km from Korenmarkt
              </span>
            )}
            {dossier.street_type && (
              <span className="text-muted-foreground">· {dossier.street_type.replace(/_/g, " ")}</span>
            )}
            {av.agency && (
              <span className="text-muted-foreground">
                · Agency: <strong className="text-foreground font-medium">{av.agency}</strong>
              </span>
            )}
          </div>
          {dossier.title && (
            <h3 className="font-display text-lg font-semibold text-foreground leading-snug">
              {dossier.title}
            </h3>
          )}
          {dossier.regime?.advisory && (
            <p className="text-xs rounded-md bg-warning/10 border border-warning/30 text-warning px-3 py-1.5 inline-block">
              <strong>{dossier.regime.label}:</strong> {dossier.regime.advisory}
            </p>
          )}
          {dossier.caveats && dossier.caveats.length > 0 && (
            <div className="space-y-1.5 pt-0.5">
              {dossier.caveats.map((cav) => (
                <div
                  key={cav.code}
                  className={`text-xs rounded-md px-3 py-1.5 border ${
                    cav.severity === "high"
                      ? "bg-negative/8 border-negative/30 text-negative"
                      : "bg-warning/10 border-warning/30 text-warning"
                  }`}
                >
                  <strong>⚠ {cav.label}:</strong> {cav.message}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <Button
            type="button"
            variant={isCompared ? "default" : "outline"}
            size="sm"
            onClick={onToggleCompare}
            disabled={!isCompared && !canAddCompare}
            className="h-8 text-xs"
          >
            <Scale className="w-3.5 h-3.5 mr-1.5" />
            {isCompared ? "In Compare" : canAddCompare ? "Add to Compare" : "Compare Full (4)"}
          </Button>
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
            href={dossier.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 h-8 px-3 rounded-md bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
          >
            Immoweb #{dossier.property_id}
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>

      {/* Live Check Banner */}
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
            Open listing
          </a>
        </div>
      )}

      {/* Photo Strip (if local cached photos exist) */}
      {dossier.photos.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
            <span className="font-semibold uppercase tracking-wider">
              Listing Photos ({dossier.photos.length} cached
              {dossier.picture_count ? ` of ${dossier.picture_count} total` : ""})
              {dossier.jepa.pct_any_render != null && dossier.jepa.pct_any_render >= 30 && (
                <span className="ml-2 text-warning font-semibold normal-case">
                  · {Math.round(dossier.jepa.pct_any_render)}% flagged CGI/staged
                  {dossier.jepa.pct_cgi != null && dossier.jepa.pct_cgi > 0
                    ? ` (${Math.round(dossier.jepa.pct_cgi)}% hard CGI)`
                    : ""}
                </span>
              )}
            </span>
            {dossier.has_virtual_tour && <span className="text-primary font-medium">3D Virtual Tour listed</span>}
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
            {dossier.photos.slice(0, 6).map((src, idx) => (
              <a
                key={src}
                href={src}
                target="_blank"
                rel="noopener noreferrer"
                className="group relative aspect-[4/3] rounded-lg overflow-hidden border border-border bg-muted"
              >
                <img
                  src={src}
                  alt={`Listing photo ${idx + 1}`}
                  loading="lazy"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                />
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Row 1: Valuation + JEPA Photo Honesty + Availability & Price History */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Card 1: Phase-7 Valuation */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2.5 text-xs">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
              <Sparkles className="w-3 h-3 text-primary" />
              Phase-7 Valuation Breakdown
            </span>
            {v.delta_pct != null && (
              <span className={`font-bold tabular-nums ${v.delta_pct >= 0 ? "text-positive" : "text-negative"}`}>
                {formatPctSigned(v.delta_pct)}
              </span>
            )}
          </div>

          {v.low_80_eur != null && v.high_80_eur != null && v.predicted_mm_eur != null && v.asking_eur != null && (
            <div className="space-y-1">
              <div className="relative h-4 rounded-full bg-muted/70 overflow-hidden">
                <div
                  className="absolute top-0 bottom-0 bg-primary/20 border-x border-primary/50"
                  style={{
                    left: `${toPct(v.low_80_eur)}%`,
                    width: `${Math.max(4, toPct(v.high_80_eur) - toPct(v.low_80_eur))}%`,
                  }}
                />
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-primary"
                  style={{ left: `${toPct(v.predicted_mm_eur)}%` }}
                />
                <div
                  className={`absolute top-1 bottom-1 w-2 -ml-1 rounded-full ${
                    (v.delta_pct ?? 0) >= 0 ? "bg-positive" : "bg-negative"
                  }`}
                  style={{ left: `${toPct(v.asking_eur)}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground tabular-nums">
                <span>80% low: {formatEurCompact(v.low_80_eur)}</span>
                <span>Fair: {formatEurCompact(v.predicted_mm_eur)}</span>
                <span>high: {formatEurCompact(v.high_80_eur)}</span>
              </div>
            </div>
          )}

          <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums pt-1 border-t border-border/60">
            <dt className="text-muted-foreground">Asking price</dt>
            <dd className="text-right font-semibold">{v.asking_eur ? formatEur(v.asking_eur) : "—"}</dd>
            <dt className="text-muted-foreground">Fair (Multimodal)</dt>
            <dd className="text-right font-semibold text-primary">
              {v.predicted_mm_eur ? formatEur(v.predicted_mm_eur) : "—"}
            </dd>
            <dt className="text-muted-foreground">Fair (Specs+Text only)</dt>
            <dd className="text-right font-medium text-muted-foreground">
              {v.predicted_tt_eur ? formatEur(v.predicted_tt_eur) : "—"}
            </dd>
            <dt className="text-muted-foreground">Photo impact</dt>
            <dd
              className={`text-right font-semibold ${
                (v.vision_delta_pct ?? 0) >= 2
                  ? "text-positive"
                  : (v.vision_delta_pct ?? 0) <= -2
                    ? "text-negative"
                    : "text-foreground"
              }`}
            >
              {v.vision_delta_pct != null ? formatPctSigned(v.vision_delta_pct) : "—"}
              {v.vision_delta_eur != null ? ` (${formatEurSigned(v.vision_delta_eur)})` : ""}
            </dd>
            <dt className="text-muted-foreground">Ask → Fair €/m²</dt>
            <dd className="text-right font-medium">
              {v.asking_psm ? formatPsm(v.asking_psm) : "—"} → {v.fair_psm ? formatPsm(v.fair_psm) : "—"}
            </dd>
            {dossier.comparables && (
              <>
                <dt className="text-muted-foreground">Local comps median</dt>
                <dd className="text-right font-medium">
                  {formatEurCompact(dossier.comparables.median_price_k * 1000)} (n={dossier.comparables.count})
                </dd>
              </>
            )}
          </dl>
        </div>

        {/* Card 2: JEPA Photo Honesty */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2.5 text-xs">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
              <ShieldCheck className="w-3 h-3 text-primary" />
              Cross-Modal JEPA Photo Audit
            </span>
            {dossier.jepa.detail && (
              <span
                className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                  dossier.jepa.detail.tone === "positive"
                    ? "bg-positive/10 text-positive border-positive/30"
                    : dossier.jepa.detail.tone === "negative"
                      ? "bg-negative/10 text-negative border-negative/30"
                      : dossier.jepa.detail.tone === "warning"
                        ? "bg-warning/10 text-warning border-warning/30"
                        : "bg-muted text-foreground border-border"
                }`}
              >
                {dossier.jepa.detail.verdict_label}
              </span>
            )}
          </div>

          {dossier.jepa.detail ? (
            <>
              <p className="text-[11px] text-muted-foreground leading-relaxed">{dossier.jepa.detail.summary}</p>
              <div className="grid grid-cols-3 gap-1.5 text-center tabular-nums">
                <div className="rounded bg-muted/50 px-2 py-1">
                  <p className="text-[10px] text-muted-foreground">Δz Quality</p>
                  <p className="font-semibold">
                    {dossier.jepa.detail.dz_quality > 0 ? "+" : ""}{dossier.jepa.detail.dz_quality.toFixed(2)}σ
                  </p>
                </div>
                <div className="rounded bg-muted/50 px-2 py-1">
                  <p className="text-[10px] text-muted-foreground">Energy</p>
                  <p className="font-semibold">{Math.round(dossier.jepa.detail.energy_percentile)}th pct</p>
                </div>
                <div className="rounded bg-muted/50 px-2 py-1">
                  <p className="text-[10px] text-muted-foreground">Room Spread</p>
                  <p className="font-semibold">{Math.round(dossier.jepa.detail.cherry_percentile)}th pct</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 pt-1 border-t border-border/60 tabular-nums">
                {dossier.jepa.detail.sub_scores.map((ax) => (
                  <div key={ax.axis} className="flex justify-between">
                    <span className="text-muted-foreground truncate">{ax.axis}</span>
                    <span
                      className={`font-semibold ${
                        ax.z >= 0.45 ? "text-positive" : ax.z <= -0.45 ? "text-negative" : "text-foreground"
                      }`}
                    >
                      {ax.z > 0 ? "+" : ""}{ax.z.toFixed(2)}σ
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums">
              <dt className="text-muted-foreground">Visual Surprise Δz</dt>
              <dd className="text-right font-semibold">
                {dossier.jepa.dz_quality != null
                  ? `${dossier.jepa.dz_quality > 0 ? "+" : ""}${dossier.jepa.dz_quality.toFixed(2)}σ`
                  : "—"}
              </dd>
              <dt className="text-muted-foreground">Contradiction Energy</dt>
              <dd className="text-right font-medium">{fmtVal(dossier.jepa.energy_sym)}</dd>
              <dt className="text-muted-foreground">Cherry-Pick Spread</dt>
              <dd className="text-right font-medium">{fmtVal(dossier.jepa.cherry_spread)}</dd>
            </dl>
          )}
        </div>

        {/* Card 3: Availability, Market History & Price Timeline */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2.5 text-xs">
          <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
            <Calendar className="w-3 h-3 text-primary" />
            Availability &amp; Scrape History
          </div>
          <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums">
            <dt className="text-muted-foreground">First → Last seen</dt>
            <dd className="text-right font-medium">
              {av.first_seen || "—"} → {av.last_seen || "—"}
            </dd>
            <dt className="text-muted-foreground">Tracked stay</dt>
            <dd className="text-right font-medium">
              {av.listing_duration_days != null ? `${av.listing_duration_days}d` : "—"}
              {av.n_snapshots ? ` (${av.n_snapshots} scrapes)` : ""}
            </dd>
            <dt className="text-muted-foreground">Price cut</dt>
            <dd className="text-right font-medium">
              {av.had_price_cut ? (
                <span className="text-warning font-semibold">
                  Yes{av.price_cut_eur ? ` (−${formatEurCompact(av.price_cut_eur)})` : ""}
                </span>
              ) : (
                "None"
              )}
            </dd>
            <dt className="text-muted-foreground">Under option</dt>
            <dd className="text-right font-medium">{yesNo(av.is_under_option)}</dd>
            {(av.view_count != null || av.bookmark_count != null) && (
              <>
                <dt className="text-muted-foreground">Views / Saves</dt>
                <dd className="text-right font-medium">
                  {av.view_count ?? "—"} / {av.bookmark_count ?? "—"}
                </dd>
              </>
            )}
            {av.seller_type && (
              <>
                <dt className="text-muted-foreground">Seller type</dt>
                <dd className="text-right font-medium">{av.seller_type}</dd>
              </>
            )}
          </dl>
          {av.price_history.length > 0 && (
            <div className="pt-1.5 border-t border-border/60 space-y-1">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Snapshot Timeline ({av.price_history.length})
              </p>
              <div className="max-h-24 overflow-y-auto space-y-0.5 tabular-nums text-[11px]">
                {av.price_history.slice(-6).map((pt, i) => (
                  <div key={`${pt.date}-${i}`} className="flex justify-between text-muted-foreground">
                    <span>{pt.date}</span>
                    <span className="text-foreground font-medium">
                      {pt.price_eur ? formatEur(pt.price_eur) : "—"}
                      {pt.is_under_option ? " · Option" : ""}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Row 2: Structure & Rooms | Energy, Compliance & Financials | Environment, Noise, Transit & GRB */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Card 4: Structure, Rooms & Amenities */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2.5 text-xs">
          <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
            <Building2 className="w-3 h-3 text-primary" />
            Structure, Layout &amp; Amenities
          </div>
          <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums">
            <dt className="text-muted-foreground">Subtype</dt>
            <dd className="text-right font-medium">
              {(s.property_subtype || s.property_type || "—").replace(/_/g, " ")}
            </dd>
            <dt className="text-muted-foreground">Condition / Year</dt>
            <dd className="text-right font-medium">
              {s.building_condition ? formatCondition(s.building_condition) : "—"}
              {s.year_built ? ` · ${s.year_built}` : ""}
            </dd>
            <dt className="text-muted-foreground">Livable / Plot</dt>
            <dd className="text-right font-medium">
              {s.livable_area ? `${Math.round(s.livable_area)} m²` : "—"}
              {s.property_size ? ` / ${Math.round(s.property_size)} m²` : ""}
            </dd>
            <dt className="text-muted-foreground">Living / Kitchen</dt>
            <dd className="text-right font-medium">
              {s.living_room_surface ? `${Math.round(s.living_room_surface)} m²` : "—"} /{" "}
              {s.kitchen_surface ? `${Math.round(s.kitchen_surface)} m²` : "—"}
            </dd>
            {s.kitchen_type && (
              <>
                <dt className="text-muted-foreground">Kitchen type</dt>
                <dd className="text-right font-medium">{s.kitchen_type.replace(/_/g, " ")}</dd>
              </>
            )}
            <dt className="text-muted-foreground">Beds / Baths / WC</dt>
            <dd className="text-right font-medium">
              {fmtVal(s.bedrooms)} / {fmtVal(s.bathrooms)} / {fmtVal(s.toilets)}
            </dd>
            <dt className="text-muted-foreground">Facades / Width</dt>
            <dd className="text-right font-medium">
              {fmtVal(s.facade_count)}
              {s.street_facade_width ? ` · ${s.street_facade_width} m` : ""}
            </dd>
            {(s.floor_number != null || s.total_floors != null) && (
              <>
                <dt className="text-muted-foreground">Floor / Floors</dt>
                <dd className="text-right font-medium">
                  {fmtVal(s.floor_number)} / {fmtVal(s.total_floors)}
                </dd>
              </>
            )}
            {(s.garden_orientation || s.terrace_orientation) && (
              <>
                <dt className="text-muted-foreground">Orientation</dt>
                <dd className="text-right font-medium">
                  {s.garden_orientation || s.terrace_orientation}
                </dd>
              </>
            )}
            {(s.parking_indoor || s.parking_outdoor || s.parking_closed_box) && (
              <>
                <dt className="text-muted-foreground">Parking (in/out/box)</dt>
                <dd className="text-right font-medium">
                  {s.parking_indoor ?? 0} / {s.parking_outdoor ?? 0} / {s.parking_closed_box ?? 0}
                </dd>
              </>
            )}
          </dl>
          {s.amenities.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-2 border-t border-border/60">
              {s.amenities.map((a) => (
                <span key={a} className="px-1.5 py-0.5 rounded bg-muted text-[10px] text-foreground">
                  {a}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Card 5: Energy, Certificates, Flood & Financials */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2.5 text-xs">
          <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
            <Leaf className="w-3 h-3 text-primary" />
            Energy, Compliance &amp; Financials
          </div>
          <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums">
            <dt className="text-muted-foreground">EPC Label / Score</dt>
            <dd className="text-right font-semibold">
              {e.energy_label ?? "—"}
              {e.primary_energy_consumption != null ? ` (${e.primary_energy_consumption} kWh/m²)` : ""}
            </dd>
            <dt className="text-muted-foreground">Heating</dt>
            <dd className="text-right font-medium">{e.heating_type ? e.heating_type.replace(/_/g, " ") : "—"}</dd>
            <dt className="text-muted-foreground">Double glazing / Solar</dt>
            <dd className="text-right font-medium">
              {yesNo(e.has_double_glazing)} / {yesNo(e.has_photovoltaic_panels)}
            </dd>
            <dt className="text-muted-foreground">Renovation rule</dt>
            <dd className="text-right font-medium">{yesNo(e.renovation_obligation)}</dd>
            <dt className="text-muted-foreground">Electrical / Asbestos</dt>
            <dd className="text-right font-medium">
              {fmtVal(e.electrical_cert)} / {fmtVal(e.asbestos_cert)}
            </dd>
            <dt className="text-muted-foreground">Flood P / G score</dt>
            <dd className="text-right font-medium">
              {e.p_score ?? "—"} / {e.g_score ?? "—"}
            </dd>
            {e.land_use_designation && (
              <>
                <dt className="text-muted-foreground">Zoning</dt>
                <dd className="text-right font-medium truncate" title={e.land_use_designation}>
                  {e.land_use_designation.replace(/_/g, " ")}
                </dd>
              </>
            )}
            <dt className="text-muted-foreground">Cadastral income (KI)</dt>
            <dd className="text-right font-medium">{f.cadastral_income ? `€${f.cadastral_income}` : "—"}</dd>
            {f.monthly_costs != null && (
              <>
                <dt className="text-muted-foreground">Monthly charges</dt>
                <dd className="text-right font-medium">€{f.monthly_costs}/mo</dd>
              </>
            )}
            {f.expected_monthly_rental_income != null && (
              <>
                <dt className="text-muted-foreground">Expected rent</dt>
                <dd className="text-right font-medium">€{f.expected_monthly_rental_income}/mo</dd>
              </>
            )}
            <dt className="text-muted-foreground">Sale flags</dt>
            <dd className="text-right font-medium">
              {[
                f.has_starting_price ? "Starting price" : null,
                f.is_biddit_sale ? "Biddit" : null,
                f.is_public_sale ? "Public sale" : null,
                f.is_subject_to_vat ? "VAT (21%)" : null,
                f.is_newly_built ? "New build" : null,
              ]
                .filter(Boolean)
                .join(", ") || "Standard sale"}
            </dd>
          </dl>
        </div>

        {/* Card 6: Flanders / Stad Gent Environment, Noise, Mobility & GRB */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2.5 text-xs">
          <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
            <Trees className="w-3 h-3 text-primary" />
            Ghent Micro-Location &amp; Environment
          </div>
          <dl className="grid grid-cols-2 gap-y-1.5 gap-x-3 tabular-nums">
            <dt className="text-muted-foreground">Mobiscore</dt>
            <dd className="text-right font-semibold text-primary">
              {env.mobiscore != null ? `${env.mobiscore.toFixed(1)} / 10` : "—"}
            </dd>
            <dt className="text-muted-foreground">Noise (day / night)</dt>
            <dd className="text-right font-medium">
              {env.noise_total_db != null ? `${env.noise_total_db} dB` : "—"} /{" "}
              {env.noise_night_db != null ? `${env.noise_night_db} dB` : "—"}
            </dd>
            <dt className="text-muted-foreground">Air NO₂ / PM2.5</dt>
            <dd className="text-right font-medium">
              {env.no2_ugm3 != null ? `${env.no2_ugm3}` : "—"} / {env.pm25_ugm3 != null ? `${env.pm25_ugm3} µg/m³` : "—"}
            </dd>
            <dt className="text-muted-foreground">Park / Water / Trees</dt>
            <dd className="text-right font-medium">
              {fmtMeters(env.dist_park_m)} · {fmtMeters(env.dist_water_m)} · {env.trees_100m ?? "—"} trees
            </dd>
            <dt className="text-muted-foreground">Transit / Tram / Rail</dt>
            <dd className="text-right font-medium">
              {fmtMeters(env.dist_transit_m)} · {fmtMeters(env.dist_tram_m)} · {fmtMeters(env.dist_rail_station_m)}
            </dd>
            <dt className="text-muted-foreground">School / Motorway</dt>
            <dd className="text-right font-medium">
              {fmtMeters(env.dist_school_m)} · {fmtMeters(env.dist_motorway_m)}
            </dd>
            <dt className="text-muted-foreground">Zones (LEZ / Car-free)</dt>
            <dd className="text-right font-medium">
              {yesNo(env.in_lez, "LEZ", "Outside LEZ")}
              {env.in_autovrij ? " · Car-free" : ""}
              {env.parking_zone ? ` · ${env.parking_zone}` : ""}
            </dd>
            <dt className="text-muted-foreground">GRB Parcel / Footprint</dt>
            <dd className="text-right font-medium">
              {env.grb_parcel_area ? `${Math.round(env.grb_parcel_area)} m²` : "—"} /{" "}
              {env.grb_footprint ? `${Math.round(env.grb_footprint)} m²` : "—"}
            </dd>
            <dt className="text-muted-foreground">Heritage / Cambio / EV</dt>
            <dd className="text-right font-medium">
              {env.is_protected_monument
                ? "Monument"
                : env.is_heritage_inventory
                  ? "Inventory"
                  : "None"}{" "}
              · {env.cambio_500m ?? 0} · {env.ev_chargers_400m ?? 0}
            </dd>
          </dl>
        </div>
      </div>

      {/* Full Listing Description */}
      {desc && (
        <div className="rounded-lg border border-border bg-card p-4 space-y-1.5 text-xs">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px] text-muted-foreground">
              <FileText className="w-3 h-3 text-primary" />
              Listing Description {dossier.description_language ? `(${dossier.description_language.toUpperCase()})` : ""}
            </span>
            {desc.length > 420 && (
              <button
                type="button"
                onClick={() => setShowFullDesc((v) => !v)}
                className="text-primary font-medium hover:underline cursor-pointer"
              >
                {showFullDesc ? "Show less" : "Read full description"}
              </button>
            )}
          </div>
          <p className="text-muted-foreground leading-relaxed whitespace-pre-line">{shortDesc}</p>
        </div>
      )}
    </div>
  );
}

export default function DealDossierPanel({
  deal,
  liveCheck,
  checking,
  isCompared,
  canAddCompare,
  onCheckLive,
  onToggleCompare,
}: {
  deal: DealItem;
  liveCheck?: LiveCheckResult;
  checking: boolean;
  isCompared: boolean;
  canAddCompare: boolean;
  onCheckLive: () => void;
  onToggleCompare: () => void;
}) {
  const { data, isLoading, error } = usePropertyDossier(deal.property_id);

  if (isLoading) {
    return (
      <div className="bg-muted/25 border-t border-border/70 px-5 py-6 space-y-3">
        <div className="h-5 w-1/3 bg-muted rounded animate-warm-pulse" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-44 bg-muted/70 rounded-lg animate-warm-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-muted/25 border-t border-border/70 px-5 py-4 text-xs text-muted-foreground">
        Full dossier unavailable for #{deal.property_id} ({(error as Error)?.message ?? "not found"}).{" "}
        <a href={deal.url} target="_blank" rel="noopener noreferrer" className="text-primary underline">
          Open on Immoweb
        </a>
      </div>
    );
  }

  return (
    <DossierBody
      dossier={data}
      liveCheck={liveCheck}
      checking={checking}
      isCompared={isCompared}
      canAddCompare={canAddCompare}
      onCheckLive={onCheckLive}
      onToggleCompare={onToggleCompare}
    />
  );
}
