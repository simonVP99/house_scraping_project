import { useMemo } from "react";
import { ExternalLink, Scale, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCompareProperties } from "../hooks/useInsights";
import type { PropertyDossier } from "../types/insights";
import {
  formatCondition,
  formatEur,
  formatEurCompact,
  formatEurSigned,
  formatPctSigned,
  formatPsm,
} from "../lib/format";

interface CompareRowSpec {
  label: string;
  section: string;
  render: (d: PropertyDossier) => string;
  numeric?: (d: PropertyDossier) => number | null;
  prefer?: "higher" | "lower";
}

function yesNo(v: boolean | null | undefined) {
  if (v === true) return "Yes";
  if (v === false) return "No";
  return "—";
}

function fmtMeters(m: number | null | undefined) {
  if (m == null) return "—";
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`;
}

const COMPARE_ROWS: CompareRowSpec[] = [
  // Valuation & Deal
  {
    section: "Valuation & Deal",
    label: "Asking Price",
    render: (d) => (d.valuation.asking_eur ? formatEur(d.valuation.asking_eur) : "—"),
    numeric: (d) => d.valuation.asking_eur,
    prefer: "lower",
  },
  {
    section: "Valuation & Deal",
    label: "Phase-7 Fair Value (Multimodal)",
    render: (d) => (d.valuation.predicted_mm_eur ? formatEur(d.valuation.predicted_mm_eur) : "—"),
    numeric: (d) => d.valuation.predicted_mm_eur,
  },
  {
    section: "Valuation & Deal",
    label: "80% Conformal Band",
    render: (d) =>
      d.valuation.low_80_eur && d.valuation.high_80_eur
        ? `${formatEurCompact(d.valuation.low_80_eur)} – ${formatEurCompact(d.valuation.high_80_eur)}`
        : "—",
  },
  {
    section: "Valuation & Deal",
    label: "Value Gap (%)",
    render: (d) =>
      d.valuation.delta_pct != null
        ? `${formatPctSigned(d.valuation.delta_pct)} (${d.valuation.delta_eur ? formatEurSigned(d.valuation.delta_eur) : ""})`
        : "—",
    numeric: (d) => d.valuation.delta_pct,
    prefer: "higher",
  },
  {
    section: "Valuation & Deal",
    label: "Asking €/m²",
    render: (d) => (d.valuation.asking_psm ? formatPsm(d.valuation.asking_psm) : "—"),
    numeric: (d) => d.valuation.asking_psm,
    prefer: "lower",
  },
  {
    section: "Valuation & Deal",
    label: "Fair Value €/m²",
    render: (d) => (d.valuation.fair_psm ? formatPsm(d.valuation.fair_psm) : "—"),
    numeric: (d) => d.valuation.fair_psm,
  },
  {
    section: "Valuation & Deal",
    label: "Photo Price Impact",
    render: (d) =>
      d.valuation.vision_delta_pct != null ? formatPctSigned(d.valuation.vision_delta_pct) : "—",
    numeric: (d) => d.valuation.vision_delta_pct,
    prefer: "higher",
  },
  {
    section: "Valuation & Deal",
    label: "JEPA Visual Surprise (Δz)",
    render: (d) =>
      d.jepa.dz_quality != null
        ? `${d.jepa.dz_quality > 0 ? "+" : ""}${d.jepa.dz_quality.toFixed(2)}σ`
        : "—",
    numeric: (d) => d.jepa.dz_quality,
    prefer: "higher",
  },
  {
    section: "Valuation & Deal",
    label: "JEPA Contradiction Energy",
    render: (d) => (d.jepa.energy_sym != null ? d.jepa.energy_sym.toFixed(2) : "—"),
    numeric: (d) => d.jepa.energy_sym,
    prefer: "lower",
  },
  {
    section: "Valuation & Deal",
    label: "Property Regime",
    render: (d) => d.regime?.label ?? "Standard Residential",
  },

  // Availability & History
  {
    section: "Availability & History",
    label: "Availability Status",
    render: (d) => {
      const s = d.availability.status;
      if (s === "active_recent") return `Active (${d.availability.last_seen ?? "latest"})`;
      if (s === "seen_recently") return `Seen ${d.availability.last_seen ?? "recently"}`;
      if (s === "under_option") return "Under option";
      if (s === "sold_or_rented") return "Sold / Removed";
      return `Last seen ${d.availability.last_seen ?? "earlier"}`;
    },
  },
  {
    section: "Availability & History",
    label: "First → Last Seen",
    render: (d) => `${d.availability.first_seen ?? "—"} → ${d.availability.last_seen ?? "—"}`,
  },
  {
    section: "Availability & History",
    label: "Days Tracked / Scrapes",
    render: (d) =>
      d.availability.listing_duration_days != null
        ? `${d.availability.listing_duration_days}d (${d.availability.n_snapshots ?? 1}×)`
        : "—",
  },
  {
    section: "Availability & History",
    label: "Price Cut Observed",
    render: (d) =>
      d.availability.had_price_cut
        ? `Yes${d.availability.price_cut_eur ? ` (−${formatEurCompact(d.availability.price_cut_eur)})` : ""}`
        : "No",
  },
  {
    section: "Availability & History",
    label: "Agency",
    render: (d) => d.availability.agency ?? "—",
  },

  // Structure & Space
  {
    section: "Structure & Space",
    label: "Type / Subtype",
    render: (d) => (d.structure.property_subtype || d.structure.property_type || "—").replace(/_/g, " "),
  },
  {
    section: "Structure & Space",
    label: "Building Condition",
    render: (d) => (d.structure.building_condition ? formatCondition(d.structure.building_condition) : "—"),
  },
  {
    section: "Structure & Space",
    label: "Year Built",
    render: (d) => (d.structure.year_built != null ? String(d.structure.year_built) : "—"),
    numeric: (d) => d.structure.year_built,
    prefer: "higher",
  },
  {
    section: "Structure & Space",
    label: "Livable Area",
    render: (d) => (d.structure.livable_area ? `${Math.round(d.structure.livable_area)} m²` : "—"),
    numeric: (d) => d.structure.livable_area,
    prefer: "higher",
  },
  {
    section: "Structure & Space",
    label: "Plot Size (Reported / GRB)",
    render: (d) => {
      const rep = d.structure.property_size ? `${Math.round(d.structure.property_size)} m²` : null;
      const grb = d.environment.grb_parcel_area ? `${Math.round(d.environment.grb_parcel_area)} m²` : null;
      if (rep && grb) return `${rep} (GRB ${grb})`;
      return rep || grb || "—";
    },
    numeric: (d) => d.structure.property_size ?? d.environment.grb_parcel_area,
    prefer: "higher",
  },
  {
    section: "Structure & Space",
    label: "Bedrooms / Bathrooms",
    render: (d) => `${d.structure.bedrooms ?? "—"} bed / ${d.structure.bathrooms ?? "—"} bath`,
    numeric: (d) => d.structure.bedrooms,
    prefer: "higher",
  },
  {
    section: "Structure & Space",
    label: "Facades / Width",
    render: (d) =>
      d.structure.facade_count != null
        ? `${d.structure.facade_count}${d.structure.street_facade_width ? ` (${d.structure.street_facade_width} m)` : ""}`
        : "—",
    numeric: (d) => d.structure.facade_count,
    prefer: "higher",
  },
  {
    section: "Structure & Space",
    label: "Garden / Terrace",
    render: (d) => {
      const g = d.structure.garden_surface ? `Garden ${Math.round(d.structure.garden_surface)} m²` : null;
      const t = d.structure.terrace_surface ? `Terrace ${Math.round(d.structure.terrace_surface)} m²` : null;
      return [g, t].filter(Boolean).join(" · ") || "—";
    },
  },
  {
    section: "Structure & Space",
    label: "Parking (in / out / box)",
    render: (d) => {
      const a = d.structure.parking_indoor ?? 0;
      const b = d.structure.parking_outdoor ?? 0;
      const c = d.structure.parking_closed_box ?? 0;
      return a + b + c > 0 ? `${a} / ${b} / ${c}` : "—";
    },
    numeric: (d) => (d.structure.parking_indoor ?? 0) + (d.structure.parking_outdoor ?? 0) + (d.structure.parking_closed_box ?? 0),
    prefer: "higher",
  },
  {
    section: "Structure & Space",
    label: "Amenities",
    render: (d) => (d.structure.amenities.length > 0 ? d.structure.amenities.join(", ") : "—"),
  },

  // Energy, Compliance & Financials
  {
    section: "Energy, Compliance & Financials",
    label: "EPC Label",
    render: (d) =>
      d.energy_compliance.energy_label
        ? `${d.energy_compliance.energy_label}${
            d.energy_compliance.primary_energy_consumption != null
              ? ` (${d.energy_compliance.primary_energy_consumption} kWh/m²)`
              : ""
          }`
        : "—",
    numeric: (d) => d.energy_compliance.primary_energy_consumption,
    prefer: "lower",
  },
  {
    section: "Energy, Compliance & Financials",
    label: "Heating / Glazing / Solar",
    render: (d) =>
      `${d.energy_compliance.heating_type?.replace(/_/g, " ") ?? "—"} · 2×Glaze: ${yesNo(
        d.energy_compliance.has_double_glazing,
      )} · Solar: ${yesNo(d.energy_compliance.has_photovoltaic_panels)}`,
  },
  {
    section: "Energy, Compliance & Financials",
    label: "Renovation Obligation",
    render: (d) => yesNo(d.energy_compliance.renovation_obligation),
  },
  {
    section: "Energy, Compliance & Financials",
    label: "Electrical / Asbestos Cert",
    render: (d) => `${d.energy_compliance.electrical_cert ?? "—"} / ${d.energy_compliance.asbestos_cert ?? "—"}`,
  },
  {
    section: "Energy, Compliance & Financials",
    label: "Flood P / G Score",
    render: (d) => `${d.energy_compliance.p_score ?? "—"} / ${d.energy_compliance.g_score ?? "—"}`,
  },
  {
    section: "Energy, Compliance & Financials",
    label: "Cadastral Income (KI)",
    render: (d) => (d.financial.cadastral_income ? `€${d.financial.cadastral_income}` : "—"),
    numeric: (d) => d.financial.cadastral_income,
    prefer: "lower",
  },
  {
    section: "Energy, Compliance & Financials",
    label: "Monthly Costs",
    render: (d) => (d.financial.monthly_costs != null ? `€${d.financial.monthly_costs}/mo` : "—"),
    numeric: (d) => d.financial.monthly_costs,
    prefer: "lower",
  },

  // Ghent Micro-Location & Environment
  {
    section: "Ghent Micro-Location & Environment",
    label: "Neighbourhood",
    render: (d) => d.neighbourhood ?? "—",
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Distance to Korenmarkt",
    render: (d) => (d.dist_from_center_km != null ? `${d.dist_from_center_km.toFixed(1)} km` : "—"),
    numeric: (d) => d.dist_from_center_km,
    prefer: "lower",
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Mobiscore",
    render: (d) => (d.environment.mobiscore != null ? `${d.environment.mobiscore.toFixed(1)} / 10` : "—"),
    numeric: (d) => d.environment.mobiscore,
    prefer: "higher",
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Noise (Day / Night)",
    render: (d) =>
      d.environment.noise_total_db != null
        ? `${d.environment.noise_total_db} dB / ${d.environment.noise_night_db ?? "—"} dB`
        : "—",
    numeric: (d) => d.environment.noise_total_db,
    prefer: "lower",
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Air Quality (NO₂ / PM2.5)",
    render: (d) =>
      d.environment.no2_ugm3 != null
        ? `${d.environment.no2_ugm3} / ${d.environment.pm25_ugm3 ?? "—"} µg/m³`
        : "—",
    numeric: (d) => d.environment.no2_ugm3,
    prefer: "lower",
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Nearest Park / Trees (100m)",
    render: (d) => `${fmtMeters(d.environment.dist_park_m)} · ${d.environment.trees_100m ?? "—"} trees`,
    numeric: (d) => d.environment.dist_park_m,
    prefer: "lower",
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Nearest Transit / Tram / Rail",
    render: (d) =>
      `${fmtMeters(d.environment.dist_transit_m)} · ${fmtMeters(d.environment.dist_tram_m)} · ${fmtMeters(
        d.environment.dist_rail_station_m,
      )}`,
    numeric: (d) => d.environment.dist_transit_m,
    prefer: "lower",
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Nearest School / Water",
    render: (d) => `${fmtMeters(d.environment.dist_school_m)} · ${fmtMeters(d.environment.dist_water_m)}`,
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "LEZ / Car-free / Parking Zone",
    render: (d) =>
      `${d.environment.in_lez ? "LEZ" : "Outside LEZ"}${d.environment.in_autovrij ? " · Car-free" : ""}${
        d.environment.parking_zone ? ` · ${d.environment.parking_zone}` : ""
      }`,
  },
  {
    section: "Ghent Micro-Location & Environment",
    label: "Heritage Status",
    render: (d) =>
      d.environment.is_protected_monument
        ? "Protected monument"
        : d.environment.is_heritage_inventory
          ? "Heritage inventory"
          : d.environment.in_protected_townscape
            ? "Protected townscape"
            : "None",
  },
];

export default function CompareModal({
  propertyIds,
  onRemove,
  onClear,
  onClose,
}: {
  propertyIds: string[];
  onRemove: (pid: string) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const { data, isLoading, error } = useCompareProperties(propertyIds, propertyIds.length > 0);
  const properties = data?.properties ?? [];

  const groupedSections = useMemo(() => {
    const map = new Map<string, CompareRowSpec[]>();
    for (const row of COMPARE_ROWS) {
      const list = map.get(row.section) ?? [];
      list.push(row);
      map.set(row.section, list);
    }
    return Array.from(map.entries());
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/50 backdrop-blur-[2px] flex items-center justify-center p-3 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-label="Side-by-side property comparison"
      onClick={onClose}
    >
      <div
        className="bg-background border border-border rounded-2xl shadow-2xl w-full max-w-7xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-border bg-card flex items-center justify-between gap-4 shrink-0">
          <div className="flex items-center gap-2.5">
            <Scale className="w-5 h-5 text-primary" />
            <div>
              <h2 className="font-display text-xl font-bold text-foreground">
                Side-by-Side Property Comparison ({properties.length || propertyIds.length} / 4)
              </h2>
              <p className="text-xs text-muted-foreground">
                Best value in each comparable row is highlighted in green. All valuation, specs, compliance, and Flanders/Gent environment fields aligned.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" size="sm" onClick={onClear}>
              Clear all
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={onClose} aria-label="Close comparison">
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="overflow-auto flex-1">
          {isLoading ? (
            <div className="p-8 space-y-3">
              {[0, 1, 2, 3, 4].map((i) => (
                <div key={i} className="h-12 bg-muted/60 rounded animate-warm-pulse" />
              ))}
            </div>
          ) : error ? (
            <div className="p-8 text-sm text-negative">{(error as Error).message}</div>
          ) : properties.length === 0 ? (
            <div className="p-12 text-center text-sm text-muted-foreground">
              No properties selected to compare.
            </div>
          ) : (
            <table className="w-full table-fixed text-xs border-collapse">
              <thead className="sticky top-0 z-10 bg-card border-b border-border shadow-sm">
                <tr>
                  <th className="text-left px-4 py-3 font-semibold text-muted-foreground w-52 align-bottom">
                    Feature
                  </th>
                  {properties.map((p) => (
                    <th key={p.property_id} className="text-left px-4 py-3 align-top border-l border-border/70">
                      <div className="space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <p className="font-display text-sm font-bold text-foreground truncate" title={p.address}>
                              {p.neighbourhood || p.address}
                            </p>
                            <p className="text-[11px] font-normal text-muted-foreground truncate" title={p.address}>
                              {p.address}
                            </p>
                          </div>
                          <button
                            type="button"
                            onClick={() => onRemove(p.property_id)}
                            title="Remove from comparison"
                            className="p-1 rounded hover:bg-muted text-muted-foreground hover:text-foreground cursor-pointer shrink-0"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>

                        {p.photos[0] ? (
                          <div className="h-32 w-full rounded-lg overflow-hidden border border-border bg-muted">
                            <img src={p.photos[0]} alt={p.address} className="w-full h-full object-cover" />
                          </div>
                        ) : (
                          <div className="h-32 w-full rounded-lg border border-dashed border-border bg-muted/40 flex items-center justify-center text-[11px] text-muted-foreground">
                            No cached photo
                          </div>
                        )}

                        <div className="flex items-center justify-between pt-0.5 tabular-nums">
                          <span className="font-bold text-sm text-foreground">
                            {p.valuation.asking_eur ? formatEurCompact(p.valuation.asking_eur) : "—"}
                          </span>
                          <a
                            href={p.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-[11px] font-medium text-primary hover:underline"
                          >
                            #{p.property_id} <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {groupedSections.map(([section, rows]) => (
                  <SectionRows key={section} section={section} rows={rows} properties={properties} />
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

function SectionRows({
  section,
  rows,
  properties,
}: {
  section: string;
  rows: CompareRowSpec[];
  properties: PropertyDossier[];
}) {
  return (
    <>
      <tr className="bg-muted/60 border-y border-border">
        <td colSpan={properties.length + 1} className="px-4 py-2 text-[11px] font-bold uppercase tracking-wider text-primary">
          {section}
        </td>
      </tr>
      {rows.map((row) => {
        let bestVal: number | null = null;
        if (row.numeric && row.prefer && properties.length >= 2) {
          const vals = properties
            .map((p) => row.numeric!(p))
            .filter((v): v is number => v != null && Number.isFinite(v));
          if (vals.length >= 2) {
            const minV = Math.min(...vals);
            const maxV = Math.max(...vals);
            if (minV !== maxV) {
              bestVal = row.prefer === "higher" ? maxV : minV;
            }
          }
        }
        return (
          <tr key={`${section}-${row.label}`} className="border-b border-border/60 hover:bg-muted/20">
            <td className="px-4 py-2.5 font-medium text-muted-foreground align-top">{row.label}</td>
            {properties.map((p) => {
              const numVal = row.numeric ? row.numeric(p) : null;
              const isBest = bestVal != null && numVal != null && numVal === bestVal;
              return (
                <td
                  key={p.property_id}
                  className={`px-4 py-2.5 border-l border-border/60 align-top tabular-nums ${
                    isBest ? "bg-positive/8 text-positive font-semibold" : "text-foreground"
                  }`}
                >
                  {row.render(p)}
                </td>
              );
            })}
          </tr>
        );
      })}
    </>
  );
}
