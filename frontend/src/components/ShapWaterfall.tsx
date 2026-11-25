import { useState, useMemo } from "react";
import type { ShapData } from "../types/prediction";
import { formatEur, formatEurSigned } from "../lib/format";

interface Props {
  shap: ShapData;
  predictedPrice: number;
}

type ViewMode = "grouped" | "comparable" | "all";

const BASELINE_KEYS = new Set([
  "neighbourhood",
  "postal_code",
  "property_type_HOUSE",
  "property_subtype",
  "dist_from_center",
  "distance_to_street",
  "street_type",
  "statbel_sector",
  "sp_idw_psqm_k5",
  "sp_idw_psqm_k15",
  "sp_idw_psqm_k40",
  "sp_type_wijk_psqm",
  "sp_type_sector_psqm",
  "sp_type_postal_psqm",
  "te_street_log_psqm",
]);

const FEATURE_GROUPS: Record<string, string[]> = {
  "Location & neighbourhood": [
    "neighbourhood", "postal_code", "dist_from_center", "distance_to_street",
    "street_type", "statbel_sector", "latitude", "longitude",
    "sp_idw_psqm_k5", "sp_idw_psqm_k15", "sp_idw_psqm_k40",
    "sp_type_wijk_psqm", "sp_type_sector_psqm", "sp_type_postal_psqm",
    "te_street_log_psqm", "sector_median_income", "log_sector_income_x_area",
    "fl_mobiscore", "fl_mobiscore_x_area", "fl_noise_road_db", "fl_dist_water_m",
    "fl_dist_monument_m", "fl_in_heritage", "fl_heritage_x_epc",
    "gent_parking_zone_ord", "gent_crime_per_1k",
  ],
  "Size & building footprint": [
    "property_type_HOUSE", "property_subtype", "livable_area", "log_livable_area",
    "property_size", "log_property_size_p1", "facade_count", "floor",
    "grb_footprint_area", "grb_parcel_area", "grb_log_parcel_area",
    "grb_log_footprint", "grb_implied_floors", "grb_open_plot_area",
  ],
  "Rooms & layout": [
    "number_of_bedrooms", "number_of_bathrooms", "number_of_toilets",
    "number_of_rooms", "area_per_bedroom", "area_per_room",
    "has_living_room", "has_dining_room", "has_office", "has_dressing_room",
  ],
  "Condition & age": [
    "building_condition", "condition_ordinal", "condition_x_area",
    "year_built", "age", "age_x_condition", "kitchen_type",
  ],
  "Energy & EPC": [
    "energy_label", "epc_ordinal", "epc_ef_area_penalty",
    "primary_energy_consumption", "heating_type",
    "has_double_glazing", "has_photovoltaic_panels", "has_heat_pump",
  ],
  "Outdoor & garden": [
    "has_garden", "garden_surface", "garden_orientation",
    "has_terrace", "terrace_surface", "terrace_orientation",
    "outdoor_surface_total", "log_outdoor_surface", "has_swimming_pool",
  ],
  "Parking & amenities": [
    "parking_indoor", "parking_outdoor", "total_parking",
    "indoor_parking_count", "outdoor_parking_count",
    "has_lift", "has_attic", "has_basement", "has_air_conditioning",
    "cadastral_income", "log_cadastral_income", "ki_per_sqm",
  ],
  "Description, photos & macro": [
    "_other", "txt_blend_psqm", "txt_super_tabresid", "txt_tab_plus_text_psqm",
    "txt_e5l_a5_price", "txt_e5l_a12_psqm", "txt_tfidf_a2_price", "txt_tfidf_a14_psqm",
    "txt_tfidf_spresid", "txt_tfidf_tabresid", "txt_e5l_tabresid", "txt_sgd_psqm",
    "vis_blend_psqm", "vis_blend_tabtxt_resid",
    "section_exterior", "section_interior", "pictures_per_bedroom",
  ],
};

interface BarItem {
  label: string;
  value: number;
}

const MODES: { key: ViewMode; label: string }[] = [
  { key: "grouped", label: "Grouped" },
  { key: "comparable", label: "vs Comparables" },
  { key: "all", label: "Detailed" },
];

export default function ShapWaterfall({ shap, predictedPrice }: Props) {
  const [mode, setMode] = useState<ViewMode>("grouped");

  const { baseLabel, baseValue, bars } = useMemo(() => {
    if (mode === "all") {
      return {
        baseLabel: "Ghent baseline",
        baseValue: shap.base_value_eur,
        bars: shap.features.map((f) => ({ label: f.feature, value: f.value_eur })),
      };
    }

    if (mode === "comparable") {
      let comparableBase = shap.base_value_eur;
      const remaining: BarItem[] = [];
      for (const f of shap.features) {
        if (BASELINE_KEYS.has(f.feature_key)) {
          comparableBase += f.value_eur;
        } else {
          remaining.push({ label: f.feature, value: f.value_eur });
        }
      }
      return {
        baseLabel: "Location & type base",
        baseValue: comparableBase,
        bars: remaining,
      };
    }

    const groupTotals = new Map<string, number>();
    let otherTotal = 0;

    for (const f of shap.features) {
      let matched = false;
      for (const [groupName, keys] of Object.entries(FEATURE_GROUPS)) {
        if (keys.includes(f.feature_key)) {
          groupTotals.set(groupName, (groupTotals.get(groupName) || 0) + f.value_eur);
          matched = true;
          break;
        }
      }
      if (!matched) {
        otherTotal += f.value_eur;
      }
    }

    const grouped: BarItem[] = [];
    for (const [name, total] of groupTotals.entries()) {
      if (Math.abs(total) >= 500) {
        grouped.push({ label: name, value: total });
      }
    }
    if (Math.abs(otherTotal) >= 500) {
      grouped.push({ label: "Other signals", value: otherTotal });
    }
    grouped.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    return {
      baseLabel: "Ghent baseline",
      baseValue: shap.base_value_eur,
      bars: grouped,
    };
  }, [shap, mode]);

  const segments = useMemo(() => {
    const out: Array<BarItem & { start: number; end: number }> = [];
    let acc = baseValue;
    for (const b of bars) {
      const start = acc;
      const end = acc + b.value;
      acc = end;
      out.push({ ...b, start, end });
    }
    return out;
  }, [baseValue, bars]);

  const allValues = [baseValue, predictedPrice, ...segments.flatMap((s) => [s.start, s.end])];
  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const padding = Math.max((rawMax - rawMin) * 0.08, 10_000);
  const minVal = rawMin - padding;
  const maxVal = rawMax + padding;
  const range = Math.max(maxVal - minVal, 1);

  const toPct = (v: number) => Math.max(0, Math.min(100, ((v - minVal) / range) * 100));

  return (
    <section aria-label="Price breakdown" className="rounded-xl border border-border bg-card p-5">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
        <div>
          <h3 className="font-display text-lg font-semibold">Price Drivers (TreeSHAP)</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Exact euro contributions from{" "}
            <span className="font-medium text-foreground">{baseLabel.toLowerCase()}</span>{" "}
            ({formatEur(baseValue)}) to fair value ({formatEur(predictedPrice)})
          </p>
        </div>

        <div className="inline-flex rounded-lg bg-muted p-0.5 border border-border/60" role="tablist" aria-label="Breakdown view">
          {MODES.map((m) => (
            <button
              key={m.key}
              type="button"
              role="tab"
              aria-selected={mode === m.key}
              onClick={() => setMode(m.key)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                mode === m.key
                  ? "bg-card text-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-1.5" role="list" aria-label="Feature price impacts">
        <WaterfallRow
          label={baseLabel}
          valueLabel={formatEur(baseValue)}
          barLeft={0}
          barWidth={toPct(baseValue)}
          colorClass="bg-muted-foreground/25"
          isBase
        />

        {segments.map((seg, i) => {
          const isPositive = seg.value >= 0;
          const left = toPct(Math.min(seg.start, seg.end));
          const right = toPct(Math.max(seg.start, seg.end));
          const width = Math.max(right - left, 0.8);

          return (
            <WaterfallRow
              key={`${seg.label}-${i}`}
              label={seg.label}
              valueLabel={formatEurSigned(seg.value)}
              barLeft={left}
              barWidth={width}
              colorClass={isPositive ? "bg-positive" : "bg-negative"}
              isPositive={isPositive}
            />
          );
        })}

        <div className="pt-1.5 mt-1.5 border-t border-border/70">
          <WaterfallRow
            label="Predicted fair value"
            valueLabel={formatEur(predictedPrice)}
            barLeft={0}
            barWidth={toPct(predictedPrice)}
            colorClass="bg-primary"
            isBase
          />
        </div>
      </div>

      {/* Aligned axis ticks */}
      <div className="grid grid-cols-[120px_1fr_84px] md:grid-cols-[185px_1fr_92px] gap-3 mt-2 text-[10px] text-muted-foreground tabular-nums">
        <span />
        <div className="flex justify-between">
          <span>{formatEur(minVal)}</span>
          <span>{formatEur((minVal + maxVal) / 2)}</span>
          <span>{formatEur(maxVal)}</span>
        </div>
        <span />
      </div>
    </section>
  );
}

function WaterfallRow({
  label,
  valueLabel,
  barLeft,
  barWidth,
  colorClass,
  isBase,
  isPositive,
}: {
  label: string;
  valueLabel: string;
  barLeft: number;
  barWidth: number;
  colorClass: string;
  isBase?: boolean;
  isPositive?: boolean;
}) {
  return (
    <div
      className="grid grid-cols-[120px_1fr_84px] md:grid-cols-[185px_1fr_92px] items-center gap-3 h-7 group"
      role="listitem"
    >
      <div className="text-xs text-right truncate" title={label}>
        <span
          className={
            isBase
              ? "font-semibold text-foreground"
              : "text-muted-foreground group-hover:text-foreground transition-colors"
          }
        >
          {label}
        </span>
      </div>
      <div className="relative h-4 bg-muted/40 rounded overflow-hidden">
        <div
          className={`absolute top-0 h-full rounded transition-all duration-300 ${colorClass}`}
          style={{ left: `${barLeft}%`, width: `${barWidth}%` }}
        />
      </div>
      <div className="text-xs text-right tabular-nums">
        <span
          className={
            isBase
              ? "font-bold text-foreground"
              : isPositive
                ? "font-medium text-positive"
                : "font-medium text-negative"
          }
        >
          {valueLabel}
        </span>
      </div>
    </div>
  );
}
