import { useState } from "react";
import type { ShapData, ShapFeature } from "../types/prediction";

type View = "all" | "comparable" | "grouped";

const VIEW_LABELS: Record<View, string> = {
  all: "All Features",
  comparable: "From Comparables",
  grouped: "Grouped",
};

// Features that define "what kind of property in what location" — collapsed in comparable view
const BASELINE_KEYS = new Set([
  "neighbourhood",
  "postal_code",
  "property_type_HOUSE",
  "property_subtype",
  "dist_from_center",
  "distance_to_street",
  "street_type",
]);

// Feature grouping for the grouped view
const FEATURE_GROUPS: [string, Set<string>][] = [
  ["Location & micro-geo", new Set([
    "neighbourhood", "postal_code", "dist_from_center", "distance_to_street", "street_type",
    "fl_mobiscore", "fl_mobiscore_x_area", "fl_heritage_count_150m", "fl_dist_navigable_water_m",
    "fl_no2_2025_ugm3", "sector_median_income", "log_sector_income_x_area",
    "sp_prior_price", "sp_log_prior_price", "sp_same_type_idw_psqm_k15", "sp_same_type_idw_psqm_k5",
    "sp_idw_psqm_k15", "sp_type_wijk_psqm", "sp_type_sector_psqm", "te_street_log_psqm",
  ])],
  ["Type & size", new Set([
    "property_type_HOUSE", "property_subtype", "livable_area", "property_size", "log_property_size_p1",
    "facade_count", "street_facade_width", "total_floors", "floor_number",
    "grb_bldg_footprint", "grb_parcel_area", "grb_log_parcel_area", "grb_open_plot_area",
    "grb_implied_floors", "grb_adjoining_bldgs",
  ])],
  ["Rooms", new Set(["number_of_bedrooms", "number_of_bathrooms", "number_of_toilets", "kitchen_surface", "living_room_surface", "shower_room_count"])],
  ["Condition & age", new Set(["building_condition", "condition_ordinal", "condition_x_area", "age_x_condition", "year_built", "age", "kitchen_type", "is_newly_built", "renovation_obligation"])],
  ["Energy", new Set(["energy_label", "epc_ordinal", "epc_ef_area_penalty", "primary_energy_consumption", "heating_type", "has_heat_pump", "has_photovoltaic_panels", "has_double_glazing", "has_thermic_panels"])],
  ["Outdoor", new Set(["has_garden", "garden_surface", "has_terrace", "terrace_surface", "outdoor_surface_total", "log_outdoor_surface", "has_swimming_pool"])],
  ["Parking & amenities", new Set(["parking_indoor", "parking_outdoor", "total_parking", "has_lift", "has_basement", "has_attic", "monthly_costs", "cadastral_income", "log_cadastral_income", "ki_per_sqm"])],
  ["Description & photos", new Set(["_other", "txt_tab_plus_text_psqm", "txt_blend_psqm", "txt_super_tabresid", "txt_e5l_a5_price", "txt_tfidf_a2_price", "vis_blend_psqm", "vis_blend_tabtxt_resid"])],
];

interface Props {
  data: ShapData;
  predictedPrice: number;
}

interface Segment {
  label: string;
  value: number;
  start: number;
  end: number;
  isPositive: boolean;
}

function buildSegments(
  baseValue: number,
  items: { label: string; value: number }[],
): Segment[] {
  const segments: Segment[] = [];
  let running = baseValue;
  for (const item of items) {
    const start = running;
    const end = running + item.value;
    segments.push({
      label: item.label,
      value: item.value,
      start,
      end,
      isPositive: item.value >= 0,
    });
    running = end;
  }
  return segments;
}

function getComparableItems(features: ShapFeature[]): {
  effectiveBase: number;
  baselineSum: number;
  items: { label: string; value: number }[];
} {
  let baselineSum = 0;
  const remaining: { label: string; value: number }[] = [];
  for (const f of features) {
    if (BASELINE_KEYS.has(f.feature_key)) {
      baselineSum += f.value_eur;
    } else {
      remaining.push({ label: f.feature, value: f.value_eur });
    }
  }
  return { effectiveBase: baselineSum, baselineSum, items: remaining };
}

function getGroupedItems(features: ShapFeature[]): { label: string; value: number }[] {
  const groupSums = new Map<string, number>();
  const assigned = new Set<string>();

  for (const f of features) {
    for (const [groupName, keys] of FEATURE_GROUPS) {
      if (keys.has(f.feature_key)) {
        groupSums.set(groupName, (groupSums.get(groupName) ?? 0) + f.value_eur);
        assigned.add(f.feature_key);
        break;
      }
    }
  }

  // Anything not assigned goes into "Other"
  for (const f of features) {
    if (!assigned.has(f.feature_key)) {
      groupSums.set("Other", (groupSums.get("Other") ?? 0) + f.value_eur);
    }
  }

  // Return in the defined group order, skip groups with ~zero contribution
  const items: { label: string; value: number }[] = [];
  for (const [groupName] of FEATURE_GROUPS) {
    const val = groupSums.get(groupName);
    if (val !== undefined && Math.abs(val) > 50) {
      items.push({ label: groupName, value: Math.round(val) });
    }
  }
  const otherVal = groupSums.get("Other");
  if (otherVal !== undefined && Math.abs(otherVal) > 50) {
    items.push({ label: "Other", value: Math.round(otherVal) });
  }

  // Sort by absolute value descending
  items.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return items;
}

export default function ShapWaterfall({ data, predictedPrice }: Props) {
  const [view, setView] = useState<View>("comparable");
  const { base_value_eur, features } = data;

  let effectiveBase = base_value_eur;
  let baseLabel = "Average price";
  let baseDescription = `dataset average of \u20AC${base_value_eur.toLocaleString()}`;
  let segments: Segment[];

  if (view === "comparable") {
    const comp = getComparableItems(features);
    effectiveBase = base_value_eur + comp.baselineSum;
    baseLabel = "Comparable baseline";
    baseDescription = `comparable properties at \u20AC${effectiveBase.toLocaleString()}`;
    segments = buildSegments(effectiveBase, comp.items);
  } else if (view === "grouped") {
    const items = getGroupedItems(features);
    segments = buildSegments(base_value_eur, items);
  } else {
    const items = features.map((f) => ({ label: f.feature, value: f.value_eur }));
    segments = buildSegments(base_value_eur, items);
  }

  const allValues = [effectiveBase, predictedPrice];
  for (const s of segments) {
    allValues.push(s.start, s.end);
  }
  const minVal = Math.min(...allValues) * 0.95;
  const maxVal = Math.max(...allValues) * 1.05;
  const range = maxVal - minVal || 1;
  const toPercent = (v: number) => ((v - minVal) / range) * 100;

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <h3 className="font-display text-lg font-semibold">Price Breakdown</h3>
        <div className="flex gap-1">
          {(Object.keys(VIEW_LABELS) as View[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-2.5 py-1 text-[11px] rounded-full transition-colors ${
                view === v
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {VIEW_LABELS[v]}
            </button>
          ))}
        </div>
      </div>
      <p className="text-sm text-muted-foreground mt-1 mb-6">
        How each feature shifts the prediction from the{" "}
        <span className="font-semibold tabular-nums">{baseDescription}</span>
      </p>

      <div className="space-y-1" role="list" aria-label="Feature contributions to price">
        {/* Base value */}
        <WaterfallRow
          label={baseLabel}
          valueLabel={`\u20AC${effectiveBase.toLocaleString()}`}
          barLeft={toPercent(minVal)}
          barWidth={toPercent(effectiveBase) - toPercent(minVal)}
          colorClass="bg-muted-foreground/30"
          isBase
        />

        {segments.map((s, i) => {
          const left = toPercent(Math.min(s.start, s.end));
          const width = Math.abs(toPercent(s.end) - toPercent(s.start));
          const colorClass = s.isPositive ? "bg-positive" : "bg-negative";
          const sign = s.value >= 0 ? "+" : "";
          return (
            <WaterfallRow
              key={i}
              label={s.label}
              valueLabel={`${sign}\u20AC${s.value.toLocaleString()}`}
              barLeft={left}
              barWidth={Math.max(width, 0.3)}
              colorClass={colorClass}
              isPositive={s.isPositive}
            />
          );
        })}

        {/* Final prediction */}
        <WaterfallRow
          label="Predicted price"
          valueLabel={`\u20AC${predictedPrice.toLocaleString()}`}
          barLeft={toPercent(minVal)}
          barWidth={toPercent(predictedPrice) - toPercent(minVal)}
          colorClass="bg-primary"
          isBase
        />
      </div>

      <div className="flex justify-between text-[10px] text-muted-foreground mt-2 ml-22 mr-22 md:ml-34 md:mr-26">
        <span className="tabular-nums">{"\u20AC"}{Math.round(minVal).toLocaleString()}</span>
        <span className="tabular-nums">{"\u20AC"}{Math.round(maxVal).toLocaleString()}</span>
      </div>
    </div>
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
    <div className="flex items-center gap-2 h-7 group" role="listitem">
      <div className="w-20 md:w-32 text-[11px] md:text-xs text-right truncate flex-shrink-0">
        <span className={isBase ? "font-semibold text-foreground" : "text-muted-foreground group-hover:text-foreground transition-colors"}>
          {label}
        </span>
      </div>
      <div className="flex-1 relative h-5 bg-muted/40 rounded-sm overflow-hidden">
        <div
          className={`absolute top-0 h-full rounded-sm transition-all ${colorClass} ${isBase ? "opacity-70" : "opacity-80 group-hover:opacity-100"}`}
          style={{
            left: `${barLeft}%`,
            width: `${barWidth}%`,
            minWidth: "2px",
          }}
        />
      </div>
      <div className="w-20 md:w-24 text-[11px] md:text-xs text-right flex-shrink-0 tabular-nums">
        <span
          className={`font-mono ${
            isBase
              ? "font-semibold text-foreground"
              : isPositive
                ? "text-positive"
                : "text-negative"
          }`}
        >
          {valueLabel}
        </span>
      </div>
    </div>
  );
}
