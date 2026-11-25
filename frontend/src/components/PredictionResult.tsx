import { Badge } from "@/components/ui/badge";
import ConformalBandGauge from "./ConformalBandGauge";
import Comparables from "./Comparables";
import FeatureContext from "./FeatureContext";
import ShapWaterfall from "./ShapWaterfall";
import WhatIfSimulator from "./WhatIfSimulator";
import type { PredictionResult } from "../types/prediction";
import {
  formatEur,
  formatEurSigned,
  formatPctSigned,
  formatPsm,
  formatCondition,
  formatNeighbourhood,
} from "../lib/format";

interface Props {
  data: PredictionResult;
  baseFormData?: Record<string, unknown>;
  basePrediction?: number;
  onSimulate?: (nextFormData: Record<string, unknown>) => void;
  onResetSimulation?: () => void;
  isSimulating?: boolean;
}

export default function PredictionResultCard({
  data,
  baseFormData,
  basePrediction,
  onSimulate,
  onResetSimulation,
  isSimulating = false,
}: Props) {
  const { property, prediction } = data;
  const isUnderpriced = prediction.delta_eur != null && prediction.delta_eur > 0;
  const hasBand = prediction.low_eur != null && prediction.high_eur != null;
  const predPsm =
    property.livable_area && property.livable_area > 0
      ? Math.round(prediction.predicted_price / property.livable_area)
      : null;

  const bandLabel =
    prediction.band_position === "below_band"
      ? "Asking below 80% fair range"
      : prediction.band_position === "above_band"
        ? "Asking above 80% fair range"
        : prediction.band_position === "inside_band"
          ? "Asking inside 80% fair range"
          : null;

  return (
    <div className="space-y-6 stagger-children">
      {/* Hero Verdict Card */}
      <section aria-label="Price prediction result" className="rounded-xl border border-border bg-card p-6">
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-6 items-start">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
                Phase-7 Fair Value Estimate
              </p>
              <span className="text-[11px] text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                {prediction.track === "multimodal"
                  ? `Multimodal · ${prediction.n_photos_used || 6} photos + 5-model ensemble`
                  : "Tabular + Text · 5-model ensemble"}
              </span>
              {prediction.track === "multimodal" &&
                prediction.vision_shift_eur != null &&
                Math.abs(prediction.vision_shift_eur) >= 1000 && (
                  <span
                    className={`text-[11px] font-medium px-2 py-0.5 rounded-full border tabular-nums ${
                      prediction.vision_shift_eur > 0
                        ? "bg-positive/10 text-positive border-positive/25"
                        : "bg-negative/10 text-negative border-negative/25"
                    }`}
                  >
                    Photos {formatEurSigned(prediction.vision_shift_eur)}
                    {prediction.vision_shift_pct != null ? ` (${formatPctSigned(prediction.vision_shift_pct)})` : ""}
                  </span>
                )}
            </div>

            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mt-2">
              <p className="font-display text-4xl md:text-5xl font-bold tracking-tight tabular-nums">
                {formatEur(prediction.predicted_price)}
              </p>
              {predPsm != null && (
                <span className="text-sm font-medium text-muted-foreground tabular-nums">
                  {formatPsm(predPsm)}
                </span>
              )}
            </div>

            {prediction.delta_eur != null && prediction.delta_pct != null && (
              <div className="flex flex-wrap items-center gap-2.5 mt-3">
                <span
                  className={`text-base font-semibold tabular-nums ${isUnderpriced ? "text-positive" : "text-negative"}`}
                >
                  {isUnderpriced ? "\u2191" : "\u2193"} {formatEurSigned(prediction.delta_eur)}
                </span>
                <Badge
                  variant={isUnderpriced ? "default" : "destructive"}
                  className={`text-xs font-semibold tabular-nums ${
                    isUnderpriced
                      ? "bg-positive/10 text-positive border border-positive/20 hover:bg-positive/15"
                      : "bg-negative/10 text-negative border border-negative/20 hover:bg-negative/15"
                  }`}
                >
                  {formatPctSigned(prediction.delta_pct)} vs asking {formatEur(property.price)}
                </Badge>
                {bandLabel && (
                  <Badge
                    variant="outline"
                    className={`text-xs font-medium ${
                      prediction.band_position === "below_band"
                        ? "border-positive/30 text-positive bg-positive/5"
                        : prediction.band_position === "above_band"
                          ? "border-negative/30 text-negative bg-negative/5"
                          : "border-border text-muted-foreground"
                    }`}
                  >
                    {bandLabel}
                  </Badge>
                )}
              </div>
            )}
          </div>

          {/* Right-side compact property spec card */}
          <div className="md:text-right text-sm space-y-1 md:max-w-[280px] rounded-lg bg-muted/40 border border-border/60 px-4 py-3">
            {property.address && (
              <p className="font-semibold text-foreground leading-snug line-clamp-2" title={property.address}>
                {property.address}
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              {[
                property.property_type === "HOUSE" ? "House" : property.property_type === "APARTMENT" ? "Apartment" : property.property_type,
                property.livable_area ? `${Math.round(property.livable_area)} m²` : null,
                property.bedrooms ? `${property.bedrooms} bed` : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
            <p className="text-xs text-muted-foreground">
              {[
                property.condition ? formatCondition(property.condition) : null,
                property.energy_label ? `EPC ${property.energy_label}` : null,
                property.year_built ? `Built ${property.year_built}` : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
            {property.neighbourhood && (
              <p className="text-xs font-semibold text-primary pt-0.5">
                {formatNeighbourhood(property.neighbourhood)}
              </p>
            )}
          </div>
        </div>
      </section>

      {/* Visual Conformal Interval Gauge */}
      {hasBand && (
        <ConformalBandGauge
          predicted={prediction.predicted_price}
          low80={prediction.low_eur!}
          high80={prediction.high_eur!}
          low90={prediction.low_90_eur}
          high90={prediction.high_90_eur}
          asking={property.price}
          halfwidthPct={prediction.interval_halfwidth_pct}
          bandPosition={prediction.band_position}
        />
      )}

      {/* Interactive Renovation / What-If Simulator */}
      {onSimulate && onResetSimulation && baseFormData && basePrediction != null && (
        <WhatIfSimulator
          baseFormData={baseFormData}
          basePrediction={basePrediction}
          currentPrediction={prediction.predicted_price}
          onSimulate={onSimulate}
          onReset={onResetSimulation}
          isPending={isSimulating}
        />
      )}

      {/* TreeSHAP Waterfall */}
      {data.shap && data.shap.features.length > 0 && (
        <ShapWaterfall shap={data.shap} predictedPrice={prediction.predicted_price} />
      )}

      {/* Neighbourhood Percentile Context */}
      {data.context && data.context.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-5">
          <FeatureContext
            data={data.context}
            neighbourhood={property.neighbourhood ? formatNeighbourhood(property.neighbourhood) : null}
          />
        </div>
      )}

      {/* Comparable Properties */}
      {data.comparables && (
        <Comparables
          comparables={data.comparables}
          askingPrice={property.price}
          predictedPrice={prediction.predicted_price}
          livableArea={property.livable_area}
        />
      )}
    </div>
  );
}
