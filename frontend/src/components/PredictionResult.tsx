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

      {/* Property Regime Advisory (Mixed-Use Commercial / Investment / Live-Work) */}
      {property.regime?.advisory && (
        <div className="rounded-xl border border-warning/35 bg-warning/8 px-4 py-3 text-xs text-foreground flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <span className="font-semibold text-warning uppercase tracking-wider text-[11px] mr-2">
              {property.regime.label}
            </span>
            <span className="text-muted-foreground">{property.regime.advisory}</span>
          </div>
          <span className="shrink-0 text-[11px] font-medium text-muted-foreground tabular-nums">
            Segment MdAPE ~{property.regime.expected_mdape_pct}%
          </span>
        </div>
      )}

      {/* Cross-Modal JEPA Photo-vs-Specs Honesty Card (when photos ran) */}
      {prediction.jepa && (
        <section
          className={`rounded-xl border p-5 space-y-3 ${
            prediction.jepa.tone === "positive"
              ? "border-positive/30 bg-positive/5"
              : prediction.jepa.tone === "negative"
                ? "border-negative/30 bg-negative/5"
                : prediction.jepa.tone === "warning"
                  ? "border-warning/35 bg-warning/8"
                  : "border-border bg-card"
          }`}
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
                  Cross-Modal JEPA Photo Audit
                </span>
                <span
                  className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                    prediction.jepa.tone === "positive"
                      ? "bg-positive/15 text-positive border-positive/30"
                      : prediction.jepa.tone === "negative"
                        ? "bg-negative/15 text-negative border-negative/30"
                        : prediction.jepa.tone === "warning"
                          ? "bg-warning/15 text-warning border-warning/30"
                          : "bg-muted text-foreground border-border"
                  }`}
                >
                  {prediction.jepa.verdict_label}
                </span>
              </div>
              <p className="text-xs text-muted-foreground max-w-3xl leading-relaxed">
                {prediction.jepa.summary}
              </p>
            </div>

            <div className="flex flex-wrap gap-3 text-xs tabular-nums">
              <div className="rounded-lg bg-background/80 border border-border px-3 py-1.5 text-center">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Visual Surprise</p>
                <p className="font-semibold text-foreground">
                  {prediction.jepa.dz_quality > 0 ? "+" : ""}{prediction.jepa.dz_quality.toFixed(2)}σ{" "}
                  <span className="text-muted-foreground font-normal">({Math.round(prediction.jepa.quality_percentile)}th)</span>
                </p>
              </div>
              <div className="rounded-lg bg-background/80 border border-border px-3 py-1.5 text-center">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Contradiction E</p>
                <p className="font-semibold text-foreground">
                  {prediction.jepa.energy_sym.toFixed(2)}{" "}
                  <span className="text-muted-foreground font-normal">({Math.round(prediction.jepa.energy_percentile)}th)</span>
                </p>
              </div>
              <div className="rounded-lg bg-background/80 border border-border px-3 py-1.5 text-center">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Room Spread</p>
                <p className="font-semibold text-foreground">{Math.round(prediction.jepa.cherry_percentile)}th pct</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2 pt-1">
            {prediction.jepa.sub_scores.map((s) => (
              <div key={s.axis} className="rounded-md bg-background/70 border border-border/70 px-2.5 py-1.5 text-center">
                <p className="text-[10px] text-muted-foreground truncate">{s.axis}</p>
                <p
                  className={`text-xs font-semibold tabular-nums ${
                    s.z >= 0.45 ? "text-positive" : s.z <= -0.45 ? "text-negative" : "text-foreground"
                  }`}
                >
                  {s.z > 0 ? "+" : ""}{s.z.toFixed(2)}σ
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

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
