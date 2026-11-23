import { Badge } from "@/components/ui/badge";
import Comparables from "./Comparables";
import FeatureContext from "./FeatureContext";
import ShapWaterfall from "./ShapWaterfall";
import type { PredictionResult } from "../types/prediction";

interface Props {
  data: PredictionResult;
}

export default function PredictionResultCard({ data }: Props) {
  const { property, prediction } = data;
  const isUnderpriced = prediction.delta_eur != null && prediction.delta_eur > 0;

  return (
    <div className="stagger-children">
      {/* Price verdict */}
      <section aria-label="Price prediction result" className="animate-fade-in">
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-6 items-start">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
              Predicted value
            </p>
            <p className="font-display text-4xl md:text-5xl font-bold tracking-tight mt-2 tabular-nums">
              {"\u20AC"}{prediction.predicted_price.toLocaleString()}
            </p>
            {prediction.delta_eur != null && prediction.delta_pct != null && (
              <div className="flex flex-wrap items-center gap-3 mt-3">
                <span
                  className={`text-base font-semibold tabular-nums ${isUnderpriced ? "text-positive" : "text-negative"}`}
                  aria-label={`${isUnderpriced ? "Underpriced by" : "Overpriced by"} ${Math.abs(prediction.delta_eur).toLocaleString()} euros`}
                >
                  {isUnderpriced ? "\u2191" : "\u2193"}{" "}
                  {prediction.delta_eur > 0 ? "+" : ""}{"\u20AC"}{prediction.delta_eur.toLocaleString()}
                </span>
                <Badge
                  variant={isUnderpriced ? "default" : "destructive"}
                  className={`text-xs font-semibold tabular-nums ${
                    isUnderpriced
                      ? "bg-positive/10 text-positive border border-positive/20 hover:bg-positive/15"
                      : "bg-negative/10 text-negative border border-negative/20 hover:bg-negative/15"
                  }`}
                >
                  {isUnderpriced ? "\u2191" : "\u2193"}{" "}
                  {prediction.delta_pct > 0 ? "+" : ""}{prediction.delta_pct.toFixed(1)}%
                </Badge>
                <span className="text-sm text-muted-foreground">
                  vs asking {property.price ? `\u20AC${property.price.toLocaleString()}` : "N/A"}
                </span>
              </div>
            )}
          </div>

          {/* Property details */}
          <div className="text-sm space-y-0.5 md:text-right md:max-w-xs">
            <p className="font-medium text-foreground">{property.address}</p>
            <p className="text-muted-foreground">
              {property.property_subtype?.replace(/_/g, " ")} {"\u00B7"} {property.livable_area} m{"\u00B2"} {"\u00B7"} {property.bedrooms} bed
            </p>
            <p className="text-muted-foreground">
              {property.condition?.replace(/_/g, " ")} {"\u00B7"} EPC {property.energy_label} {"\u00B7"} {property.year_built}
            </p>
            {property.neighbourhood && (
              <p className="text-primary font-medium text-xs mt-1">
                {property.neighbourhood.replace(/_/g, " ")}
              </p>
            )}
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="my-10 border-t border-border" />

      {/* Explainability sections with staggered reveal */}
      {data.shap && (
        <section aria-label="Price breakdown" className="animate-fade-in mb-10">
          <ShapWaterfall data={data.shap} predictedPrice={prediction.predicted_price} />
        </section>
      )}

      {(data.context && data.context.length > 0) && (
        <section aria-label="Neighbourhood comparison" className="animate-fade-in mb-10">
          <FeatureContext data={data.context} neighbourhood={property.neighbourhood} />
        </section>
      )}

      {data.comparables && (
        <section aria-label="Comparable properties" className="animate-fade-in">
          <Comparables data={data.comparables} askingPrice={property.price} />
        </section>
      )}
    </div>
  );
}
