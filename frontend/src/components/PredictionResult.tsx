import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PredictionResult } from "../types/prediction";

interface Props {
  data: PredictionResult;
}

export default function PredictionResultCard({ data }: Props) {
  const { property, prediction } = data;
  const isUnderpriced = prediction.delta_eur != null && prediction.delta_eur > 0;

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Asking Price */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">Asking Price</p>
            <p className="text-2xl font-bold">
              {property.price ? `\u20AC${property.price.toLocaleString()}` : "N/A"}
            </p>
          </div>

          {/* Predicted Price */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">Predicted Price</p>
            <p className="text-2xl font-bold">
              {`\u20AC${prediction.predicted_price.toLocaleString()}`}
            </p>
          </div>

          {/* Delta */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">Difference</p>
            {prediction.delta_eur != null && prediction.delta_pct != null ? (
              <div>
                <p className={`text-2xl font-bold ${isUnderpriced ? "text-green-600" : "text-red-600"}`}>
                  {prediction.delta_eur > 0 ? "+" : ""}{`\u20AC${prediction.delta_eur.toLocaleString()}`}
                </p>
                <Badge variant={isUnderpriced ? "default" : "destructive"} className="mt-1">
                  {prediction.delta_pct > 0 ? "+" : ""}{prediction.delta_pct.toFixed(1)}%
                </Badge>
              </div>
            ) : (
              <p className="text-muted-foreground">N/A</p>
            )}
          </div>
        </div>

        {/* Property summary */}
        <div className="mt-6 pt-4 border-t grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div><span className="text-muted-foreground">Address:</span> {property.address}</div>
          <div><span className="text-muted-foreground">Type:</span> {property.property_subtype}</div>
          <div><span className="text-muted-foreground">Area:</span> {property.livable_area} m\u00B2</div>
          <div><span className="text-muted-foreground">Bedrooms:</span> {property.bedrooms}</div>
          <div><span className="text-muted-foreground">Year:</span> {property.year_built}</div>
          <div><span className="text-muted-foreground">Condition:</span> {property.condition}</div>
          <div><span className="text-muted-foreground">Energy:</span> {property.energy_label}</div>
          <div><span className="text-muted-foreground">Neighbourhood:</span> {property.neighbourhood}</div>
        </div>
      </CardContent>
    </Card>
  );
}
