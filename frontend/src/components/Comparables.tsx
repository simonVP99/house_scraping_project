import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import type { ComparablesData, Differentiator, AmenityContribution } from "../types/prediction";

interface Props {
  data: ComparablesData;
  askingPrice: number | null;
}

export default function Comparables({ data, askingPrice }: Props) {
  const {
    properties, median_price_per_sqm, subject_price_per_sqm,
    price_per_sqm_distribution, differentiators, amenity_contributions,
    neighbourhood, median_price_k,
  } = data;
  const displayNbh = neighbourhood?.replace(/_/g, " ") ?? "this area";
  const askingK = askingPrice ? askingPrice / 1000 : null;

  return (
    <div className="space-y-8">
      <div>
        <h3 className="font-display text-lg font-semibold">Market Context</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Compared to {data.count} similar properties in {displayNbh}
        </p>
      </div>

      {/* Price/m² scale */}
      {subject_price_per_sqm != null && median_price_per_sqm != null && (
        <PriceScale
          subjectPsm={subject_price_per_sqm}
          medianPsm={median_price_per_sqm}
          distribution={price_per_sqm_distribution}
          askingK={askingK}
          medianK={median_price_k}
        />
      )}

      {/* Differentiators */}
      {differentiators.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-3">Key differences vs comparables</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {differentiators.map((d) => (
              <DifferentiatorCard key={d.feature} diff={d} />
            ))}
          </div>
        </div>
      )}

      {/* Amenity contributions */}
      {amenity_contributions && amenity_contributions.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-1">Feature value estimates</h4>
          <p className="text-xs text-muted-foreground mb-3">
            Estimated price impact for properties with vs without each feature
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {amenity_contributions.map((a) => (
              <AmenityCard key={a.feature} amenity={a} />
            ))}
          </div>
        </div>
      )}

      {/* Comparable properties table */}
      <details className="group">
        <summary className="text-sm font-medium cursor-pointer text-muted-foreground hover:text-foreground transition-colors">
          Show {properties.length} comparable properties
        </summary>
        <div className="mt-3">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-right">Price/m{"\u00B2"}</TableHead>
                <TableHead className="text-right">Price (k)</TableHead>
                <TableHead className="text-right">Area</TableHead>
                <TableHead className="text-right">Beds</TableHead>
                <TableHead>Condition</TableHead>
                <TableHead>Energy</TableHead>
                <TableHead className="text-right">Age</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {properties.map((p, i) => (
                <TableRow key={i}>
                  <TableCell className="text-right font-medium tabular-nums">
                    {p.price_per_sqm != null ? `\u20AC${p.price_per_sqm.toLocaleString()}` : "\u2014"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{p.price_k?.toFixed(0)}</TableCell>
                  <TableCell className="text-right tabular-nums">{p.livable_area} m{"\u00B2"}</TableCell>
                  <TableCell className="text-right">{p.bedrooms ?? "\u2014"}</TableCell>
                  <TableCell className="text-xs">{p.condition?.replace(/_/g, " ") ?? "\u2014"}</TableCell>
                  <TableCell>{p.energy_label ?? "\u2014"}</TableCell>
                  <TableCell className="text-right">
                    {p.age != null ? `${Math.round(p.age)}yr` : "\u2014"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </details>
    </div>
  );
}

function PriceScale({
  subjectPsm, medianPsm, distribution, askingK, medianK,
}: {
  subjectPsm: number;
  medianPsm: number;
  distribution: number[];
  askingK: number | null;
  medianK: number;
}) {
  if (distribution.length === 0) return null;

  const min = distribution[0];
  const max = distribution[distribution.length - 1];
  const range = max - min || 1;
  const subjectPct = Math.min(100, Math.max(0, ((subjectPsm - min) / range) * 100));
  const medianPct = ((medianPsm - min) / range) * 100;

  const isBelow = subjectPsm < medianPsm;
  const diffPct = Math.round(((subjectPsm - medianPsm) / medianPsm) * 100);

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <h4 className="text-sm font-semibold">Price per m{"\u00B2"}</h4>
        <span className={`text-sm font-bold ${isBelow ? "text-positive" : "text-negative"}`}>
          {isBelow ? "\u2191 " : "\u2193 "}{diffPct > 0 ? "+" : ""}{diffPct}% vs comparables
        </span>
      </div>
      <div className="flex items-center gap-4 text-xs text-muted-foreground mb-3">
        <span>This property: <span className="font-semibold text-foreground tabular-nums">{"\u20AC"}{subjectPsm.toLocaleString()}/m{"\u00B2"}</span></span>
        <span>Median: <span className="font-semibold text-foreground tabular-nums">{"\u20AC"}{medianPsm.toLocaleString()}/m{"\u00B2"}</span></span>
      </div>

      <div className="relative h-6 bg-muted rounded-full overflow-hidden">
        {distribution.map((v, i) => (
          <div
            key={i}
            className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-foreground/15"
            style={{ left: `${((v - min) / range) * 100}%` }}
          />
        ))}
        <div
          className="absolute top-0 h-full w-px bg-foreground/30"
          style={{ left: `${medianPct}%` }}
        />
        <div
          className={`absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full border-2 border-card shadow-md ${isBelow ? "bg-positive" : "bg-negative"}`}
          style={{ left: `${subjectPct}%`, marginLeft: "-7px" }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1 tabular-nums">
        <span>{"\u20AC"}{min.toLocaleString()}</span>
        <span>{"\u20AC"}{max.toLocaleString()}</span>
      </div>

      {askingK != null && (
        <p className="text-xs text-muted-foreground mt-3">
          {isBelow ? (
            <>Asking price ({askingK.toFixed(0)}k) implies a lower price/m{"\u00B2"} than comparables (median {medianK.toFixed(0)}k), suggesting it may be <span className="text-positive font-semibold">underpriced</span>.</>
          ) : (
            <>Asking price ({askingK.toFixed(0)}k) implies a higher price/m{"\u00B2"} than comparables (median {medianK.toFixed(0)}k), suggesting it may be <span className="text-negative font-semibold">overpriced</span>.</>
          )}
        </p>
      )}
    </div>
  );
}

function AmenityCard({ amenity }: { amenity: AmenityContribution }) {
  const valueSign = amenity.estimated_value_k > 0 ? "+" : "";
  const hasIt = amenity.has_it;

  return (
    <div className={`rounded-lg border p-3 text-sm transition-colors ${hasIt ? "border-positive/20 bg-positive/5" : "border-border bg-muted/30"}`}>
      <div className="flex items-center justify-between">
        <span className="font-medium">
          <span className={hasIt ? "text-positive" : "text-muted-foreground"}>{hasIt ? "\u2713" : "\u2717"}</span>
          {" "}{amenity.feature}
          {amenity.surface ? ` (${amenity.surface} m\u00B2)` : ""}
          {amenity.count ? ` (\u00D7${amenity.count})` : ""}
        </span>
        <span className={`font-bold tabular-nums ${amenity.estimated_value_k > 0 ? "text-positive" : "text-negative"}`}>
          {valueSign}{amenity.estimated_value_k.toFixed(0)}k
        </span>
      </div>
      <div className="text-xs text-muted-foreground mt-1">
        {amenity.pct_in_neighbourhood}% in area have this
      </div>
    </div>
  );
}

function DifferentiatorCard({ diff }: { diff: Differentiator }) {
  const isPositive = diff.impact === "positive";
  return (
    <div className={`flex items-start gap-2 rounded-lg border p-3 text-sm ${isPositive ? "border-positive/20 bg-positive/5" : "border-warning/20 bg-warning/5"}`}>
      <span className={`text-sm leading-none mt-0.5 ${isPositive ? "text-positive" : "text-warning"}`}>
        {isPositive ? "\u2191" : "\u2193"}
      </span>
      <div>
        <span className="font-medium">{diff.feature}:</span>{" "}
        <span>{String(diff.subject)}</span>{" "}
        <span className="text-muted-foreground">
          ({diff.direction} than avg {String(diff.comparable_avg)})
        </span>
      </div>
    </div>
  );
}
