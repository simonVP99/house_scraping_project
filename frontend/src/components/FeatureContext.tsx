import type { FeatureComparison } from "../types/prediction";

interface Props {
  data: FeatureComparison[];
  neighbourhood: string | null;
}

export default function FeatureContext({ data, neighbourhood }: Props) {
  if (data.length === 0) return null;
  const displayNbh = neighbourhood?.replace(/_/g, " ") ?? "neighbourhood";

  return (
    <div>
      <h3 className="font-display text-lg font-semibold">Neighbourhood Context</h3>
      <p className="text-sm text-muted-foreground mt-1 mb-6">
        How this property compares to {displayNbh} medians
      </p>
      <div className="space-y-4">
        {data.map((item) => (
          <FeatureBar key={item.feature} item={item} />
        ))}
      </div>
    </div>
  );
}

function FeatureBar({ item }: { item: FeatureComparison }) {
  const pct = Math.min(Math.max(item.percentile, 2), 98);
  const isAbove = item.direction === "above";

  return (
    <div>
      <div className="flex items-baseline justify-between text-sm mb-1.5">
        <span className="font-medium text-foreground">{item.feature}</span>
        <span className="text-xs text-muted-foreground tabular-nums">
          <span className="font-semibold text-foreground">{formatValue(item.value)}</span>
          {" "}<span className="text-muted-foreground">/ median {formatValue(item.neighbourhood_median)}</span>
        </span>
      </div>
      <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`absolute top-0 left-0 h-full rounded-full transition-all ${isAbove ? "bg-primary" : "bg-warning"}`}
          style={{ width: `${pct}%` }}
        />
        <div className="absolute top-0 left-1/2 h-full w-px bg-foreground/20" />
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
        <span>Low</span>
        <span className={isAbove ? "text-primary" : "text-warning"}>
          {item.percentile}th percentile
        </span>
        <span>High</span>
      </div>
    </div>
  );
}

function formatValue(v: number): string {
  if (v >= 1000) return v.toLocaleString();
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(1);
}
