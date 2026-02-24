import { Card, CardContent } from "@/components/ui/card";
import type { DashboardStats } from "../types/neighbourhood";

interface Props {
  stats: DashboardStats;
}

const items = [
  { key: "total_properties" as const, label: "Properties", format: (v: number) => v.toLocaleString() },
  { key: "num_neighbourhoods" as const, label: "Neighbourhoods", format: (v: number) => String(v) },
  { key: "median_price_k" as const, label: "Median Price", format: (v: number) => `${v.toFixed(0)}k\u20AC` },
  { key: "avg_listing_duration" as const, label: "Avg Duration", format: (v: number) => `${v.toFixed(0)}d` },
  { key: "pct_under_option" as const, label: "Under Option", format: (v: number) => `${v.toFixed(1)}%` },
  { key: "median_delta_pct" as const, label: "Median Delta", format: (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%` },
];

export default function KpiStrip({ stats }: Props) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      {items.map(({ key, label, format }) => (
        <Card key={key}>
          <CardContent className="pt-4 pb-4 text-center">
            <p className="text-2xl font-bold">{format(stats[key])}</p>
            <p className="text-xs text-muted-foreground uppercase tracking-wide mt-1">{label}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
