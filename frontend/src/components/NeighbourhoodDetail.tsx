import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import PriceTrendChart from "./PriceTrendChart";
import LiveListingsTable from "./LiveListingsTable";
import { useNeighbourhoodDetail } from "../hooks/useNeighbourhoods";
import { useLiveListings, useRefreshListings } from "../hooks/useLiveListings";
import { RefreshCw } from "lucide-react";

interface Props {
  name: string;
}

export default function NeighbourhoodDetail({ name }: Props) {
  const { data, isLoading, isError } = useNeighbourhoodDetail(name);
  const { data: listings, isLoading: listingsLoading } = useLiveListings(name, true);
  const refreshMutation = useRefreshListings();

  if (isLoading) return <div className="text-muted-foreground">Loading details...</div>;
  if (isError || !data) return <div className="text-negative">Failed to load details</div>;

  const { stats, trend } = data;
  const displayName = name.replace(/_/g, " ");

  return (
    <div className="space-y-6">
      {/* Stats card */}
      <Card className="bg-foreground text-background">
        <CardContent className="pt-6">
          <h2 className="font-display text-2xl font-bold mb-4">{displayName}</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { value: stats.count.toLocaleString(), label: "Properties" },
              { value: `${stats.median_price_k.toFixed(0)}k`, label: "Median Price" },
              { value: `${stats.avg_area.toFixed(0)} m²`, label: "Avg Area" },
              { value: stats.avg_age != null ? `${stats.avg_age.toFixed(0)} yr` : "—", label: "Avg Age" },
              { value: stats.top_condition?.replace(/_/g, " ") ?? "—", label: "Top Condition" },
              { value: stats.top_energy_label ?? "—", label: "Top Energy" },
            ].map(({ value, label }) => (
              <div key={label} className="bg-background/10 rounded-lg p-3 text-center">
                <div className="text-lg font-bold tabular-nums">{value}</div>
                <div className="text-xs text-background/60 mt-1">{label}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-sm text-background/50">
            Price range: {stats.min_price_k.toFixed(0)}k€ – {stats.max_price_k.toFixed(0)}k€
          </div>
        </CardContent>
      </Card>

      {/* Price trend */}
      {trend.length > 0 && <PriceTrendChart data={trend} neighbourhoodName={name} />}

      {/* Live listings with refresh */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-display text-lg font-semibold">
            Listings in {displayName}
            {listings && (
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                ({listings.length})
              </span>
            )}
          </h3>
          <Button
            variant="outline"
            size="sm"
            disabled={refreshMutation.isPending}
            onClick={() => refreshMutation.mutate()}
          >
            <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${refreshMutation.isPending ? "animate-spin" : ""}`} />
            {refreshMutation.isPending ? "Refreshing..." : "Refresh"}
          </Button>
        </div>

        {listingsLoading && (
          <div className="text-muted-foreground py-8 text-center">Loading listings...</div>
        )}
        {listings && listings.length > 0 && <LiveListingsTable listings={listings} />}
        {listings && listings.length === 0 && !listingsLoading && (
          <div className="text-muted-foreground text-center py-8">
            No listings found for {displayName}.
          </div>
        )}
      </div>
    </div>
  );
}
