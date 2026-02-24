import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import PriceTrendChart from "./PriceTrendChart";
import LiveListingsTable from "./LiveListingsTable";
import { useNeighbourhoodDetail } from "../hooks/useNeighbourhoods";
import { useLiveListings, useRefreshListings } from "../hooks/useLiveListings";
import { RefreshCw, ExternalLink } from "lucide-react";

interface Props {
  name: string;
}

export default function NeighbourhoodDetail({ name }: Props) {
  const [activeTab, setActiveTab] = useState<"overview" | "listings">("overview");
  const { data, isLoading, isError } = useNeighbourhoodDetail(name);
  const { data: listings, isLoading: listingsLoading } = useLiveListings(name, activeTab === "listings");
  const refreshMutation = useRefreshListings();

  if (isLoading) return <div className="text-muted-foreground">Loading details...</div>;
  if (isError || !data) return <div className="text-red-600">Failed to load details</div>;

  const { stats, trend, listings: overviewListings } = data;
  const displayName = name.replace(/_/g, " ");

  return (
    <div className="space-y-4">
      {/* Stats card */}
      <Card className="bg-gradient-to-br from-slate-900 to-blue-900 text-white">
        <CardContent className="pt-6">
          <h2 className="text-xl font-bold mb-4">{displayName}</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { value: stats.count.toLocaleString(), label: "Properties" },
              { value: `${stats.median_price_k.toFixed(0)}k`, label: "Median Price" },
              { value: `${stats.avg_area.toFixed(0)} m\u00B2`, label: "Avg Area" },
              { value: stats.avg_age != null ? `${stats.avg_age.toFixed(0)} yr` : "\u2014", label: "Avg Age" },
              { value: stats.top_condition?.replace(/_/g, " ") ?? "\u2014", label: "Top Condition" },
              { value: stats.top_energy_label ?? "\u2014", label: "Top Energy" },
            ].map(({ value, label }) => (
              <div key={label} className="bg-white/15 rounded-lg p-3 text-center">
                <div className="text-lg font-bold">{value}</div>
                <div className="text-xs text-slate-300 mt-1">{label}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-sm text-slate-400">
            Price range: {stats.min_price_k.toFixed(0)}k\u20AC \u2013 {stats.max_price_k.toFixed(0)}k\u20AC
          </div>
        </CardContent>
      </Card>

      {/* Tab switcher */}
      <div className="flex items-center gap-2 border-b">
        <button
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "overview"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
          onClick={() => setActiveTab("overview")}
        >
          Overview
        </button>
        <button
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "listings"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
          onClick={() => setActiveTab("listings")}
        >
          Live Listings
          {listings && (
            <span className="ml-1.5 inline-flex items-center rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
              {listings.length}
            </span>
          )}
        </button>
        {activeTab === "listings" && (
          <Button
            variant="outline"
            size="sm"
            className="ml-auto mb-1"
            disabled={refreshMutation.isPending}
            onClick={() => refreshMutation.mutate()}
          >
            <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${refreshMutation.isPending ? "animate-spin" : ""}`} />
            {refreshMutation.isPending ? "Refreshing..." : "Refresh"}
          </Button>
        )}
      </div>

      {/* Tab content */}
      {activeTab === "overview" && (
        <>
          {trend.length > 0 && <PriceTrendChart data={trend} neighbourhoodName={name} />}

          {overviewListings.length > 0 && (
            <Card>
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-3">Current Listings ({overviewListings.length})</h3>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Property</TableHead>
                      <TableHead className="text-right">Price (k\u20AC)</TableHead>
                      <TableHead className="text-right">Predicted (k\u20AC)</TableHead>
                      <TableHead className="text-right">\u0394 %</TableHead>
                      <TableHead className="text-right">Area (m\u00B2)</TableHead>
                      <TableHead className="text-right">Beds</TableHead>
                      <TableHead>Condition</TableHead>
                      <TableHead>Energy</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {overviewListings.map((l) => (
                      <TableRow key={l.property_id}>
                        <TableCell>
                          <a
                            href={`https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/${l.property_id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:underline inline-flex items-center gap-1"
                          >
                            {l.property_id}
                            <ExternalLink className="h-3 w-3 flex-shrink-0" />
                          </a>
                        </TableCell>
                        <TableCell className="text-right">{l.price?.toFixed(0)}</TableCell>
                        <TableCell className="text-right">{l.predicted_k?.toFixed(0)}</TableCell>
                        <TableCell className="text-right">
                          {l.delta_pct != null ? (
                            <span className={l.delta_pct > 0 ? "text-green-600" : "text-red-600"}>
                              {l.delta_pct > 0 ? "+" : ""}{l.delta_pct.toFixed(1)}%
                            </span>
                          ) : "\u2014"}
                        </TableCell>
                        <TableCell className="text-right">{l.livable_area}</TableCell>
                        <TableCell className="text-right">{l.number_of_bedrooms}</TableCell>
                        <TableCell>{l.building_condition?.replace(/_/g, " ")}</TableCell>
                        <TableCell>{l.energy_label}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {activeTab === "listings" && (
        <>
          {listingsLoading && <div className="text-muted-foreground">Loading live listings...</div>}
          {listings && <LiveListingsTable listings={listings} />}
          {listings && listings.length === 0 && !listingsLoading && (
            <div className="text-muted-foreground text-center py-8">
              No live listings found for {displayName}.
            </div>
          )}
        </>
      )}
    </div>
  );
}
