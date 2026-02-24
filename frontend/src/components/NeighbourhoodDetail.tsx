import { Card, CardContent } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import PriceTrendChart from "./PriceTrendChart";
import { useNeighbourhoodDetail } from "../hooks/useNeighbourhoods";

interface Props {
  name: string;
}

export default function NeighbourhoodDetail({ name }: Props) {
  const { data, isLoading, isError } = useNeighbourhoodDetail(name);

  if (isLoading) return <div className="text-muted-foreground">Loading details...</div>;
  if (isError || !data) return <div className="text-red-600">Failed to load details</div>;

  const { stats, trend, listings } = data;
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

      {/* Price trend */}
      {trend.length > 0 && <PriceTrendChart data={trend} neighbourhoodName={name} />}

      {/* Current listings */}
      {listings.length > 0 && (
        <Card>
          <CardContent className="pt-6">
            <h3 className="font-semibold mb-3">Current Listings ({listings.length})</h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead className="text-right">Price (k\u20AC)</TableHead>
                  <TableHead className="text-right">Predicted (k\u20AC)</TableHead>
                  <TableHead className="text-right">Delta %</TableHead>
                  <TableHead className="text-right">Area (m\u00B2)</TableHead>
                  <TableHead className="text-right">Beds</TableHead>
                  <TableHead>Condition</TableHead>
                  <TableHead>Energy</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {listings.map((l) => (
                  <TableRow key={l.property_id}>
                    <TableCell>{l.property_id}</TableCell>
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
    </div>
  );
}
