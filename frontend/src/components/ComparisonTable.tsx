import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { NeighbourhoodSummary } from "../types/neighbourhood";

interface Props {
  neighbourhoods: NeighbourhoodSummary[];
  onSelect: (name: string) => void;
}

export default function ComparisonTable({ neighbourhoods, onSelect }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Neighbourhood Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Neighbourhood</TableHead>
              <TableHead className="text-right">Listings</TableHead>
              <TableHead className="text-right">Median Price (k\u20AC)</TableHead>
              <TableHead className="text-right">\u20AC/m\u00B2</TableHead>
              <TableHead className="text-right">Avg Duration (d)</TableHead>
              <TableHead className="text-right">% Underpriced</TableHead>
              <TableHead className="text-right">Median Delta %</TableHead>
              <TableHead className="text-right">% Under Option</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {neighbourhoods.map((nbh) => (
              <TableRow
                key={nbh.name}
                className="cursor-pointer hover:bg-slate-50"
                onClick={() => onSelect(nbh.name)}
              >
                <TableCell className="font-medium">{nbh.name.replace(/_/g, " ")}</TableCell>
                <TableCell className="text-right">{nbh.count}</TableCell>
                <TableCell className="text-right">{nbh.median_price_k.toFixed(0)}</TableCell>
                <TableCell className="text-right">{nbh.median_price_sqm.toFixed(0)}</TableCell>
                <TableCell className="text-right">{nbh.avg_listing_duration.toFixed(0)}</TableCell>
                <TableCell className="text-right">
                  {nbh.pct_underpriced != null ? `${nbh.pct_underpriced.toFixed(1)}%` : "\u2014"}
                </TableCell>
                <TableCell className="text-right">
                  {nbh.median_delta_pct != null ? (
                    <span className={nbh.median_delta_pct > 0 ? "text-green-600" : "text-red-600"}>
                      {nbh.median_delta_pct > 0 ? "+" : ""}{nbh.median_delta_pct.toFixed(1)}%
                    </span>
                  ) : "\u2014"}
                </TableCell>
                <TableCell className="text-right">{nbh.pct_under_option.toFixed(1)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
