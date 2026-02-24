import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface TrendPoint {
  month: string;
  city_median_price_k: number | null;
  neighbourhood_median_price_k: number | null;
}

interface Props {
  data: TrendPoint[];
  neighbourhoodName: string;
}

export default function PriceTrendChart({ data, neighbourhoodName }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          Monthly Median Price: {neighbourhoodName.replace(/_/g, " ")} vs City
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={data}>
            <XAxis dataKey="month" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="city_median_price_k"
              name="City-wide"
              stroke="#94a3b8"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="neighbourhood_median_price_k"
              name={neighbourhoodName.replace(/_/g, " ")}
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
