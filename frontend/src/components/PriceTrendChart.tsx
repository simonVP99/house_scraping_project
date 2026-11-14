import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent } from "@/components/ui/card";

interface TrendPoint {
  month: string;
  city_median_price_k: number | null;
  neighbourhood_median_price_k: number | null;
}

interface Props {
  data: TrendPoint[];
  neighbourhoodName: string;
}

// Terracotta primary and warm muted for city-wide
const COLOR_NEIGHBOURHOOD = "#9a4a2a"; // warm terracotta
const COLOR_CITY = "#b8a596"; // warm muted stone

export default function PriceTrendChart({ data, neighbourhoodName }: Props) {
  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="font-display text-base font-semibold mb-4">
          Monthly Median Price: {neighbourhoodName.replace(/_/g, " ")} vs City
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <XAxis
              dataKey="month"
              tick={{ fontSize: 11, fill: "#8a7a6e" }}
              axisLine={{ stroke: "#d6cdc4" }}
              tickLine={{ stroke: "#d6cdc4" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "#8a7a6e" }}
              axisLine={{ stroke: "#d6cdc4" }}
              tickLine={{ stroke: "#d6cdc4" }}
              tickFormatter={(v) => `${v}k`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#faf8f5",
                border: "1px solid #d6cdc4",
                borderRadius: "6px",
                fontSize: "12px",
              }}
              formatter={(value) => `${Number(value ?? 0).toFixed(0)}k`}
            />
            <Legend
              wrapperStyle={{ fontSize: "12px" }}
            />
            <Line
              type="monotone"
              dataKey="city_median_price_k"
              name="City-wide"
              stroke={COLOR_CITY}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="neighbourhood_median_price_k"
              name={neighbourhoodName.replace(/_/g, " ")}
              stroke={COLOR_NEIGHBOURHOOD}
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
