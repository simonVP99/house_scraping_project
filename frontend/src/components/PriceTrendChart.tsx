import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";
import { formatEurCompact } from "../lib/format";

interface TrendPoint {
  month: string;
  city_median_price_k: number | null;
  neighbourhood_median_price_k: number | null;
  city_median_price_eur?: number | null;
  neighbourhood_median_price_eur?: number | null;
}

interface Props {
  data: TrendPoint[];
  neighbourhoodName: string;
}

const COLOR_NEIGHBOURHOOD = "#9a4a2a";
const COLOR_CITY = "#a69586";

export default function PriceTrendChart({ data, neighbourhoodName }: Props) {
  const chartData = data.map((d) => ({
    month: d.month,
    city_eur: d.city_median_price_eur ?? (d.city_median_price_k != null ? d.city_median_price_k * 1000 : null),
    nbh_eur:
      d.neighbourhood_median_price_eur ??
      (d.neighbourhood_median_price_k != null ? d.neighbourhood_median_price_k * 1000 : null),
  }));

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="mb-4">
        <h3 className="font-display text-base font-semibold">
          Monthly Median Price: {neighbourhoodName} vs Ghent
        </h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Continuous calendar axis — gaps indicate months without sufficient scraped listings
        </p>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={chartData} margin={{ top: 5, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" vertical={false} />
          <XAxis
            dataKey="month"
            tick={{ fontSize: 10, fill: "#7d6f63" }}
            axisLine={{ stroke: "#d6cdc4" }}
            tickLine={false}
            minTickGap={24}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#7d6f63" }}
            axisLine={false}
            tickLine={false}
            width={52}
            tickFormatter={(v) => formatEurCompact(Number(v))}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#faf8f5",
              border: "1px solid #d6cdc4",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(value) => formatEurCompact(Number(value ?? 0))}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          <Line
            type="monotone"
            dataKey="city_eur"
            name="Ghent City-wide"
            stroke={COLOR_CITY}
            strokeWidth={1.75}
            strokeDasharray="4 3"
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="nbh_eur"
            name={neighbourhoodName}
            stroke={COLOR_NEIGHBOURHOOD}
            strokeWidth={2.5}
            dot={{ r: 2, fill: COLOR_NEIGHBOURHOOD }}
            connectNulls={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
