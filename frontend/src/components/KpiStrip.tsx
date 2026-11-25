import type { DashboardStats } from "../types/neighbourhood";
import { formatEurCompact, formatPsm, formatPctSigned } from "../lib/format";

interface Props {
  stats: DashboardStats;
}

export default function KpiStrip({ stats }: Props) {
  const items = [
    {
      label: "Clean Corpus",
      value: stats.total_properties.toLocaleString(),
      sub: `${stats.num_neighbourhoods} Ghent wijken`,
    },
    {
      label: "Median Asking",
      value: formatEurCompact(stats.median_price_eur ?? stats.median_price_k * 1000),
      sub: stats.median_price_sqm ? formatPsm(stats.median_price_sqm) : "city median",
    },
    {
      label: "Avg Market Stay",
      value: `${stats.avg_listing_duration.toFixed(0)}d`,
      sub: "first to last snapshot",
    },
    {
      label: "Under Option",
      value: `${stats.pct_under_option.toFixed(1)}%`,
      sub: "ever marked under option",
    },
    {
      label: "Price Cuts",
      value: stats.pct_price_cut != null ? `${stats.pct_price_cut.toFixed(1)}%` : "—",
      sub: "observed price drops",
    },
    {
      label: "Median OOF Delta",
      value: formatPctSigned(stats.median_delta_pct),
      sub: "Phase-7 fair vs asking",
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {items.map((item) => (
        <div
          key={item.label}
          className="rounded-xl border border-border bg-card px-4 py-3.5"
        >
          <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
            {item.label}
          </p>
          <p className="font-display text-2xl font-bold tabular-nums mt-1 text-foreground">
            {item.value}
          </p>
          <p className="text-[11px] text-muted-foreground mt-0.5 truncate">{item.sub}</p>
        </div>
      ))}
    </div>
  );
}
