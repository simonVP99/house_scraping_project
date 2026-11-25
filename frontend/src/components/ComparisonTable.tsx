import { useState, useMemo } from "react";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import type { NeighbourhoodSummary } from "../types/neighbourhood";
import { formatEurCompact, formatPsm, formatPctSigned, formatNeighbourhood } from "../lib/format";

interface Props {
  neighbourhoods: NeighbourhoodSummary[];
  selected?: string | null;
  onSelect: (name: string) => void;
}

type SortKey =
  | "name"
  | "count"
  | "median_price_eur"
  | "median_price_sqm"
  | "avg_listing_duration"
  | "pct_underpriced"
  | "median_delta_pct"
  | "pct_under_option";

function SortIcon({ active, dir }: { active: boolean; dir: "asc" | "desc" }) {
  if (!active) return <ArrowUpDown className="ml-1 inline h-3 w-3 opacity-35" />;
  return dir === "asc" ? (
    <ArrowUp className="ml-1 inline h-3 w-3 text-primary" />
  ) : (
    <ArrowDown className="ml-1 inline h-3 w-3 text-primary" />
  );
}

export default function ComparisonTable({ neighbourhoods, selected, onSelect }: Props) {
  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("median_price_sqm");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "name" ? "asc" : "desc");
    }
  };

  const rows = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const filtered = q
      ? neighbourhoods.filter((n) =>
          formatNeighbourhood(n.name, n.display_name).toLowerCase().includes(q),
        )
      : neighbourhoods;

    return [...filtered].sort((a, b) => {
      if (sortKey === "name") {
        const na = formatNeighbourhood(a.name, a.display_name);
        const nb = formatNeighbourhood(b.name, b.display_name);
        return sortDir === "asc" ? na.localeCompare(nb) : nb.localeCompare(na);
      }
      const getVal = (n: NeighbourhoodSummary): number => {
        if (sortKey === "median_price_eur") return n.median_price_eur ?? n.median_price_k * 1000;
        const v = n[sortKey];
        return typeof v === "number" && Number.isFinite(v) ? v : -Infinity;
      };
      return sortDir === "asc" ? getVal(a) - getVal(b) : getVal(b) - getVal(a);
    });
  }, [neighbourhoods, filter, sortKey, sortDir]);

  const headers: { key: SortKey; label: string; align?: "right" }[] = [
    { key: "name", label: "Neighbourhood" },
    { key: "count", label: "Listings", align: "right" },
    { key: "median_price_eur", label: "Median Price", align: "right" },
    { key: "median_price_sqm", label: "€/m²", align: "right" },
    { key: "avg_listing_duration", label: "Avg Stay", align: "right" },
    { key: "pct_underpriced", label: "% Below Fair", align: "right" },
    { key: "median_delta_pct", label: "Phase-7 Δ", align: "right" },
    { key: "pct_under_option", label: "Under Option", align: "right" },
  ];

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="px-5 py-4 border-b border-border flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-display text-lg font-semibold">All Ghent Neighbourhoods</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Click any row to inspect monthly trend and Phase-7 scored listings · {rows.length} of {neighbourhoods.length} shown
          </p>
        </div>
        <Input
          placeholder="Filter neighbourhoods..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="max-w-xs"
        />
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              {headers.map((h) => (
                <TableHead
                  key={h.key}
                  onClick={() => toggleSort(h.key)}
                  className={`cursor-pointer select-none whitespace-nowrap text-xs hover:text-foreground ${
                    h.align === "right" ? "text-right" : ""
                  }`}
                >
                  {h.label}
                  <SortIcon active={sortKey === h.key} dir={sortDir} />
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((n) => {
              const isSelected = selected === n.name;
              const label = formatNeighbourhood(n.name, n.display_name);
              const priceEur = n.median_price_eur ?? n.median_price_k * 1000;
              return (
                <TableRow
                  key={n.name}
                  onClick={() => onSelect(n.name)}
                  className={`cursor-pointer transition-colors ${
                    isSelected ? "bg-primary/8 hover:bg-primary/10" : "hover:bg-muted/50"
                  }`}
                >
                  <TableCell className="font-medium">
                    <span className="flex items-center gap-2">
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-primary shrink-0" />}
                      {label}
                    </span>
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {n.count.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right tabular-nums font-medium">
                    {formatEurCompact(priceEur)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums font-semibold">
                    {formatPsm(n.median_price_sqm)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {n.avg_listing_duration.toFixed(0)}d
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {n.pct_underpriced != null ? `${n.pct_underpriced.toFixed(1)}%` : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums font-medium">
                    {n.median_delta_pct != null ? (
                      <span className={n.median_delta_pct >= 0 ? "text-positive" : "text-negative"}>
                        {formatPctSigned(n.median_delta_pct)}
                      </span>
                    ) : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {n.pct_under_option.toFixed(1)}%
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
