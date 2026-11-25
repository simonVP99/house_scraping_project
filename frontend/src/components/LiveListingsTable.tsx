import { useState, useMemo } from "react";
import { ArrowUpDown, ArrowUp, ArrowDown, ExternalLink } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import type { LiveListing } from "../types/property";
import { formatEur, formatEurCompact, formatPctSigned, formatCondition } from "../lib/format";

interface Props {
  listings: LiveListing[];
}

type SortKey = "asking_price" | "predicted_price" | "delta_pct" | "livable_area" | "bedrooms";

function SortIcon({ active, dir }: { active: boolean; dir: "asc" | "desc" }) {
  if (!active) return <ArrowUpDown className="ml-1 inline h-3 w-3 opacity-35" />;
  return dir === "asc" ? (
    <ArrowUp className="ml-1 inline h-3 w-3 text-primary" />
  ) : (
    <ArrowDown className="ml-1 inline h-3 w-3 text-primary" />
  );
}

export default function LiveListingsTable({ listings }: Props) {
  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("delta_pct");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const rows = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const filtered = q
      ? listings.filter(
          (l) =>
            (l.address ?? "").toLowerCase().includes(q) ||
            (l.neighbourhood ?? "").toLowerCase().includes(q) ||
            (l.energy_label ?? "").toLowerCase().includes(q),
        )
      : listings;

    return [...filtered].sort((a, b) => {
      const av = Number(a[sortKey] ?? -Infinity);
      const bv = Number(b[sortKey] ?? -Infinity);
      return sortDir === "asc" ? av - bv : bv - av;
    });
  }, [listings, filter, sortKey, sortDir]);

  if (listings.length === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-muted-foreground">
          {rows.length} of {listings.length} live scraped listings
        </span>
        <Input
          placeholder="Filter street, type, EPC..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="max-w-xs h-8 text-xs"
        />
      </div>

      <div className="rounded-lg border border-border overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs">Property</TableHead>
              <TableHead onClick={() => toggleSort("asking_price")} className="text-xs text-right cursor-pointer select-none">
                Asking <SortIcon active={sortKey === "asking_price"} dir={sortDir} />
              </TableHead>
              <TableHead onClick={() => toggleSort("predicted_price")} className="text-xs text-right cursor-pointer select-none">
                Fair Value (80%) <SortIcon active={sortKey === "predicted_price"} dir={sortDir} />
              </TableHead>
              <TableHead onClick={() => toggleSort("delta_pct")} className="text-xs text-right cursor-pointer select-none">
                Delta <SortIcon active={sortKey === "delta_pct"} dir={sortDir} />
              </TableHead>
              <TableHead onClick={() => toggleSort("livable_area")} className="text-xs text-right cursor-pointer select-none">
                Area <SortIcon active={sortKey === "livable_area"} dir={sortDir} />
              </TableHead>
              <TableHead className="text-xs">Cond. / EPC</TableHead>
              <TableHead className="text-xs text-right">Link</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((l) => {
              const isDeal = l.delta_pct != null && l.delta_pct > 0;
              return (
                <TableRow key={l.property_id}>
                  <TableCell>
                    <p className="text-sm font-medium">{l.address || "Listing"}</p>
                    <p className="text-[11px] text-muted-foreground">
                      {l.bedrooms != null ? `${l.bedrooms} bed` : ""}{l.bedrooms != null && l.neighbourhood ? " · " : ""}{l.neighbourhood}
                    </p>
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-sm">
                    {formatEur(l.asking_price)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    <p className="text-sm font-semibold">{formatEur(l.predicted_price)}</p>
                    {l.low_eur != null && l.high_eur != null && (
                      <p className="text-[10px] text-muted-foreground">
                        {formatEurCompact(l.low_eur)}–{formatEurCompact(l.high_eur)}
                      </p>
                    )}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    <span className={`text-sm font-bold ${isDeal ? "text-positive" : "text-negative"}`}>
                      {formatPctSigned(l.delta_pct)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-sm">
                    {l.livable_area ? `${Math.round(l.livable_area)} m²` : "—"}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {formatCondition(l.condition)} {l.energy_label ? `· EPC ${l.energy_label}` : ""}
                  </TableCell>
                  <TableCell className="text-right">
                    <a
                      href={l.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
                    >
                      View <ExternalLink className="w-3 h-3" />
                    </a>
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
