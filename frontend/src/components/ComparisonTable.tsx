import { useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type SortingState,
  type ColumnDef,
  type ColumnFiltersState,
} from "@tanstack/react-table";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import type { NeighbourhoodSummary } from "../types/neighbourhood";

interface Props {
  neighbourhoods: NeighbourhoodSummary[];
  onSelect: (name: string) => void;
}

const columns: ColumnDef<NeighbourhoodSummary>[] = [
  {
    accessorKey: "name",
    header: "Neighbourhood",
    cell: ({ getValue }) => (
      <span className="font-medium">{(getValue() as string).replace(/_/g, " ")}</span>
    ),
    filterFn: (row, _columnId, filterValue: string) => {
      const name = (row.getValue("name") as string).replace(/_/g, " ").toLowerCase();
      return name.includes(filterValue.toLowerCase());
    },
  },
  {
    accessorKey: "count",
    header: "Listings",
    cell: ({ getValue }) => getValue() as number,
  },
  {
    accessorKey: "median_price_k",
    header: "Median Price (k€)",
    cell: ({ getValue }) => (getValue() as number).toFixed(0),
  },
  {
    accessorKey: "median_price_sqm",
    header: "€/m²",
    cell: ({ getValue }) => (getValue() as number).toFixed(0),
  },
  {
    accessorKey: "avg_listing_duration",
    header: "Avg Duration (d)",
    cell: ({ getValue }) => (getValue() as number).toFixed(0),
  },
  {
    accessorKey: "pct_underpriced",
    header: "% Underpriced",
    cell: ({ getValue }) => {
      const v = getValue() as number | null;
      return v != null ? `${v.toFixed(1)}%` : "\u2014";
    },
    sortUndefined: "last",
  },
  {
    accessorKey: "median_delta_pct",
    header: "Median Δ %",
    cell: ({ getValue }) => {
      const v = getValue() as number | null;
      if (v == null) return "\u2014";
      return (
        <span className={v > 0 ? "text-positive" : "text-negative"}>
          {v > 0 ? "+" : ""}{v.toFixed(1)}%
        </span>
      );
    },
    sortUndefined: "last",
  },
  {
    accessorKey: "pct_under_option",
    header: "% Under Option",
    cell: ({ getValue }) => `${(getValue() as number).toFixed(1)}%`,
  },
];

function SortIcon({ sorted }: { sorted: false | "asc" | "desc" }) {
  if (sorted === "asc") return <ArrowUp className="ml-1 inline h-3.5 w-3.5" />;
  if (sorted === "desc") return <ArrowDown className="ml-1 inline h-3.5 w-3.5" />;
  return <ArrowUpDown className="ml-1 inline h-3.5 w-3.5 opacity-30" />;
}

export default function ComparisonTable({ neighbourhoods, onSelect }: Props) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const table = useReactTable({
    data: neighbourhoods,
    columns,
    state: { sorting, columnFilters, globalFilter },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: (row, _columnId, filterValue: string) => {
      const name = (row.getValue("name") as string).replace(/_/g, " ").toLowerCase();
      return name.includes(filterValue.toLowerCase());
    },
  });

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle>Neighbourhood Comparison</CardTitle>
        <Input
          placeholder="Filter neighbourhoods..."
          value={globalFilter}
          onChange={(e) => setGlobalFilter(e.target.value)}
          className="max-w-xs"
        />
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    const isFirst = header.index === 0;
                    return (
                      <TableHead
                        key={header.id}
                        className={`${!isFirst ? "text-right" : ""} cursor-pointer select-none hover:bg-muted/50`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        <SortIcon sorted={header.column.getIsSorted()} />
                      </TableHead>
                    );
                  })}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => onSelect(row.original.name)}
                  >
                    {row.getVisibleCells().map((cell) => {
                      const isFirst = cell.column.getIndex() === 0;
                      return (
                        <TableCell key={cell.id} className={!isFirst ? "text-right" : ""}>
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-24 text-center">
                    No neighbourhoods match your filter.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Click column headers to sort. {table.getRowModel().rows.length} of {neighbourhoods.length} shown.
        </p>
      </CardContent>
    </Card>
  );
}
