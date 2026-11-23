import { useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type SortingState,
  type ColumnDef,
} from "@tanstack/react-table";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ArrowUpDown, ArrowUp, ArrowDown, ExternalLink } from "lucide-react";
import type { LiveListing } from "../types/property";

interface Props {
  listings: LiveListing[];
}

function fmtEur(v: number) {
  return `\u20AC${v.toLocaleString("nl-BE")}`;
}

const columns: ColumnDef<LiveListing>[] = [
  {
    accessorKey: "address",
    header: "Address",
    cell: ({ row }) => (
      <a
        href={row.original.url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary hover:underline inline-flex items-center gap-1"
      >
        {row.original.address || `Property ${row.original.property_id}`}
        <ExternalLink className="h-3 w-3 flex-shrink-0" />
      </a>
    ),
    filterFn: (row, _columnId, filterValue: string) => {
      const addr = (row.getValue("address") as string).toLowerCase();
      return addr.includes(filterValue.toLowerCase());
    },
  },
  {
    accessorKey: "asking_price",
    header: "Asking (€)",
    cell: ({ getValue }) => fmtEur(getValue() as number),
  },
  {
    accessorKey: "predicted_price",
    header: "Predicted (€)",
    cell: ({ getValue }) => fmtEur(getValue() as number),
  },
  {
    accessorKey: "delta_pct",
    header: "Δ %",
    cell: ({ getValue }) => {
      const v = getValue() as number;
      return (
        <span className={v > 0 ? "text-positive font-medium" : "text-negative font-medium"}>
          {v > 0 ? "+" : ""}{v.toFixed(1)}%
        </span>
      );
    },
  },
  {
    accessorKey: "livable_area",
    header: "Area (m²)",
    cell: ({ getValue }) => {
      const v = getValue() as number | null;
      return v != null ? v : "\u2014";
    },
    sortUndefined: "last",
  },
  {
    accessorKey: "bedrooms",
    header: "Beds",
    cell: ({ getValue }) => {
      const v = getValue() as number | null;
      return v != null ? v : "\u2014";
    },
    sortUndefined: "last",
  },
  {
    accessorKey: "condition",
    header: "Condition",
    cell: ({ getValue }) => {
      const v = getValue() as string | null;
      return v ? v.replace(/_/g, " ") : "\u2014";
    },
  },
  {
    accessorKey: "energy_label",
    header: "Energy",
    cell: ({ getValue }) => (getValue() as string | null) ?? "\u2014",
  },
];

function SortIcon({ sorted }: { sorted: false | "asc" | "desc" }) {
  if (sorted === "asc") return <ArrowUp className="ml-1 inline h-3.5 w-3.5" />;
  if (sorted === "desc") return <ArrowDown className="ml-1 inline h-3.5 w-3.5" />;
  return <ArrowUpDown className="ml-1 inline h-3.5 w-3.5 opacity-30" />;
}

export default function LiveListingsTable({ listings }: Props) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "delta_pct", desc: true },
  ]);
  const [globalFilter, setGlobalFilter] = useState("");

  const table = useReactTable({
    data: listings,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: (row, _columnId, filterValue: string) => {
      const addr = (row.getValue("address") as string).toLowerCase();
      return addr.includes(filterValue.toLowerCase());
    },
  });

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold">
            Properties for Sale ({listings.length})
          </h3>
          <Input
            placeholder="Filter by address..."
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="max-w-xs"
          />
        </div>
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
                  <TableRow key={row.id}>
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
                    No listings match your filter.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Click column headers to sort. {table.getRowModel().rows.length} of {listings.length} shown.
        </p>
      </CardContent>
    </Card>
  );
}
