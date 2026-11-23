import { useState } from "react";
import KpiStrip from "../components/KpiStrip";
import NeighbourhoodMap from "../components/NeighbourhoodMap";
import ComparisonTable from "../components/ComparisonTable";
import NeighbourhoodDetail from "../components/NeighbourhoodDetail";
import { useStats } from "../hooks/useStats";
import { useNeighbourhoods } from "../hooks/useNeighbourhoods";

export default function NeighbourhoodsPage() {
  const [selectedNbh, setSelectedNbh] = useState<string | null>(null);
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data: neighbourhoods, isLoading: nbhLoading } = useNeighbourhoods();

  if (statsLoading || nbhLoading) {
    return <div className="text-muted-foreground">Loading dashboard...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight">Neighbourhood Dashboard</h1>
        <p className="text-muted-foreground">Market analytics across Ghent neighbourhoods</p>
      </div>

      {stats && <KpiStrip stats={stats} />}
      {neighbourhoods && (
        <NeighbourhoodMap neighbourhoods={neighbourhoods} onSelect={setSelectedNbh} />
      )}

      {neighbourhoods && (
        <ComparisonTable neighbourhoods={neighbourhoods} onSelect={setSelectedNbh} />
      )}

      {selectedNbh && <NeighbourhoodDetail name={selectedNbh} />}
    </div>
  );
}
