import { useState } from "react";
import KpiStrip from "../components/KpiStrip";
import NeighbourhoodMap from "../components/NeighbourhoodMap";
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
        <h1 className="text-2xl font-bold">Neighbourhood Dashboard</h1>
        <p className="text-muted-foreground">Market analytics across Ghent neighbourhoods</p>
      </div>

      {stats && <KpiStrip stats={stats} />}
      {neighbourhoods && (
        <NeighbourhoodMap neighbourhoods={neighbourhoods} onSelect={setSelectedNbh} />
      )}

      {/* ComparisonTable and NeighbourhoodDetail added in next tasks */}
      {selectedNbh && (
        <div className="text-sm text-muted-foreground">
          Selected: {selectedNbh} (detail view coming next)
        </div>
      )}
    </div>
  );
}
