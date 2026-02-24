import { useQuery } from "@tanstack/react-query";
import { api } from "../services/api";

export function useNeighbourhoods() {
  return useQuery({ queryKey: ["neighbourhoods"], queryFn: api.getNeighbourhoods });
}

export function useNeighbourhoodDetail(name: string | null) {
  return useQuery({
    queryKey: ["neighbourhood", name],
    queryFn: () => api.getNeighbourhoodDetail(name!),
    enabled: !!name,
  });
}
