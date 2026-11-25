import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../services/api";
import type { AffordabilityQuery } from "../types/insights";

export function useMarketInsights() {
  return useQuery({
    queryKey: ["market-insights"],
    queryFn: api.getMarketInsights,
    staleTime: 1000 * 60 * 15,
  });
}

export function useModelMetadata() {
  return useQuery({
    queryKey: ["model-metadata"],
    queryFn: api.getModelMetadata,
    staleTime: 1000 * 60 * 15,
  });
}

export function useAffordability() {
  return useMutation({
    mutationFn: (query: AffordabilityQuery) => api.evaluateAffordability(query),
  });
}
