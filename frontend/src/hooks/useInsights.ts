import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../services/api";
import type { AffordabilityQuery, DealsQuery } from "../types/insights";

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

export function useDeals(params: DealsQuery) {
  return useQuery({
    queryKey: ["deals", params],
    queryFn: () => api.getDeals(params),
    staleTime: 1000 * 60 * 2,
  });
}

export function usePropertyDossier(propertyId: string | null) {
  return useQuery({
    queryKey: ["property-dossier", propertyId],
    queryFn: () => api.getPropertyDossier(propertyId!),
    enabled: Boolean(propertyId),
    staleTime: 1000 * 60 * 10,
  });
}

export function useCompareProperties(propertyIds: string[], enabled: boolean) {
  const key = propertyIds.join(",");
  return useQuery({
    queryKey: ["compare-properties", key],
    queryFn: () => api.compareProperties(propertyIds),
    enabled: enabled && propertyIds.length > 0,
    staleTime: 1000 * 60 * 10,
  });
}
