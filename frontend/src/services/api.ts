import type { PredictionResult, ManualPredictionResult } from "../types/prediction";
import type { NeighbourhoodSummary, NeighbourhoodDetail, DashboardStats } from "../types/neighbourhood";
import type { TodayDeal, LiveListing } from "../types/property";
import type { EdaReport } from "../types/eda";
import type {
  AffordabilityQuery,
  AffordabilityResponse,
  DealsQuery,
  DealsResponse,
  MarketInsights,
  ModelMetadata,
} from "../types/insights";

const BASE = "/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  getStats: () => fetchJson<DashboardStats>(`${BASE}/stats`),

  getNeighbourhoods: () =>
    fetchJson<{ neighbourhoods: NeighbourhoodSummary[] }>(`${BASE}/neighbourhoods`)
      .then((r) => r.neighbourhoods),

  getNeighbourhoodDetail: (name: string) =>
    fetchJson<NeighbourhoodDetail>(`${BASE}/neighbourhoods/${encodeURIComponent(name)}`),

  getTodayDeals: () =>
    fetchJson<{ properties: TodayDeal[] }>(`${BASE}/properties/today`)
      .then((r) => r.properties),

  geocode: (address: string) =>
    fetchJson<{
      latitude: number;
      longitude: number;
      postal_code: string;
      city: string;
      street: string;
      house_number: string;
      neighbourhood: string | null;
      display_name: string;
    }>(`${BASE}/geocode?address=${encodeURIComponent(address)}`),

  predictFromUrl: (url: string) =>
    fetchJson<PredictionResult>(`${BASE}/predict/url?url=${encodeURIComponent(url)}`),

  predictFromForm: (formData: Record<string, unknown>) =>
    fetchJson<PredictionResult>(`${BASE}/predict/form`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    }),

  predictManual: (formData: Record<string, unknown>) =>
    fetchJson<ManualPredictionResult>(`${BASE}/predict/manual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    }),

  getLiveListings: (neighbourhood?: string) => {
    const params = neighbourhood ? `?neighbourhood=${encodeURIComponent(neighbourhood)}` : "";
    return fetchJson<{ listings: LiveListing[]; date_used: string | null }>(
      `${BASE}/listings/live${params}`,
    ).then((r) => r.listings);
  },

  refreshListings: () =>
    fetchJson<{ count: number; date_used: string | null }>(`${BASE}/listings/refresh`, {
      method: "POST",
    }),

  getListingsCount: () =>
    fetchJson<{ count: number; date_used: string | null }>(`${BASE}/listings/count`),

  getDeals: (params?: DealsQuery) => {
    const qs = new URLSearchParams();
    if (params?.neighbourhood && params.neighbourhood !== "ALL") qs.set("neighbourhood", params.neighbourhood);
    if (params?.property_type && params.property_type !== "ALL") qs.set("property_type", params.property_type);
    if (params?.max_price) qs.set("max_price", String(params.max_price));
    if (params?.min_epc && params.min_epc !== "ANY") qs.set("min_epc", params.min_epc);
    if (params?.min_delta_pct != null) qs.set("min_delta_pct", String(params.min_delta_pct));
    if (params?.below_band_only) qs.set("below_band_only", "true");
    if (params?.limit) qs.set("limit", String(params.limit));
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJson<DealsResponse>(`${BASE}/deals${suffix}`);
  },

  getEdaReport: () => fetchJson<EdaReport>(`${BASE}/eda`),

  getModelMetadata: () => fetchJson<ModelMetadata>(`${BASE}/model/metadata`),

  getMarketInsights: () => fetchJson<MarketInsights>(`${BASE}/insights`),

  evaluateAffordability: (query: AffordabilityQuery) =>
    fetchJson<AffordabilityResponse>(`${BASE}/affordability`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(query),
    }),
};
