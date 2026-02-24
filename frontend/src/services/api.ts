import type { PredictionResult, ManualPredictionResult } from "../types/prediction";
import type { NeighbourhoodSummary, NeighbourhoodDetail, DashboardStats } from "../types/neighbourhood";
import type { TodayDeal } from "../types/property";

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

  predictFromUrl: (url: string) =>
    fetchJson<PredictionResult>(`${BASE}/predict/url?url=${encodeURIComponent(url)}`),

  predictManual: (formData: Record<string, unknown>) =>
    fetchJson<ManualPredictionResult>(`${BASE}/predict/manual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    }),
};
