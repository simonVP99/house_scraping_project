import type { Geometry } from "geojson";
import type { TodayDeal } from "./property";

export interface NeighbourhoodSummary {
  name: string;
  count: number;
  median_price_k: number;
  median_price_sqm: number;
  avg_listing_duration: number;
  median_listing_duration: number;
  pct_under_option: number;
  median_delta_pct: number | null;
  pct_underpriced: number | null;
  geojson: Geometry | null;
}

export interface NeighbourhoodDetail {
  name: string;
  stats: {
    count: number;
    median_price_k: number;
    min_price_k: number;
    max_price_k: number;
    avg_area: number;
    avg_age: number | null;
    top_condition: string | null;
    top_energy_label: string | null;
  };
  trend: Array<{
    month: string;
    city_median_price_k: number | null;
    neighbourhood_median_price_k: number | null;
  }>;
  listings: TodayDeal[];
}

export interface DashboardStats {
  total_properties: number;
  num_neighbourhoods: number;
  median_price_k: number;
  avg_listing_duration: number;
  pct_under_option: number;
  median_delta_pct: number;
}
