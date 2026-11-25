import type { Geometry } from "geojson";
import type { TodayDeal } from "./property";

export interface NeighbourhoodSummary {
  name: string;
  display_name?: string;
  count: number;
  median_price_k: number;
  median_price_eur?: number;
  median_price_sqm: number;
  avg_listing_duration: number;
  median_listing_duration: number;
  pct_under_option: number;
  pct_price_cut?: number | null;
  median_delta_pct: number | null;
  pct_underpriced: number | null;
  geojson: Geometry | null;
}

export interface RecentScoredListing {
  property_id: string;
  asking_eur: number;
  predicted_eur: number;
  delta_pct: number;
  livable_area: number | null;
  bedrooms: number | null;
  property_type: string;
  condition: string | null;
  energy_label: string | null;
  date: string | null;
}

export interface NeighbourhoodDetail {
  name: string;
  display_name?: string;
  stats: {
    count: number;
    display_name?: string;
    median_price_k: number;
    median_price_eur?: number;
    median_price_sqm?: number | null;
    min_price_k: number;
    max_price_k: number;
    min_price_eur?: number;
    max_price_eur?: number;
    avg_area: number;
    median_area?: number;
    avg_age: number | null;
    pct_houses?: number | null;
    pct_price_cut?: number | null;
    top_condition: string | null;
    top_energy_label: string | null;
  };
  trend: Array<{
    month: string;
    city_median_price_k: number | null;
    neighbourhood_median_price_k: number | null;
    city_median_price_eur?: number | null;
    neighbourhood_median_price_eur?: number | null;
  }>;
  listings: TodayDeal[];
  recent_scored?: RecentScoredListing[];
}

export interface DashboardStats {
  total_properties: number;
  num_neighbourhoods: number;
  median_price_k: number;
  median_price_eur?: number;
  median_price_sqm?: number | null;
  avg_listing_duration: number;
  pct_under_option: number;
  pct_price_cut?: number | null;
  median_delta_pct: number;
}
