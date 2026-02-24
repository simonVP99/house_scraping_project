export interface TodayDeal {
  property_id: string;
  price: number;
  predicted_k: number;
  delta_pct: number;
  livable_area: number;
  number_of_bedrooms: number;
  building_condition: string;
  energy_label: string;
  neighbourhood?: string;
  postal_code?: string;
}
