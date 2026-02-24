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

export interface LiveListing {
  property_id: string;
  address: string;
  url: string;
  asking_price: number;
  predicted_price: number;
  delta_eur: number;
  delta_pct: number;
  livable_area: number | null;
  bedrooms: number | null;
  condition: string | null;
  energy_label: string | null;
  neighbourhood: string;
}
