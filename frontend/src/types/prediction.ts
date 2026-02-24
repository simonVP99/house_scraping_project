export interface PredictionResult {
  property: {
    property_id: string | null;
    address: string;
    price: number | null;
    livable_area: number | null;
    bedrooms: number | null;
    year_built: number | null;
    condition: string | null;
    energy_label: string | null;
    property_type: string | null;
    property_subtype: string | null;
    neighbourhood: string | null;
  };
  prediction: {
    predicted_price: number;
    delta_eur: number | null;
    delta_pct: number | null;
  };
  form_data: Record<string, unknown>;
}

export interface ManualPredictionResult {
  predicted_price: number;
}
