export interface TrackMetrics {
  mdape: number;
  mape: number;
  hit_10: number;
  hit_15: number;
  mae: number;
  rmse: number;
  log_rmse: number;
  r2: number;
  bias_pct: number;
}

export interface TrackSummary {
  n_gbdt_features: number;
  n_nn_features: number;
  oof_normal: TrackMetrics;
  temporal_2026_normal: TrackMetrics;
  conformal_80: {
    oof_cov_pct: number;
    oof_median_hw_pct: number;
    tmp_cov_pct: number;
    tmp_median_hw_pct: number;
    q80: number;
    q90: number;
  };
  huber_coefs: Record<string, number>;
  huber_intercept: number;
}

export interface ModelMetadata {
  model_version: string;
  git_sha: string;
  trained_at_utc: string;
  n_train_rows: number;
  n_normal_rows: number;
  agency_blind: boolean;
  uses_qwen3: boolean;
  deflator_formula: string;
  cpi_ref_2025: number;
  hpi_ref_2025: number;
  latest_deflator: number;
  members: string[];
  tracks: {
    tabular_text: TrackSummary;
    multimodal: TrackSummary;
  };
}

export interface AgencyAuditRow {
  agency: string;
  listings: number;
  median_asking_eur: number;
  median_fair_eur: number;
  median_premium_pct: number;
  pct_overpriced_10: number;
  pct_underpriced_10: number;
  pct_price_cut: number;
  median_duration_days: number;
  median_jepa_dz?: number | null;
}

export interface EpcLabelRow {
  label: string;
  count: number;
  median_price_eur: number;
  median_price_sqm: number;
  median_area: number;
  vs_c_psqm_pct: number;
  mandatory_renovation: boolean;
}

export interface EpcBreakdown {
  property_type: string;
  labels: EpcLabelRow[];
}

export interface ConditionLadderRow {
  code: string;
  label: string;
  count: number;
  median_price_eur: number;
  median_price_sqm: number;
  median_area: number;
  pct_houses: number;
}

export interface OverpricingBucket {
  bucket: string;
  count: number;
  pct_price_cut: number;
  pct_under_option: number;
  avg_duration_days: number;
  median_duration_days: number;
}

export interface ErrorBandRow {
  band: string;
  count: number;
  mdape: number;
  hit_10: number;
  hit_15: number;
  mae_eur: number;
}

export interface MonthlyPoint {
  month: string;
  listings: number;
  median_price_eur: number;
  median_fair_eur: number;
  median_price_sqm: number;
}

export interface JepaEnergyQuintile {
  quintile: string;
  count: number;
  mdape: number;
  p80_err: number;
  pct_price_cut: number;
  pct_under_option: number;
  avg_duration_days: number;
}

export interface JepaExampleListing {
  property_id: string;
  url: string;
  neighbourhood: string;
  property_type: string;
  livable_area: number | null;
  energy_label: string | null;
  condition: string | null;
  asking_eur: number;
  fair_eur: number;
  vision_shift_pct: number;
  dz_quality: number;
  energy_sym: number;
  had_price_cut: boolean;
  duration_days: number;
  title: string | null;
}

export interface PropertyRegimeRow {
  code: string;
  label: string;
  count: number;
  median_asking_eur: number;
  mdape: number;
  p80_err: number;
  avg_duration_days: number;
  median_jepa_energy: number | null;
}

export interface JepaForensics {
  n_scored: number;
  energy_quintiles: JepaEnergyQuintile[];
  fair_priced_controlled: {
    count: number;
    q1_duration_days: number;
    q5_duration_days: number;
    q1_price_cut_pct: number;
    q5_price_cut_pct: number;
    q1_under_option_pct: number;
    q5_under_option_pct: number;
  };
  condition_honesty: {
    total_claimed_top: number;
    bottom15_looks_dated: {
      count: number;
      pct_price_cut: number;
      median_vision_shift_pct: number;
      avg_duration_days: number;
    };
    top15_looks_sharp: {
      count: number;
      pct_price_cut: number;
      median_vision_shift_pct: number;
      avg_duration_days: number;
    };
  };
  cherry_quintiles: { quintile: string; count: number; mdape: number; p80_err: number }[];
  regime_breakdown: PropertyRegimeRow[];
  examples: {
    overclaimed: JepaExampleListing[];
    hidden_gems: JepaExampleListing[];
  };
}

export interface MarketInsights {
  generated_from: string;
  corpus: {
    total_properties: number;
    normal_properties: number;
    raw_snapshots: number;
    neighbourhoods_mapped: number;
    date_min: string | null;
    date_max: string | null;
    median_price_eur: number;
    median_price_sqm: number;
  };
  model_metadata: ModelMetadata;
  agency_audit: AgencyAuditRow[];
  epc_cliff: EpcBreakdown[];
  condition_ladder: ConditionLadderRow[];
  overpricing_forensics: OverpricingBucket[];
  vision_impact: {
    multimodal_mdape: number;
    tabular_text_mdape: number;
    mdape_gain_pts: number;
    median_abs_shift_pct: number;
    p90_abs_shift_pct: number;
    pct_shifted_over_5pct: number;
    histogram: { lo: number; hi: number; mid: number; count: number }[];
  };
  jepa_forensics?: JepaForensics | null;
  error_by_band: ErrorBandRow[];
  monthly_series: MonthlyPoint[];
}

export interface AffordabilityQuery {
  budget_eur: number;
  property_type: "HOUSE" | "APARTMENT";
  livable_area: number;
  bedrooms: number;
  condition: string;
  condition_mode?: "min" | "exact";
  energy_label: string;
  epc_mode?: "min" | "exact";
  year_built: number;
  has_garden?: boolean;
  has_terrace?: boolean;
  condition_tiers?: string[];
  energy_tiers?: string[];
}

export interface AffordabilityNeighbourhood {
  name: string;
  display_name: string;
  count: number;
  matching_count?: number;
  lat: number;
  lon: number;
  predicted_eur: number;
  entry_eur?: number;
  low_eur: number;
  high_eur: number;
  low_90_eur: number;
  high_90_eur: number;
  interval_halfwidth_pct: number;
  budget_gap_eur: number;
  budget_gap_pct: number;
  prob_affordable: number;
  tier: "comfortable" | "realistic" | "stretch" | "out_of_reach";
  median_price_sqm: number;
  median_dist_center_km: number | null;
  geojson: GeoJSON.GeoJsonObject | null;
}

export interface DealItem {
  property_id: string;
  address: string;
  url: string;
  asking_price: number;
  predicted_price: number;
  low_eur: number;
  high_eur: number;
  interval_halfwidth_pct: number;
  band_position: "below_band" | "inside_band" | "above_band";
  delta_eur: number;
  delta_pct: number;
  livable_area: number | null;
  bedrooms: number | null;
  condition: string | null;
  energy_label: string | null;
  property_type: string | null;
  neighbourhood: string | null;
  date_obtained: string | null;
  had_price_cut?: boolean;
  property_regime?: string | null;
  jepa_dz_quality?: number | null;
  jepa_energy_sym?: number | null;
  source: "live" | "corpus_recent" | string;
}

export interface DealsQuery {
  neighbourhood?: string;
  property_type?: string;
  max_price?: number;
  min_epc?: string;
  min_delta_pct?: number;
  below_band_only?: boolean;
  limit?: number;
}

export interface DealsResponse {
  deals: DealItem[];
  total: number;
  source: "live_scrape" | "corpus_recent" | "empty" | string;
  date_used: string | null;
}

export interface AffordabilityResponse {
  query: AffordabilityQuery;
  neighbourhoods: AffordabilityNeighbourhood[];
  model_version: string;
}
