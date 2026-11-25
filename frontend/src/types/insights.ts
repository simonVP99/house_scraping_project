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

export type DealAvailabilityStatus =
  | "active_recent"
  | "seen_recently"
  | "under_option"
  | "sold_or_rented"
  | "likely_gone";

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
  property_size?: number | null;
  bedrooms: number | null;
  bathrooms?: number | null;
  year_built?: number | null;
  condition: string | null;
  energy_label: string | null;
  primary_energy_consumption?: number | null;
  property_type: string | null;
  property_subtype?: string | null;
  neighbourhood: string | null;
  postal_code?: string | null;
  dist_from_center?: number | null;
  has_garden?: boolean;
  garden_surface?: number | null;
  has_terrace?: boolean;
  terrace_surface?: number | null;
  has_lift?: boolean;
  has_photovoltaic_panels?: boolean;
  has_heat_pump?: boolean;
  cadastral_income?: number | null;
  date_obtained: string | null;
  first_seen?: string | null;
  last_seen?: string | null;
  days_since_last_seen?: number | null;
  latest_scrape_date?: string | null;
  availability_status?: DealAvailabilityStatus | string;
  listing_duration_days?: number | null;
  n_snapshots?: number | null;
  is_under_option?: boolean;
  had_price_cut?: boolean;
  price_cut_eur?: number | null;
  price_cut_pct?: number | null;
  agency?: string | null;
  title?: string | null;
  description_snippet?: string | null;
  vision_delta_eur?: number | null;
  vision_delta_pct?: number | null;
  property_regime?: string | null;
  jepa_dz_quality?: number | null;
  jepa_energy_sym?: number | null;
  vis_pct_any_render?: number | null;
  has_renders?: boolean | null;
  has_condition_mislabel?: boolean | null;
  has_text_under_option?: boolean | null;
  caveat_codes?: string | null;
  source: "live" | "corpus_recent" | string;
}

export interface DataCaveat {
  code: string;
  severity: "high" | "medium" | "low" | string;
  label: string;
  message: string;
}

export interface LiveCheckResult {
  property_id: string;
  live_status: "online" | "offline" | "under_option" | "sold_or_rented" | "blocked" | "unknown" | "error";
  label: string;
  url: string;
}

export interface DealsQuery {
  neighbourhood?: string;
  property_type?: string;
  min_price?: number;
  max_price?: number;
  min_predicted?: number;
  max_predicted?: number;
  min_epc?: string;
  min_condition?: string;
  min_area?: number;
  min_bedrooms?: number;
  regime?: string;
  has_garden?: boolean;
  has_terrace?: boolean;
  price_cut_only?: boolean;
  min_jepa_dz?: number;
  search?: string;
  include_archive?: boolean;
  min_delta_pct?: number;
  below_band_only?: boolean;
  available_only?: boolean;
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

export interface DossierPricePoint {
  date: string | null;
  price_eur: number | null;
  price_old_eur: number | null;
  is_under_option: boolean;
  is_sold_or_rented: boolean;
  view_count: number | null;
  bookmark_count: number | null;
}

export interface PropertyDossier {
  property_id: string;
  url: string;
  address: string;
  street: string | null;
  house_number: string | null;
  postal_code: string;
  city: string;
  neighbourhood: string | null;
  neighbourhood_key: string | null;
  statbel_sector: string | null;
  latitude: number | null;
  longitude: number | null;
  dist_from_center_km: number | null;
  street_type: string | null;
  title: string | null;
  description: string | null;
  description_language: string | null;
  photos: string[];
  picture_count: number | null;
  floor_plan_count: number | null;
  has_virtual_tour: boolean | null;
  valuation: {
    asking_eur: number | null;
    predicted_mm_eur: number | null;
    predicted_tt_eur: number | null;
    low_80_eur: number | null;
    high_80_eur: number | null;
    interval_halfwidth_pct: number;
    band_position: "below_band" | "inside_band" | "above_band" | null;
    delta_eur: number | null;
    delta_pct: number | null;
    asking_psm: number | null;
    fair_psm: number | null;
    vision_delta_eur: number | null;
    vision_delta_pct: number | null;
    ens_std_mm: number | null;
  };
  jepa: {
    dz_quality: number | null;
    energy_sym: number | null;
    cherry_spread: number | null;
    pct_cgi?: number | null;
    pct_vstage?: number | null;
    pct_any_render?: number | null;
    detail?: {
      verdict: string;
      verdict_label: string;
      tone: "positive" | "negative" | "warning" | "neutral";
      summary: string;
      dz_quality: number;
      quality_percentile: number;
      energy_sym: number;
      energy_percentile: number;
      cherry_spread: number;
      cherry_percentile: number;
      pct_cgi?: number;
      pct_vstage?: number;
      pct_any_render?: number;
      caveats?: DataCaveat[];
      sub_scores: { axis: string; z: number }[];
    } | null;
  };
  caveats?: DataCaveat[];
  regime: {
    code: string;
    label: string;
    expected_mdape_pct?: number;
    is_non_standard: boolean;
    advisory: string | null;
  } | null;
  availability: {
    status: DealAvailabilityStatus | string;
    first_seen: string | null;
    last_seen: string | null;
    days_since_last_seen: number | null;
    latest_scrape_date: string | null;
    listing_duration_days: number | null;
    n_snapshots: number | null;
    is_under_option: boolean;
    is_sold_or_rented: boolean;
    had_price_cut: boolean;
    price_cut_eur: number | null;
    price_cut_pct: number | null;
    publication_date: string | null;
    last_modification_date: string | null;
    view_count: number | null;
    bookmark_count: number | null;
    agency: string | null;
    seller_type: string | null;
    price_history: DossierPricePoint[];
  };
  structure: {
    property_type: string | null;
    property_subtype: string | null;
    building_condition: string | null;
    construction_type: string | null;
    year_built: number | null;
    age: number | null;
    livable_area: number | null;
    property_size: number | null;
    living_room_surface: number | null;
    kitchen_surface: number | null;
    kitchen_type: string | null;
    facade_count: number | null;
    street_facade_width: number | null;
    floor_number: number | null;
    total_floors: number | null;
    bedrooms: number | null;
    bathrooms: number | null;
    toilets: number | null;
    shower_rooms: number | null;
    rooms: number | null;
    garden_surface: number | null;
    garden_orientation: string | null;
    terrace_surface: number | null;
    terrace_orientation: string | null;
    parking_indoor: number | null;
    parking_outdoor: number | null;
    parking_closed_box: number | null;
    amenities: string[];
  };
  energy_compliance: {
    energy_label: string | null;
    primary_energy_consumption: number | null;
    carbon_emission: number | null;
    epc_cert_year: number | null;
    heating_type: string | null;
    has_double_glazing: boolean | null;
    has_photovoltaic_panels: boolean | null;
    has_heat_pump: boolean | null;
    renovation_obligation: boolean | null;
    electrical_cert: string | null;
    asbestos_cert: string | null;
    oil_tank_status: string | null;
    p_score: string | null;
    g_score: string | null;
    flood_zone_type: string | null;
    land_use_designation: string | null;
    permit_obtained: boolean | null;
    urban_planning_breaches: boolean | null;
  };
  financial: {
    cadastral_income: number | null;
    monthly_costs: number | null;
    is_subject_to_vat: boolean | null;
    has_starting_price: boolean | null;
    is_public_sale: boolean | null;
    is_biddit_sale: boolean | null;
    is_newly_built: boolean | null;
    is_investment_property: boolean | null;
    expected_monthly_rental_income: number | null;
    availability_period_type: string | null;
  };
  environment: {
    mobiscore: number | null;
    noise_total_db: number | null;
    noise_road_db: number | null;
    noise_night_db: number | null;
    no2_ugm3: number | null;
    pm25_ugm3: number | null;
    urban_heat_c: number | null;
    buurtgroen_pct: number | null;
    trees_100m: number | null;
    dist_park_m: number | null;
    parks_600m: number | null;
    dist_transit_m: number | null;
    transit_stops_400m: number | null;
    dist_tram_m: number | null;
    dist_rail_station_m: number | null;
    dist_school_m: number | null;
    schools_500m: number | null;
    dist_water_m: number | null;
    dist_motorway_m: number | null;
    in_lez: boolean | null;
    in_autovrij: boolean | null;
    in_kotzone: boolean | null;
    parking_zone: string | null;
    circulatie_sector: string | null;
    cambio_500m: number | null;
    ev_chargers_400m: number | null;
    is_protected_monument: boolean | null;
    is_heritage_inventory: boolean | null;
    in_protected_townscape: boolean | null;
    flood_fluvial_class: string | null;
    flood_pluvial_class: string | null;
    grb_parcel_area: number | null;
    grb_footprint: number | null;
    grb_adjoining_bldgs: number | null;
    grb_footprint_to_parcel: number | null;
  };
  comparables?: {
    count: number;
    median_price_k: number;
    min_price_k: number;
    max_price_k: number;
    properties: {
      property_id?: string;
      price_k: number;
      price_eur?: number;
      livable_area: number;
      bedrooms: number;
      condition: string;
      energy_label: string;
      price_per_sqm: number;
    }[];
  } | null;
}

export interface ComparePropertiesResponse {
  properties: PropertyDossier[];
  count: number;
}
