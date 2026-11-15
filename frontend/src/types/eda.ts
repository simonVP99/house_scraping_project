export type FieldKind = "numeric" | "categorical" | "boolean" | "text" | "date" | "geo";

export interface NumericStats {
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  p01: number | null;
  p10: number | null;
  p25: number | null;
  median: number | null;
  p75: number | null;
  p90: number | null;
  p99: number | null;
  max: number | null;
  zeros: number;
  negatives: number;
  outliers_iqr3?: number;
}

export interface FieldValue {
  value: string | number | boolean | null;
  count: number;
  pct: number;
}

export interface FieldProfile {
  name: string;
  dtype: string;
  kind: FieldKind;
  non_null: number;
  null_count: number;
  null_pct: number;
  n_unique: number | null;
  is_leaky: boolean;
  is_id_like: boolean;
  flags: string[];
  stats?: NumericStats;
  values?: FieldValue[];
  range?: { min: string; max: string };
  length?: {
    count: number;
    mean: number | null;
    median: number | null;
    max: number | null;
    at_2500_cap: number;
    placeholder_dash: number;
  };
}

export interface Correlation {
  field: string;
  pearson: number | null;
  spearman: number | null;
  n: number;
  is_leaky: boolean;
}

export interface GroupStat {
  value: string | number | boolean | null;
  median: number | null;
  mean: number | null;
  n: number;
}

export interface SpatialCell {
  gx: number;
  gy: number;
  median_price: number;
  mean_price: number;
  n: number;
  median_price_sqm: number | null;
  lat: number;
  lon: number;
}

export interface PostalStat {
  postal_code: string | number;
  median_price: number;
  n: number;
  lat: number;
  lon: number;
}

export interface EdaReport {
  label: string;
  generated_at: string;
  dataset: {
    rows: number;
    columns: number;
    target: string;
    target_trim: [number, number];
    priced_rows: number;
  };
  fields: FieldProfile[];
  correlations: Correlation[];
  target_by_group: Record<string, GroupStat[]>;
  missingness: {
    total_fields: number;
    fields: { field: string; null_pct: number; null_count: number }[];
    bands: Record<string, number>;
    rows_with_no_nulls: number;
  };
  temporal: {
    min: string;
    max: string;
    distinct_days: number;
    daily: { date: string; count: number }[];
    monthly: { month: string; count: number }[];
  };
  spatial: {
    grid_size_deg: number;
    n_with_coords: number;
    n_priced_with_coords: number;
    bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
    cells: SpatialCell[];
    postal_codes: PostalStat[];
    neighbourhoods: PostalStat[];
  };
  duplicates: {
    unique_property_ids?: number;
    rows?: number;
    rows_per_property?: number;
    address_groups?: number;
    address_groups_with_multiple_ids?: number;
    ids_in_multi_id_groups?: number;
  };
}
