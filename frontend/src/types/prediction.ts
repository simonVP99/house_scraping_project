export interface Comparable {
  property_id?: string | null;
  url?: string | null;
  price_k: number;
  price_eur?: number | null;
  livable_area: number;
  bedrooms: number | null;
  condition: string | null;
  energy_label: string | null;
  age: number | null;
  price_per_sqm: number | null;
}

export interface Differentiator {
  feature: string;
  subject: string | number;
  comparable_avg: string | number;
  direction: string;
  impact: "positive" | "negative";
}

export interface AmenityContribution {
  feature: string;
  has_it: boolean;
  estimated_value_k: number;
  pct_in_neighbourhood: number;
  surface?: number;
  count?: number;
}

export interface ComparablesData {
  properties: Comparable[];
  median_price_k: number;
  min_price_k: number;
  max_price_k: number;
  median_price_per_sqm: number | null;
  subject_price_per_sqm: number | null;
  price_per_sqm_distribution: number[];
  differentiators: Differentiator[];
  amenity_contributions: AmenityContribution[];
  count: number;
  neighbourhood: string | null;
}

export interface FeatureComparison {
  feature: string;
  value: number;
  neighbourhood_median: number;
  percentile: number;
  direction: "above" | "below" | "equal";
}

export interface ShapFeature {
  feature: string;
  feature_key: string;
  value_k: number;
  value_eur: number;
}

export interface ShapData {
  base_value_k: number;
  base_value_eur: number;
  features: ShapFeature[];
}

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
    predicted_eur?: number;
    low_eur?: number;
    high_eur?: number;
    low_90_eur?: number;
    high_90_eur?: number;
    interval_halfwidth_pct?: number;
    confidence_level?: number;
    band_position?: "below_band" | "inside_band" | "above_band" | null;
    delta_eur: number | null;
    delta_pct: number | null;
    track?: "tabular_text" | "multimodal" | string;
    has_photos?: boolean;
    n_photos_used?: number;
    vision_source?: string | null;
    vision_shift_eur?: number | null;
    vision_shift_pct?: number | null;
    model_version?: string;
  };
  form_data: Record<string, unknown>;
  comparables?: ComparablesData | null;
  context?: FeatureComparison[];
  shap?: ShapData | null;
}

export interface ManualPredictionResult extends Partial<PredictionResult> {
  predicted_price: number;
}

export interface FeatureFieldMeta {
  key: string;
  label: string;
  type: "number" | "text" | "boolean" | "select";
  options?: string[];
  group: string;
}

export const FEATURE_FIELDS: FeatureFieldMeta[] = [
  // Location
  { key: "latitude", label: "Latitude", type: "number", group: "Location" },
  { key: "longitude", label: "Longitude", type: "number", group: "Location" },
  { key: "postal_code", label: "Postal Code", type: "text", group: "Location" },
  { key: "city", label: "City", type: "text", group: "Location" },
  { key: "street", label: "Street", type: "text", group: "Location" },
  { key: "house_number", label: "House Number", type: "text", group: "Location" },

  // Property
  { key: "property_type", label: "Property Type", type: "select", group: "Property", options: ["HOUSE", "APARTMENT"] },
  { key: "property_subtype", label: "Subtype", type: "select", group: "Property", options: ["HOUSE", "VILLA", "APARTMENT", "FLAT_STUDIO", "PENTHOUSE", "DUPLEX", "TRIPLEX", "GROUND_FLOOR", "MANOR_HOUSE", "TOWN_HOUSE", "BUNGALOW", "FARMHOUSE", "COUNTRY_COTTAGE", "EXCEPTIONAL_PROPERTY", "MIXED_USE_BUILDING", "APARTMENT_BLOCK", "LOFT", "SERVICE_FLAT", "KOT"] },
  { key: "livable_area", label: "Livable Area (m\u00B2)", type: "number", group: "Property" },
  { key: "property_size", label: "Plot Size (m\u00B2)", type: "number", group: "Property" },
  { key: "floor_number", label: "Floor Number", type: "number", group: "Property" },
  { key: "total_floors", label: "Total Floors", type: "number", group: "Property" },
  { key: "facade_count", label: "Facades", type: "number", group: "Property" },
  { key: "street_facade_width", label: "Street Facade Width (m)", type: "number", group: "Property" },

  // Rooms
  { key: "number_of_bedrooms", label: "Bedrooms", type: "number", group: "Rooms" },
  { key: "number_of_bathrooms", label: "Bathrooms", type: "number", group: "Rooms" },
  { key: "number_of_toilets", label: "Toilets", type: "number", group: "Rooms" },
  { key: "shower_room_count", label: "Shower Rooms", type: "number", group: "Rooms" },
  { key: "kitchen_surface", label: "Kitchen Area (m\u00B2)", type: "number", group: "Rooms" },
  { key: "living_room_surface", label: "Living Room Area (m\u00B2)", type: "number", group: "Rooms" },

  // Condition & Energy
  { key: "building_condition", label: "Condition", type: "select", group: "Condition & Energy", options: ["AS_NEW", "GOOD", "TO_BE_DONE_UP", "JUST_RENOVATED", "TO_RENOVATE", "TO_RESTORE"] },
  { key: "year_built", label: "Year Built", type: "number", group: "Condition & Energy" },
  { key: "kitchen_type", label: "Kitchen Type", type: "select", group: "Condition & Energy", options: ["INSTALLED", "HYPER_EQUIPPED", "SEMI_EQUIPPED", "NOT_INSTALLED", "USA_INSTALLED", "USA_HYPER_EQUIPPED", "USA_SEMI_EQUIPPED"] },
  { key: "energy_label", label: "Energy Label", type: "select", group: "Condition & Energy", options: ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"] },
  { key: "primary_energy_consumption", label: "Primary Energy (kWh/m\u00B2)", type: "number", group: "Condition & Energy" },
  { key: "heating_type", label: "Heating Type", type: "select", group: "Condition & Energy", options: ["GAS", "ELECTRIC", "FUELOIL", "PELLET", "WOOD", "CARBON", "SOLAR"] },
  { key: "renovation_obligation", label: "Renovation Obligation", type: "select", group: "Condition & Energy", options: ["YES", "NO", "Unknown"] },

  // Outdoor
  { key: "has_terrace", label: "Has Terrace", type: "boolean", group: "Outdoor" },
  { key: "terrace_surface", label: "Terrace Area (m\u00B2)", type: "number", group: "Outdoor" },
  { key: "has_garden", label: "Has Garden", type: "boolean", group: "Outdoor" },
  { key: "garden_surface", label: "Garden Area (m\u00B2)", type: "number", group: "Outdoor" },
  { key: "has_swimming_pool", label: "Has Pool", type: "boolean", group: "Outdoor" },

  // Amenities
  { key: "has_living_room", label: "Has Living Room", type: "boolean", group: "Amenities" },
  { key: "has_attic", label: "Has Attic", type: "boolean", group: "Amenities" },
  { key: "has_basement", label: "Has Basement", type: "boolean", group: "Amenities" },
  { key: "has_lift", label: "Has Lift", type: "boolean", group: "Amenities" },
  { key: "has_dressing_room", label: "Has Dressing Room", type: "boolean", group: "Amenities" },
  { key: "has_air_conditioning", label: "Has AC", type: "boolean", group: "Amenities" },
  { key: "has_caretaker", label: "Has Caretaker", type: "boolean", group: "Amenities" },
  { key: "parking_indoor", label: "Indoor Parking", type: "number", group: "Amenities" },
  { key: "parking_outdoor", label: "Outdoor Parking", type: "number", group: "Amenities" },
  { key: "is_furnished", label: "Furnished", type: "boolean", group: "Amenities" },
  { key: "monthly_costs", label: "Monthly Costs (\u20AC)", type: "number", group: "Amenities" },

  // Energy Systems
  { key: "has_photovoltaic_panels", label: "Solar Panels", type: "boolean", group: "Energy Systems" },
  { key: "has_double_glazing", label: "Double Glazing", type: "boolean", group: "Energy Systems" },
  { key: "has_heat_pump", label: "Heat Pump", type: "boolean", group: "Energy Systems" },
  { key: "has_thermic_panels", label: "Thermic Panels", type: "boolean", group: "Energy Systems" },

  // Other
  { key: "flood_zone_type", label: "Flood Zone", type: "select", group: "Other", options: ["NON_FLOOD_ZONE", "POSSIBLE_FLOOD_ZONE", "EFFECTIVE_FLOOD_ZONE"] },
  { key: "p_score", label: "P-Score", type: "number", group: "Other" },
  { key: "g_score", label: "G-Score", type: "number", group: "Other" },
  { key: "cadastral_income", label: "Cadastral Income (\u20AC)", type: "number", group: "Other" },
  { key: "price", label: "Asking Price (\u20AC)", type: "number", group: "Other" },
  { key: "is_under_option", label: "Under Option", type: "boolean", group: "Other" },
  { key: "is_newly_built", label: "Newly Built", type: "boolean", group: "Other" },
];
