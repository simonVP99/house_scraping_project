import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useFormPrediction } from "../hooks/usePrediction";
import { api } from "../services/api";
import PredictionResultCard from "./PredictionResult";
import FeatureForm from "./FeatureForm";
import type { PredictionResult } from "../types/prediction";

interface FormValues {
  address: string;
  property_type: string;
  livable_area: string;
  number_of_bedrooms: string;
  year_built: string;
  building_condition: string;
  energy_label: string;
  price: string;
  // Auto-filled from geocoding
  latitude: string;
  longitude: string;
  postal_code: string;
  city: string;
  street: string;
  house_number: string;
  neighbourhood: string;
}

const INITIAL: FormValues = {
  address: "",
  property_type: "",
  livable_area: "",
  number_of_bedrooms: "",
  year_built: "",
  building_condition: "",
  energy_label: "",
  price: "",
  latitude: "",
  longitude: "",
  postal_code: "",
  city: "",
  street: "",
  house_number: "",
  neighbourhood: "",
};

export default function ManualEntry() {
  const [form, setForm] = useState<FormValues>({ ...INITIAL });
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [geocodeStatus, setGeocodeStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [geocodeDisplay, setGeocodeDisplay] = useState<string | null>(null);
  const mutation = useFormPrediction();

  const set = (key: keyof FormValues, val: string) => {
    setForm((prev) => ({ ...prev, [key]: val }));
    setValidationError(null);
  };

  const handleGeocode = async () => {
    if (!form.address.trim()) {
      setValidationError("Enter an address to look up");
      return;
    }
    setGeocodeStatus("loading");
    setGeocodeDisplay(null);
    try {
      const geo = await api.geocode(form.address.trim());
      setForm((prev) => ({
        ...prev,
        latitude: String(geo.latitude),
        longitude: String(geo.longitude),
        postal_code: geo.postal_code || prev.postal_code,
        city: geo.city || prev.city,
        street: geo.street || prev.street,
        house_number: geo.house_number || prev.house_number,
        neighbourhood: geo.neighbourhood || "",
      }));
      setGeocodeDisplay(geo.display_name);
      setGeocodeStatus("done");
    } catch (e) {
      setGeocodeStatus("error");
      setValidationError(e instanceof Error ? e.message : "Geocoding failed");
    }
  };

  const validate = (): Record<string, unknown> | null => {
    if (!form.property_type) {
      setValidationError("Property type is required");
      return null;
    }
    const area = Number(form.livable_area);
    if (!area || area <= 0) {
      setValidationError("Livable area is required");
      return null;
    }
    const hasLatLon = form.latitude && form.longitude;
    const hasPostal = form.postal_code;
    if (!hasLatLon && !hasPostal) {
      setValidationError("Look up an address first, or enter a postal code");
      return null;
    }

    const data: Record<string, unknown> = {
      property_type: form.property_type,
      livable_area: area,
    };

    if (form.number_of_bedrooms) data.number_of_bedrooms = Number(form.number_of_bedrooms);
    if (form.year_built) data.year_built = Number(form.year_built);
    if (form.building_condition) data.building_condition = form.building_condition;
    if (form.energy_label) data.energy_label = form.energy_label;
    if (form.price) data.price = Number(form.price);
    if (form.latitude) data.latitude = Number(form.latitude);
    if (form.longitude) data.longitude = Number(form.longitude);
    if (form.postal_code) data.postal_code = form.postal_code;
    if (form.city) data.city = form.city;
    if (form.street) data.street = form.street;
    if (form.house_number) data.house_number = form.house_number;

    return data;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const data = validate();
    if (!data) return;
    mutation.mutate(data, {
      onSuccess: (res) => setResult(res),
    });
  };

  const handleRepredict = (formData: Record<string, unknown>) => {
    mutation.mutate(formData, {
      onSuccess: (res) => setResult(res),
    });
  };

  return (
    <div className="space-y-8">
      <form onSubmit={handleSubmit} className="space-y-6">
            {/* Address lookup */}
            <div>
              <h4 className="text-sm font-semibold mb-2 text-foreground">
                Address <span className="text-muted-foreground font-normal">*</span>
              </h4>
              <div className="flex gap-2">
                <Input
                  type="text"
                  value={form.address}
                  onChange={(e) => set("address", e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleGeocode();
                    }
                  }}
                  placeholder="e.g. Veldstraat 1, 9000 Gent"
                  className="flex-1"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleGeocode}
                  disabled={geocodeStatus === "loading"}
                >
                  {geocodeStatus === "loading" ? "Looking up..." : "Look up"}
                </Button>
              </div>
              {geocodeDisplay && (
                <p className="text-xs text-positive mt-1">{geocodeDisplay}</p>
              )}
              {geocodeStatus === "done" && form.neighbourhood && (
                <p className="text-xs text-muted-foreground mt-0.5">
                  Neighbourhood: <span className="font-medium">{form.neighbourhood}</span>
                  {" | "} {form.postal_code} {form.city}
                  {" | "} {form.latitude}, {form.longitude}
                </p>
              )}
              {geocodeStatus === "done" && !form.neighbourhood && (
                <p className="text-xs text-amber-600 mt-0.5">
                  Location found but outside known neighbourhoods.
                  Prediction will still work but neighbourhood context won't be available.
                </p>
              )}
            </div>

            {/* Required property info */}
            <div>
              <h4 className="text-sm font-semibold mb-2 text-foreground">
                Property <span className="text-muted-foreground font-normal">*</span>
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground">
                    Property Type *
                  </label>
                  <select
                    value={form.property_type}
                    onChange={(e) => set("property_type", e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <option value="">Select...</option>
                    <option value="HOUSE">House</option>
                    <option value="APARTMENT">Apartment</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">
                    Livable Area (m²) *
                  </label>
                  <Input
                    type="number"
                    value={form.livable_area}
                    onChange={(e) => set("livable_area", e.target.value)}
                    placeholder="e.g. 120"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">
                    Asking Price (€)
                  </label>
                  <Input
                    type="number"
                    value={form.price}
                    onChange={(e) => set("price", e.target.value)}
                    placeholder="e.g. 350000"
                  />
                </div>
              </div>
            </div>

            {/* Optional features */}
            <div>
              <h4 className="text-sm font-semibold mb-2 text-foreground">
                Optional <span className="text-muted-foreground font-normal">improves accuracy</span>
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground">Bedrooms</label>
                  <Input
                    type="number"
                    value={form.number_of_bedrooms}
                    onChange={(e) => set("number_of_bedrooms", e.target.value)}
                    placeholder="3"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">Year Built</label>
                  <Input
                    type="number"
                    value={form.year_built}
                    onChange={(e) => set("year_built", e.target.value)}
                    placeholder="1960"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">Condition</label>
                  <select
                    value={form.building_condition}
                    onChange={(e) => set("building_condition", e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <option value="">Unknown</option>
                    <option value="AS_NEW">As New</option>
                    <option value="GOOD">Good</option>
                    <option value="JUST_RENOVATED">Just Renovated</option>
                    <option value="TO_BE_DONE_UP">To Be Done Up</option>
                    <option value="TO_RENOVATE">To Renovate</option>
                    <option value="TO_RESTORE">To Restore</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">Energy Label</label>
                  <select
                    value={form.energy_label}
                    onChange={(e) => set("energy_label", e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <option value="">Unknown</option>
                    {["A++", "A+", "A", "B", "C", "D", "E", "F", "G"].map((l) => (
                      <option key={l} value={l}>{l}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {validationError && (
              <p className="text-sm text-negative">{validationError}</p>
            )}
            {mutation.isError && (
              <p className="text-sm text-negative">{mutation.error.message}</p>
            )}

            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? (
                <span className="flex items-center gap-2">
                  <span className="h-3.5 w-3.5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  Predicting
                </span>
              ) : "Predict Price"}
            </Button>
          </form>

      {result && (
        <>
          <PredictionResultCard data={result} />
          <FeatureForm
            formData={result.form_data}
            onRepredict={handleRepredict}
            isPending={mutation.isPending}
          />
        </>
      )}
    </div>
  );
}
