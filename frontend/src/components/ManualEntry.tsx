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

const QUICK_EXAMPLES: { label: string; values: FormValues }[] = [
  {
    label: "1930s Townhouse · Binnenstad (145 m², EPC C)",
    values: {
      address: "Kortrijksesteenweg 100, 9000 Gent",
      property_type: "HOUSE",
      livable_area: "145",
      number_of_bedrooms: "3",
      year_built: "1935",
      building_condition: "GOOD",
      energy_label: "C",
      price: "425000",
      latitude: "51.0365",
      longitude: "3.7138",
      postal_code: "9000",
      city: "Gent",
      street: "Kortrijksesteenweg",
      house_number: "100",
      neighbourhood: "Stationsbuurt_Noord",
    },
  },
  {
    label: "Fixer-upper Rowhouse · Dampoort (130 m², EPC F)",
    values: {
      address: "Antwerpsesteenweg 45, 9040 Gent",
      property_type: "HOUSE",
      livable_area: "130",
      number_of_bedrooms: "3",
      year_built: "1920",
      building_condition: "TO_RENOVATE",
      energy_label: "F",
      price: "285000",
      latitude: "51.0574",
      longitude: "3.7445",
      postal_code: "9040",
      city: "Gent",
      street: "Antwerpsesteenweg",
      house_number: "45",
      neighbourhood: "Dampoort",
    },
  },
  {
    label: "Modern 2-Bed Flat · Zuid (92 m², EPC B)",
    values: {
      address: "Koningin Maria Hendrikaplein 12, 9000 Gent",
      property_type: "APARTMENT",
      livable_area: "92",
      number_of_bedrooms: "2",
      year_built: "2014",
      building_condition: "AS_NEW",
      energy_label: "B",
      price: "365000",
      latitude: "51.0358",
      longitude: "3.7107",
      postal_code: "9000",
      city: "Gent",
      street: "Koningin Maria Hendrikaplein",
      house_number: "12",
      neighbourhood: "Stationsbuurt_Zuid",
    },
  },
];

export default function ManualEntry() {
  const [form, setForm] = useState<FormValues>({ ...INITIAL });
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [baselineResult, setBaselineResult] = useState<PredictionResult | null>(null);
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

  const buildPayload = (f: FormValues): Record<string, unknown> | null => {
    if (!f.property_type) {
      setValidationError("Property type is required");
      return null;
    }
    const area = Number(f.livable_area);
    if (!area || area <= 0) {
      setValidationError("Livable area is required");
      return null;
    }
    if (!f.latitude && !f.postal_code) {
      setValidationError("Look up an address first");
      return null;
    }
    const data: Record<string, unknown> = {
      property_type: f.property_type,
      livable_area: area,
    };
    if (f.number_of_bedrooms) data.number_of_bedrooms = Number(f.number_of_bedrooms);
    if (f.year_built) data.year_built = Number(f.year_built);
    if (f.building_condition) data.building_condition = f.building_condition;
    if (f.energy_label) data.energy_label = f.energy_label;
    if (f.price) data.price = Number(f.price);
    if (f.latitude) data.latitude = Number(f.latitude);
    if (f.longitude) data.longitude = Number(f.longitude);
    if (f.postal_code) data.postal_code = f.postal_code;
    if (f.city) data.city = f.city;
    if (f.street) data.street = f.street;
    if (f.house_number) data.house_number = f.house_number;
    if (f.neighbourhood) data.neighbourhood = f.neighbourhood;
    return data;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const data = buildPayload(form);
    if (!data) return;
    mutation.mutate(data, {
      onSuccess: (res) => {
        setResult(res);
        setBaselineResult(res);
      },
    });
  };

  const handleExample = (ex: FormValues) => {
    setForm({ ...ex });
    setGeocodeStatus("done");
    setGeocodeDisplay(`${ex.street} ${ex.house_number}, ${ex.postal_code} ${ex.city}`);
    setValidationError(null);
    const data = buildPayload(ex);
    if (data) {
      mutation.mutate(data, {
        onSuccess: (res) => {
          setResult(res);
          setBaselineResult(res);
        },
      });
    }
  };

  const handleRepredict = (formData: Record<string, unknown>) => {
    mutation.mutate(formData, {
      onSuccess: (res) => setResult(res),
    });
  };

  const handleResetSimulation = () => {
    if (baselineResult) setResult(baselineResult);
  };

  const selectClass =
    "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-ring";

  return (
    <div className="space-y-8">
      {/* Input card */}
      <form onSubmit={handleSubmit} className="rounded-xl border border-border bg-card p-6 space-y-5">
        {/* Quick try examples */}
        <div className="flex flex-wrap items-center gap-2 pb-3 border-b border-border/70">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mr-1">
            Try an example:
          </span>
          {QUICK_EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              onClick={() => handleExample(ex.values)}
              className="text-xs px-2.5 py-1 rounded-full border border-border bg-muted/40 hover:bg-muted text-foreground transition-colors cursor-pointer"
            >
              {ex.label}
            </button>
          ))}
        </div>

        {/* Address */}
        <div>
          <label htmlFor="manual-address" className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
            Address *
          </label>
          <div className="flex gap-2">
            <Input
              id="manual-address"
              placeholder="e.g. Veldstraat 10, 9000 Gent"
              value={form.address}
              onChange={(e) => set("address", e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleGeocode();
                }
              }}
              className="flex-1"
            />
            <Button type="button" variant="outline" onClick={handleGeocode} disabled={geocodeStatus === "loading"}>
              {geocodeStatus === "loading" ? "Looking up..." : "Look up"}
            </Button>
          </div>
          {geocodeStatus === "done" && geocodeDisplay && (
            <p className="mt-1.5 text-xs text-positive">
              {geocodeDisplay}
              {form.neighbourhood ? ` · ${form.neighbourhood.replace(/_/g, " ")}` : ""}
            </p>
          )}
        </div>

        {/* Core specs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label htmlFor="manual-type" className="block text-xs font-medium text-muted-foreground mb-1">
              Property Type *
            </label>
            <select
              id="manual-type"
              value={form.property_type}
              onChange={(e) => set("property_type", e.target.value)}
              className={selectClass}
            >
              <option value="">Select...</option>
              <option value="HOUSE">House</option>
              <option value="APARTMENT">Apartment</option>
            </select>
          </div>
          <div>
            <label htmlFor="manual-area" className="block text-xs font-medium text-muted-foreground mb-1">
              Livable Area (m²) *
            </label>
            <Input
              id="manual-area"
              type="number"
              min={10}
              max={1500}
              placeholder="145"
              value={form.livable_area}
              onChange={(e) => set("livable_area", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="manual-asking" className="block text-xs font-medium text-muted-foreground mb-1">
              Asking Price (€)
            </label>
            <Input
              id="manual-asking"
              type="number"
              min={25000}
              step={5000}
              placeholder="385000"
              value={form.price}
              onChange={(e) => set("price", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="manual-beds" className="block text-xs font-medium text-muted-foreground mb-1">
              Bedrooms
            </label>
            <Input
              id="manual-beds"
              type="number"
              min={0}
              max={20}
              placeholder="3"
              value={form.number_of_bedrooms}
              onChange={(e) => set("number_of_bedrooms", e.target.value)}
            />
          </div>
        </div>

        {/* Condition / EPC / Year */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label htmlFor="manual-year" className="block text-xs font-medium text-muted-foreground mb-1">
              Year Built
            </label>
            <Input
              id="manual-year"
              type="number"
              min={1700}
              max={2027}
              placeholder="1955"
              value={form.year_built}
              onChange={(e) => set("year_built", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="manual-cond" className="block text-xs font-medium text-muted-foreground mb-1">
              Condition
            </label>
            <select
              id="manual-cond"
              value={form.building_condition}
              onChange={(e) => set("building_condition", e.target.value)}
              className={selectClass}
            >
              <option value="">Unknown</option>
              <option value="AS_NEW">As new</option>
              <option value="JUST_RENOVATED">Just renovated</option>
              <option value="GOOD">Good</option>
              <option value="TO_BE_DONE_UP">To refresh</option>
              <option value="TO_RENOVATE">To renovate</option>
              <option value="TO_RESTORE">To restore</option>
            </select>
          </div>
          <div>
            <label htmlFor="manual-epc" className="block text-xs font-medium text-muted-foreground mb-1">
              Energy Label (EPC)
            </label>
            <select
              id="manual-epc"
              value={form.energy_label}
              onChange={(e) => set("energy_label", e.target.value)}
              className={selectClass}
            >
              <option value="">Unknown</option>
              {["A++", "A+", "A", "B", "C", "D", "E", "F", "G"].map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </div>
        </div>

        {validationError && <p className="text-sm text-negative" role="alert">{validationError}</p>}
        {mutation.error && <p className="text-sm text-negative" role="alert">{mutation.error.message}</p>}

        <div className="flex items-center justify-between pt-2">
          <p className="text-xs text-muted-foreground">
            Agency-blind Phase-7 Lean-5 valuation · calibrated 80% / 90% conformal range
          </p>
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? "Valuing..." : "Predict Fair Value"}
          </Button>
        </div>
      </form>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <PredictionResultCard
            data={result}
            baseFormData={baselineResult?.form_data ?? result.form_data}
            basePrediction={baselineResult?.prediction.predicted_price ?? result.prediction.predicted_price}
            onSimulate={handleRepredict}
            onResetSimulation={handleResetSimulation}
            isSimulating={mutation.isPending}
          />
          <div className="rounded-xl border border-border bg-card p-5">
            <FeatureForm
              formData={result.form_data}
              onRepredict={handleRepredict}
              isPending={mutation.isPending}
            />
          </div>
        </div>
      )}
    </div>
  );
}
