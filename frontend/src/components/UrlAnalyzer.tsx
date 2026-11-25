import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useUrlPrediction, useFormPrediction } from "../hooks/usePrediction";
import PredictionResultCard from "./PredictionResult";
import FeatureForm from "./FeatureForm";
import type { PredictionResult } from "../types/prediction";

export default function UrlAnalyzer() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [baselineResult, setBaselineResult] = useState<PredictionResult | null>(null);
  const urlMutation = useUrlPrediction();
  const formMutation = useFormPrediction();

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      urlMutation.mutate(url.trim(), {
        onSuccess: (data) => {
          setResult(data);
          setBaselineResult(data);
        },
      });
    }
  };

  const handleRepredict = (formData: Record<string, unknown>) => {
    formMutation.mutate(formData, {
      onSuccess: (data) => setResult(data),
    });
  };

  const handleResetSimulation = () => {
    if (baselineResult) setResult(baselineResult);
  };

  const isPending = urlMutation.isPending || formMutation.isPending;
  const error = urlMutation.error || formMutation.error;

  return (
    <div className="space-y-6">
      {/* URL input card */}
      <form onSubmit={handleUrlSubmit} className="rounded-xl border border-border bg-card p-6 space-y-3">
        <label htmlFor="immoweb-url" className="block text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Immoweb Listing URL
        </label>
        <div className="flex flex-col sm:flex-row gap-2">
          <Input
            id="immoweb-url"
            type="url"
            placeholder="https://www.immoweb.be/en/classified/house/for-sale/gent/9000/..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="flex-1"
            aria-describedby={error ? "url-error" : undefined}
          />
          <Button type="submit" disabled={isPending} className="shrink-0">
            {urlMutation.isPending ? (
              <span className="flex items-center gap-2">
                <span className="h-3.5 w-3.5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                Analyzing...
              </span>
            ) : "Analyze Listing"}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Parses full listing specs & description, estimates agency-blind fair value, and benchmarks against 12 local comparables.
        </p>
        {error && (
          <p id="url-error" className="text-sm text-negative" role="alert">
            {error.message}
          </p>
        )}
      </form>

      {/* Loading skeleton */}
      {isPending && !result && (
        <div className="rounded-xl border border-border bg-card p-6 space-y-5" aria-label="Loading prediction" role="status">
          <div className="space-y-2.5">
            <div className="h-3 w-36 bg-muted rounded animate-warm-pulse" />
            <div className="h-12 w-56 bg-muted rounded animate-warm-pulse" />
            <div className="h-5 w-72 bg-muted rounded animate-warm-pulse" />
          </div>
          <div className="h-20 w-full bg-muted/60 rounded-lg animate-warm-pulse" />
          <span className="sr-only">Loading prediction results...</span>
        </div>
      )}

      {/* Empty state */}
      {!result && !isPending && (
        <div className="rounded-xl border border-dashed border-border bg-card/40 py-16 px-6 text-center animate-fade-in">
          <p className="font-display text-2xl text-foreground/45 font-normal">
            Paste any Ghent Immoweb URL to inspect its fair value
          </p>
          <p className="text-sm text-muted-foreground mt-2.5 max-w-md mx-auto leading-relaxed">
            Get a calibrated 80% / 90% price range, TreeSHAP euro drivers, live renovation simulator,
            and local comparable sales — or switch to <strong>Manual Entry</strong> to try built-in Ghent examples.
          </p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <PredictionResultCard
            data={result}
            baseFormData={baselineResult?.form_data ?? result.form_data}
            basePrediction={baselineResult?.prediction.predicted_price ?? result.prediction.predicted_price}
            onSimulate={handleRepredict}
            onResetSimulation={handleResetSimulation}
            isSimulating={formMutation.isPending}
          />
          <div className="rounded-xl border border-border bg-card p-5">
            <FeatureForm
              formData={result.form_data}
              onRepredict={handleRepredict}
              isPending={formMutation.isPending}
            />
          </div>
        </div>
      )}
    </div>
  );
}
