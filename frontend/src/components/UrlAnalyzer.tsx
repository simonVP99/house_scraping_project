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
  const urlMutation = useUrlPrediction();
  const formMutation = useFormPrediction();

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      urlMutation.mutate(url.trim(), {
        onSuccess: (data) => setResult(data),
      });
    }
  };

  const handleRepredict = (formData: Record<string, unknown>) => {
    formMutation.mutate(formData, {
      onSuccess: (data) => setResult(data),
    });
  };

  const isPending = urlMutation.isPending || formMutation.isPending;
  const error = urlMutation.error || formMutation.error;

  return (
    <div className="space-y-8">
      {/* URL input */}
      <form onSubmit={handleUrlSubmit}>
        <label htmlFor="immoweb-url" className="block text-sm font-medium text-foreground mb-2">
          Immoweb listing URL
        </label>
        <div className="flex gap-2">
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
                Analyzing
              </span>
            ) : "Check Price"}
          </Button>
        </div>
        {error && (
          <p id="url-error" className="mt-2 text-sm text-negative" role="alert">
            {error.message}
          </p>
        )}
      </form>

      {/* Loading skeleton */}
      {isPending && !result && (
        <div className="space-y-6" aria-label="Loading prediction" role="status">
          <div className="space-y-3">
            <div className="h-3 w-24 bg-muted rounded animate-warm-pulse" />
            <div className="h-12 w-64 bg-muted rounded animate-warm-pulse" style={{ animationDelay: "75ms" }} />
            <div className="h-5 w-48 bg-muted rounded animate-warm-pulse" style={{ animationDelay: "150ms" }} />
          </div>
          <div className="border-t border-border pt-6 space-y-3">
            <div className="h-5 w-32 bg-muted rounded animate-warm-pulse" style={{ animationDelay: "225ms" }} />
            {[0, 1, 2, 3, 4].map((i) => (
              <div key={i} className="flex items-center gap-2 h-7">
                <div className="w-32 h-3 bg-muted rounded animate-warm-pulse" style={{ animationDelay: `${300 + i * 50}ms` }} />
                <div className="flex-1 h-5 bg-muted rounded animate-warm-pulse" style={{ animationDelay: `${325 + i * 50}ms` }} />
                <div className="w-20 h-3 bg-muted rounded animate-warm-pulse" style={{ animationDelay: `${350 + i * 50}ms` }} />
              </div>
            ))}
          </div>
          <span className="sr-only">Loading prediction results...</span>
        </div>
      )}

      {/* Empty state */}
      {!result && !isPending && (
        <div className="py-20 text-center animate-fade-in">
          <p className="font-display text-2xl text-foreground/30 font-light">
            Paste an Immoweb link to get started
          </p>
          <p className="text-sm text-muted-foreground mt-3 max-w-sm mx-auto leading-relaxed">
            We'll predict fair market value and show you
            how it compares to similar properties nearby.
          </p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-10">
          <PredictionResultCard data={result} />
          <hr className="border-border" />
          <FeatureForm
            formData={result.form_data}
            onRepredict={handleRepredict}
            isPending={formMutation.isPending}
          />
        </div>
      )}
    </div>
  );
}
