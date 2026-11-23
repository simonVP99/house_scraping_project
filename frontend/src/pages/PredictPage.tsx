import { useState } from "react";
import UrlAnalyzer from "../components/UrlAnalyzer";
import ManualEntry from "../components/ManualEntry";

export default function PredictPage() {
  const [mode, setMode] = useState<"url" | "manual">("url");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight">
          Price Predictor
        </h1>
        <p className="mt-1 text-muted-foreground">
          Analyze a property from Immoweb or enter details manually
        </p>
      </div>

      <div role="tablist" aria-label="Prediction method" className="flex gap-1 border-b border-border">
        <button
          role="tab"
          id="tab-url"
          aria-selected={mode === "url"}
          aria-controls="panel-url"
          onClick={() => setMode("url")}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
            mode === "url"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          From Immoweb URL
        </button>
        <button
          role="tab"
          id="tab-manual"
          aria-selected={mode === "manual"}
          aria-controls="panel-manual"
          onClick={() => setMode("manual")}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
            mode === "manual"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Manual Entry
        </button>
      </div>

      <div
        role="tabpanel"
        id={mode === "url" ? "panel-url" : "panel-manual"}
        aria-labelledby={mode === "url" ? "tab-url" : "tab-manual"}
      >
        {mode === "url" ? <UrlAnalyzer /> : <ManualEntry />}
      </div>
    </div>
  );
}
