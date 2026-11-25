import { useState } from "react";
import UrlAnalyzer from "../components/UrlAnalyzer";
import ManualEntry from "../components/ManualEntry";
import { useModelMetadata } from "../hooks/useInsights";

export default function PredictPage() {
  const [mode, setMode] = useState<"url" | "manual">("manual");
  const { data: meta } = useModelMetadata();

  const mmMdape = meta?.tracks.multimodal.oof_normal.mdape;
  const ttMdape = meta?.tracks.tabular_text.oof_normal.mdape;
  const cov = meta?.tracks.multimodal.conformal_80.oof_cov_pct;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-primary">
            Phase-7 Lean-5 Valuation Engine
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-bold tracking-tight mt-1">
            Property Fair-Value Estimator
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Agency-blind Ghent valuation with heteroscedastic conformal intervals, TreeSHAP attribution, and live what-if simulation
          </p>
        </div>

        {/* Live Model Telemetry Pills */}
        {meta && (
          <div className="flex flex-wrap items-center gap-2 text-xs tabular-nums">
            <span className="px-2.5 py-1 rounded-full border border-border bg-card text-foreground font-medium">
              {mmMdape?.toFixed(2)}% MdAPE <span className="text-muted-foreground">(multimodal)</span>
            </span>
            <span className="px-2.5 py-1 rounded-full border border-border bg-card text-foreground font-medium">
              {ttMdape?.toFixed(2)}% <span className="text-muted-foreground">(tabular+text)</span>
            </span>
            <span className="px-2.5 py-1 rounded-full border border-positive/30 bg-positive/5 text-positive font-medium">
              {cov?.toFixed(1)}% conformal cov.
            </span>
          </div>
        )}
      </div>

      {/* Mode segmented switch */}
      <div className="inline-flex rounded-lg bg-muted p-1 border border-border/70" role="tablist" aria-label="Prediction method">
        <button
          role="tab"
          aria-selected={mode === "manual"}
          aria-controls="panel-manual"
          id="tab-manual"
          onClick={() => setMode("manual")}
          className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors cursor-pointer ${
            mode === "manual"
              ? "bg-card text-foreground shadow-xs"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Manual Entry & Examples
        </button>
        <button
          role="tab"
          aria-selected={mode === "url"}
          aria-controls="panel-url"
          id="tab-url"
          onClick={() => setMode("url")}
          className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors cursor-pointer ${
            mode === "url"
              ? "bg-card text-foreground shadow-xs"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          From Immoweb URL
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
