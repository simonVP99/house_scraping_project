import { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ComposedChart,
  Line,
} from "recharts";
import { ExternalLink } from "lucide-react";
import { useMarketInsights } from "../hooks/useInsights";
import { useTableSort, sortGlyph } from "../hooks/useTableSort";
import EdaPage from "./EdaPage";
import type { AgencyAuditRow, EpcLabelRow, MarketInsights, PropertyRegimeRow } from "../types/insights";
import { formatCondition, formatEur, formatEurCompact, formatPsm, formatPctSigned } from "../lib/format";

type TopTab = "model" | "agencies" | "epc" | "forensics" | "jepa" | "dataset";

const TOP_TABS: { key: TopTab; label: string }[] = [
  { key: "model", label: "Phase-7 Model Lab" },
  { key: "agencies", label: "Agency Pricing Audit" },
  { key: "epc", label: "EPC & Renovation Cliff" },
  { key: "forensics", label: "Overpricing & Vision" },
  { key: "jepa", label: "JEPA Photo Honesty & Regimes" },
  { key: "dataset", label: "Full Dataset EDA (145k)" },
];

const C_PRIMARY = "#9a4a2a";
const C_POSITIVE = "#2f7d4d";
const C_NEGATIVE = "#b3402e";
const C_WARN = "#d08a2c";
const C_STONE = "#9c8d80";

const MEMBER_LABELS: Record<string, string> = {
  lgbmxt_price_hub: "LightGBM Extra-Trees (Huber)",
  spec_lgbm_price_hub: "Specialist LGBM (House/Apt)",
  cb_tuned_price_l2: "CatBoost Depth-8 (RMSE)",
  tabm_perm170_price_4s: "TabM Deep Tabular (Price, 4s)",
  tabm_perm170_psqm_4s: "TabM Deep Tabular (€/m², 4s)",
};

export default function InsightsPage() {
  const [tab, setTab] = useState<TopTab>("model");
  const { data: insights, isLoading, error } = useMarketInsights();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-primary">
            Ghent Housing Research & Model Telemetry
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-bold tracking-tight mt-1">
            Market Insights & Full EDA
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Agency-blind fair-value audits, Flemish EPC penalty curves, Phase-7 Lean-5 benchmarks, and full 145k-snapshot corpus profiling
          </p>
        </div>

        {insights && (
          <div className="flex flex-wrap gap-2 text-xs tabular-nums">
            <span className="px-2.5 py-1 rounded-full bg-card border border-border font-medium">
              {insights.corpus.total_properties.toLocaleString()} clean properties
            </span>
            <span className="px-2.5 py-1 rounded-full bg-card border border-border font-medium">
              {insights.corpus.raw_snapshots.toLocaleString()} raw snapshots
            </span>
            <span className="px-2.5 py-1 rounded-full bg-primary/10 border border-primary/25 text-primary font-semibold">
              {insights.generated_from}
            </span>
          </div>
        )}
      </div>

      {/* Primary pill navigation */}
      <div className="flex flex-wrap gap-1.5 border-b border-border pb-3" role="tablist">
        {TOP_TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={`px-3.5 py-1.5 text-sm font-medium rounded-lg transition-colors cursor-pointer ${
              tab === t.key
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/70"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "dataset" ? (
        <EdaPage embedded />
      ) : isLoading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-24 rounded-xl bg-muted animate-warm-pulse" />
            ))}
          </div>
          <div className="h-80 rounded-xl bg-muted animate-warm-pulse" />
        </div>
      ) : error || !insights ? (
        <div className="rounded-xl border border-negative/30 bg-negative/5 p-6 text-sm text-negative">
          Could not load market insights artifact. Run:{" "}
          <code className="font-mono text-xs">PYTHONPATH=src .venv-api/bin/python -u src/build_market_insights.py</code>
        </div>
      ) : (
        <>
          {tab === "model" && <ModelLabTab data={insights} />}
          {tab === "agencies" && <AgencyAuditTab data={insights} />}
          {tab === "epc" && <EpcCliffTab data={insights} />}
          {tab === "forensics" && <ForensicsTab data={insights} />}
          {tab === "jepa" && <JepaLabTab data={insights} />}
        </>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 1. Phase-7 Model Lab Tab                                                   */
/* -------------------------------------------------------------------------- */
function ModelLabTab({ data }: { data: MarketInsights }) {
  const mm = data.model_metadata.tracks.multimodal;
  const tt = data.model_metadata.tracks.tabular_text;

  const weightRows = data.model_metadata.members.map((m) => ({
    name: MEMBER_LABELS[m] ?? m,
    multimodal: Number((mm.huber_coefs[m] ?? 0).toFixed(3)),
    tabular_text: Number((tt.huber_coefs[m] ?? 0).toFixed(3)),
  }));

  return (
    <div className="space-y-6">
      {/* Dual-Track Headline Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <TrackCard
          title="Multimodal Track (Photos + Text + Tabular)"
          badge="7.27% OOF · 7.52% 2026 Holdout"
          features={`${mm.n_gbdt_features} GBDT · ${mm.n_nn_features} TabM features`}
          track={mm}
          accent="primary"
        />
        <TrackCard
          title="Tabular + Text Track (No Photos / Manual)"
          badge="7.70% OOF · 7.92% 2026 Holdout"
          features={`${tt.n_gbdt_features} GBDT · ${tt.n_nn_features} TabM features`}
          track={tt}
          accent="stone"
        />
      </div>

      {/* Model Design Contract strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <InfoPill label="Agency Blinding" value="Strictly Blind" sub="0 agency cols or target encodings" />
        <InfoPill label="Inflation Deflator" value="0.5·CPI + 0.5·HPI" sub={`Latest factor ${data.model_metadata.latest_deflator.toFixed(3)}×`} />
        <InfoPill label="Validation" value="Anti-Chaining CV" sub="Spatial+Text clusters + 2026 holdout" />
        <InfoPill label="Conformal 80%" value={`${mm.conformal_80.oof_cov_pct.toFixed(1)}% / ${mm.conformal_80.tmp_cov_pct.toFixed(1)}%`} sub={`±${mm.conformal_80.oof_median_hw_pct.toFixed(1)}% median half-width`} />
      </div>

      {/* Charts: Huber Stacker Weights + Accuracy by Price Tier */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-border bg-card p-5">
          <h3 className="font-display text-lg font-semibold">Lean-5 Ensemble Weights (Huber Stacker)</h3>
          <p className="text-xs text-muted-foreground mt-0.5 mb-4">
            Learned blending weights across 3 diverse GBDTs and 2 deep TabM neural nets
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={weightRows} layout="vertical" margin={{ left: 10, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: "#7d6f63" }} domain={[0, 0.35]} />
              <YAxis type="category" dataKey="name" width={175} tick={{ fontSize: 11, fill: "#3a2e28" }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }}
              />
              <Bar dataKey="multimodal" name="Multimodal" fill={C_PRIMARY} radius={[0, 4, 4, 0]} barSize={12} isAnimationActive={false} />
              <Bar dataKey="tabular_text" name="Tabular+Text" fill={C_STONE} radius={[0, 4, 4, 0]} barSize={12} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <h3 className="font-display text-lg font-semibold">Accuracy Across Price Tiers (Normal Homes)</h3>
          <p className="text-xs text-muted-foreground mt-0.5 mb-4">
            Out-of-fold MdAPE (%) and Hit@10% share by asking price bracket
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={data.error_by_band} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" vertical={false} />
              <XAxis dataKey="band" tick={{ fontSize: 10, fill: "#7d6f63" }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11, fill: "#7d6f63" }} unit="%" domain={[0, 14]} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: "#7d6f63" }} unit="%" domain={[40, 80]} />
              <Tooltip
                contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }}
              />
              <Bar yAxisId="left" dataKey="mdape" name="MdAPE (%)" fill={C_PRIMARY} radius={[4, 4, 0, 0]} barSize={28} isAnimationActive={false} />
              <Line yAxisId="right" type="monotone" dataKey="hit_10" name="Hit@10% (%)" stroke={C_POSITIVE} strokeWidth={2.5} dot={{ r: 3.5 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function TrackCard({
  title,
  badge,
  features,
  track,
  accent,
}: {
  title: string;
  badge: string;
  features: string;
  track: MarketInsights["model_metadata"]["tracks"]["multimodal"];
  accent: "primary" | "stone";
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="font-display text-lg font-semibold">{title}</h3>
          <p className="text-xs text-muted-foreground mt-0.5">{features}</p>
        </div>
        <span
          className={`text-xs font-semibold px-2.5 py-1 rounded-full tabular-nums ${
            accent === "primary"
              ? "bg-primary/10 text-primary border border-primary/25"
              : "bg-muted text-foreground border border-border"
          }`}
        >
          {badge}
        </span>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 text-center tabular-nums">
        <MetricBox label="OOF MdAPE" value={`${track.oof_normal.mdape.toFixed(2)}%`} />
        <MetricBox label="2026 Holdout" value={`${track.temporal_2026_normal.mdape.toFixed(2)}%`} />
        <MetricBox label="Hit@10%" value={`${track.oof_normal.hit_10.toFixed(1)}%`} />
        <MetricBox label="OOF MAE" value={formatEurCompact(track.oof_normal.mae)} />
        <MetricBox label="R²" value={track.oof_normal.r2.toFixed(3)} />
      </div>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-muted/45 border border-border/60 py-2 px-1.5">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="text-sm font-bold text-foreground mt-0.5">{value}</p>
    </div>
  );
}

function InfoPill({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{label}</p>
      <p className="font-display text-lg font-bold mt-0.5 text-foreground tabular-nums">{value}</p>
      <p className="text-[11px] text-muted-foreground mt-0.5">{sub}</p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 2. Agency Pricing Audit Tab                                                */
/* -------------------------------------------------------------------------- */
function AgencyAuditTab({ data }: { data: MarketInsights }) {
  const [search, setSearch] = useState("");
  const agencies = data.agency_audit;

  const topOver = agencies.slice(0, 10);
  const topUnder = [...agencies].slice(-10).reverse();
  const chartData = [...topOver.slice(0, 8), ...topUnder.slice(0, 8).reverse()];

  const filtered = useMemo(
    () =>
      search.trim()
        ? agencies.filter((a) => a.agency.toLowerCase().includes(search.trim().toLowerCase()))
        : agencies,
    [agencies, search],
  );

  type AgencySortKey =
    | "agency"
    | "listings"
    | "median_asking_eur"
    | "median_fair_eur"
    | "median_premium_pct"
    | "pct_overpriced_10"
    | "pct_price_cut"
    | "median_jepa_dz"
    | "median_duration_days";

  const { sorted: sortedAgencies, sort: agSort, toggle: toggleAgSort } = useTableSort<AgencyAuditRow, AgencySortKey>(
    filtered,
    "median_premium_pct",
    "desc",
  );

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-primary/20 bg-primary/5 px-5 py-4 text-sm text-foreground">
        <strong>Why this audit is unbiased:</strong> Our Phase-7 Lean-5 model is <strong>strictly agency-blind</strong> —
        agency names and target encodings were stripped from all 279 features. The median premium below measures each
        broker&apos;s systematic asking price relative to independent fair value across{" "}
        <strong>{agencies.length} active Ghent agencies</strong> (≥40 listings each).
      </div>

      {/* Diverging bar chart: highest vs sharpest agencies */}
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="font-display text-lg font-semibold">Asking Price Premium vs Agency-Blind Fair Value</h3>
        <p className="text-xs text-muted-foreground mt-0.5 mb-4">
          8 highest-premium vs 8 sharpest-pricing Ghent agencies (median (asking − fair) / fair)
        </p>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" horizontal={false} />
            <XAxis type="number" unit="%" tick={{ fontSize: 11, fill: "#7d6f63" }} domain={[-6, 8.5]} />
            <YAxis type="category" dataKey="agency" width={190} tick={{ fontSize: 11, fill: "#2e241f" }} />
            <ReferenceLine x={0} stroke="#5a4b41" strokeWidth={1.2} />
            <Tooltip
              contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }}
              formatter={(v) => formatPctSigned(Number(v), 2)}
            />
            <Bar dataKey="median_premium_pct" name="Median Asking Premium" radius={[4, 4, 4, 4]} barSize={16} isAnimationActive={false}>
              {chartData.map((entry) => (
                <Cell key={entry.agency} fill={entry.median_premium_pct >= 0 ? C_NEGATIVE : C_POSITIVE} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Full searchable agency leaderboard */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="px-5 py-4 border-b border-border flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-display text-lg font-semibold">Full Ghent Agency Scorecard ({filtered.length})</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Sorted from highest asking premium to sharpest pricing
            </p>
          </div>
          <input
            type="text"
            placeholder="Search agency name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm max-w-xs w-full focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-muted/95 backdrop-blur border-b border-border text-xs text-muted-foreground">
              <tr>
                {(
                  [
                    ["agency", "Agency", "left", "asc", undefined],
                    ["listings", "Listings", "right", "desc", undefined],
                    ["median_asking_eur", "Median Ask", "right", "desc", undefined],
                    ["median_fair_eur", "Fair Value", "right", "desc", undefined],
                    ["median_premium_pct", "Median Premium", "right", "desc", undefined],
                    ["pct_overpriced_10", ">10% Over", "right", "desc", undefined],
                    ["pct_price_cut", "Price Cuts", "right", "desc", undefined],
                    [
                      "median_jepa_dz",
                      "Photo Δz",
                      "right",
                      "desc",
                      "Cross-Modal JEPA median Δz_quality: positive = photos exceed specs, negative = photos look rougher than claimed specs",
                    ],
                    ["median_duration_days", "Med. Stay", "right", "desc", undefined],
                  ] as const
                ).map(([key, label, align, defDir, title], idx, arr) => (
                  <th
                    key={key}
                    title={title}
                    onClick={() => toggleAgSort(key, defDir)}
                    className={`${align === "left" ? "text-left" : "text-right"} ${
                      idx === 0 ? "px-4" : idx === arr.length - 1 ? "px-4" : "px-3"
                    } py-2.5 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap`}
                  >
                    {label}{" "}
                    <span className={`ml-0.5 text-[10px] ${agSort.key === key ? "text-primary" : "opacity-35"}`}>
                      {sortGlyph(agSort.key === key, agSort.dir)}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 tabular-nums">
              {sortedAgencies.map((a) => (
                <tr key={a.agency} className="hover:bg-muted/40">
                  <td className="px-4 py-2 font-medium text-foreground">{a.agency}</td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{a.listings}</td>
                  <td className="px-3 py-2 text-right">{formatEurCompact(a.median_asking_eur)}</td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{formatEurCompact(a.median_fair_eur)}</td>
                  <td className={`px-3 py-2 text-right font-semibold ${a.median_premium_pct > 1 ? "text-negative" : a.median_premium_pct < -1 ? "text-positive" : "text-foreground"}`}>
                    {formatPctSigned(a.median_premium_pct, 2)}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{a.pct_overpriced_10.toFixed(1)}%</td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{a.pct_price_cut.toFixed(1)}%</td>
                  <td
                    className={`px-3 py-2 text-right font-medium ${
                      (a.median_jepa_dz ?? 0) >= 0.2
                        ? "text-positive"
                        : (a.median_jepa_dz ?? 0) <= -0.25
                          ? "text-negative"
                          : "text-muted-foreground"
                    }`}
                  >
                    {a.median_jepa_dz != null ? `${a.median_jepa_dz > 0 ? "+" : ""}${a.median_jepa_dz.toFixed(2)}` : "—"}
                  </td>
                  <td className="px-4 py-2 text-right text-muted-foreground">{a.median_duration_days}d</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 3. EPC Renovation Cliff & Condition Ladder Tab                             */
/* -------------------------------------------------------------------------- */
const EPC_ORDER: Record<string, number> = { "A++": 0, "A+": 1, A: 2, B: 3, C: 4, D: 5, E: 6, F: 7 };

const EPC_ACCESSORS = {
  label: (r: EpcLabelRow) => EPC_ORDER[r.label] ?? 99,
};

function EpcCliffTab({ data }: { data: MarketInsights }) {
  const [ptype, setPtype] = useState<string>("House");
  const active = data.epc_cliff.find((e) => e.property_type === ptype) ?? data.epc_cliff[0];
  type EpcSortKey = "label" | "count" | "median_price_eur" | "median_price_sqm" | "vs_c_psqm_pct" | "median_area";
  const { sorted: sortedEpc, sort: epcSort, toggle: toggleEpcSort } = useTableSort<EpcLabelRow, EpcSortKey>(
    active?.labels ?? [],
    "label",
    "asc",
    EPC_ACCESSORS,
  );

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-border bg-card p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="font-display text-lg font-semibold">The Flemish EPC Cliff (€/m² by Energy Label)</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Red bars mark EPC E &amp; F — subject to Flanders&apos; mandatory 5-year renovation obligation upon purchase
            </p>
          </div>
          <div className="inline-flex rounded-lg bg-muted p-0.5 border border-border/60">
            {["House", "Apartment", "All"].map((pt) => (
              <button
                key={pt}
                type="button"
                onClick={() => setPtype(pt)}
                className={`px-3 py-1 text-xs font-medium rounded-md cursor-pointer ${
                  ptype === pt ? "bg-card text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {pt}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={active?.labels ?? []} margin={{ top: 10, right: 16, left: 4, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 12, fill: "#2e241f", fontWeight: 600 }} />
            <YAxis tick={{ fontSize: 11, fill: "#7d6f63" }} tickFormatter={(v) => `€${v}`} />
            <Tooltip
              contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }}
              formatter={(value, name) =>
                name === "median_price_sqm" ? [formatPsm(Number(value)), "Median €/m²"] : [String(value), String(name)]
              }
            />
            <Bar dataKey="median_price_sqm" name="median_price_sqm" radius={[6, 6, 0, 0]} barSize={44} isAnimationActive={false}>
              {(active?.labels ?? []).map((entry) => (
                <Cell
                  key={entry.label}
                  fill={entry.mandatory_renovation ? C_NEGATIVE : entry.label.startsWith("A") ? C_POSITIVE : C_PRIMARY}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* EPC table */}
        <div className="overflow-x-auto border border-border rounded-lg">
          <table className="w-full text-xs tabular-nums">
            <thead className="bg-muted/60 text-muted-foreground border-b border-border">
              <tr>
                {(
                  [
                    ["label", "EPC", "left", "asc"],
                    ["count", "Listings", "right", "desc"],
                    ["median_price_eur", "Median Price", "right", "desc"],
                    ["median_price_sqm", "Median €/m²", "right", "desc"],
                    ["vs_c_psqm_pct", "vs EPC C", "right", "desc"],
                    ["median_area", "Med. Area", "right", "desc"],
                  ] as const
                ).map(([key, label, align, defDir]) => (
                  <th
                    key={key}
                    onClick={() => toggleEpcSort(key, defDir)}
                    className={`${align === "left" ? "text-left" : "text-right"} px-3 py-2 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap`}
                  >
                    {label}{" "}
                    <span className={`ml-0.5 text-[10px] ${epcSort.key === key ? "text-primary" : "opacity-35"}`}>
                      {sortGlyph(epcSort.key === key, epcSort.dir)}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {sortedEpc.map((row) => (
                <tr key={row.label} className={row.mandatory_renovation ? "bg-negative/5" : ""}>
                  <td className="px-3 py-2 font-bold">
                    {row.label}
                    {row.mandatory_renovation && (
                      <span className="ml-2 text-[10px] font-medium text-negative">Mandatory reno</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{row.count.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right font-medium">{formatEur(row.median_price_eur)}</td>
                  <td className="px-3 py-2 text-right font-semibold">{formatPsm(row.median_price_sqm)}</td>
                  <td className={`px-3 py-2 text-right font-semibold ${row.vs_c_psqm_pct > 0 ? "text-positive" : row.vs_c_psqm_pct < 0 ? "text-negative" : "text-muted-foreground"}`}>
                    {formatPctSigned(row.vs_c_psqm_pct)}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{row.median_area} m²</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Condition Ladder */}
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="font-display text-lg font-semibold">Building Condition Value Ladder</h3>
        <p className="text-xs text-muted-foreground mt-0.5 mb-4">
          Median price per m² across Immoweb condition tiers (Normal Ghent homes)
        </p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.condition_ladder} margin={{ top: 5, right: 16, left: 4, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#2e241f" }} />
            <YAxis tick={{ fontSize: 11, fill: "#7d6f63" }} tickFormatter={(v) => `€${v}`} />
            <Tooltip
              contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }}
              formatter={(value) => [formatPsm(Number(value)), "Median €/m²"]}
            />
            <Bar dataKey="median_price_sqm" fill={C_PRIMARY} radius={[6, 6, 0, 0]} barSize={46} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 4. Overpricing Forensics & Multimodal Vision Impact Tab                    */
/* -------------------------------------------------------------------------- */
function ForensicsTab({ data }: { data: MarketInsights }) {
  const vis = data.vision_impact;

  return (
    <div className="space-y-6">
      {/* Overpricing -> Price Cuts vs Under Option */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-4">
        <div>
          <h3 className="font-display text-lg font-semibold">Overpricing Forensics: What Happens When Sellers Price Above Fair Value?</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Real-world validation of Phase-7 fair value: homes priced &gt;20% above fair value suffer nearly 3× the price-cut rate and 4× fewer &ldquo;under option&rdquo; sales than underpriced listings.
          </p>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={data.overpricing_forensics} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" vertical={false} />
            <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: "#2e241f" }} />
            <YAxis yAxisId="left" tick={{ fontSize: 11, fill: "#7d6f63" }} unit="%" />
            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: "#7d6f63" }} unit="%" />
            <Tooltip
              contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }}
            />
            <Bar yAxisId="left" dataKey="pct_under_option" name="Sold / Under Option (%)" fill={C_POSITIVE} radius={[4, 4, 0, 0]} barSize={32} isAnimationActive={false} />
            <Line yAxisId="right" type="monotone" dataKey="pct_price_cut" name="Price Cut Rate (%)" stroke={C_NEGATIVE} strokeWidth={2.5} dot={{ r: 4 }} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Vision Impact (Multimodal vs Tabular+Text) */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h3 className="font-display text-lg font-semibold">What Listing Photos Reveal (SigLIP2 + SO400M Vision Signal)</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Distribution of valuation shift when adding photo embeddings to the tabular + description model
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs tabular-nums">
            <span className="px-2.5 py-1 rounded-full bg-positive/10 text-positive border border-positive/25 font-semibold">
              −{vis.mdape_gain_pts.toFixed(2)} pts MdAPE gain
            </span>
            <span className="px-2.5 py-1 rounded-full bg-muted text-foreground font-medium">
              Median |shift| {vis.median_abs_shift_pct.toFixed(1)}%
            </span>
            <span className="px-2.5 py-1 rounded-full bg-muted text-foreground font-medium">
              {vis.pct_shifted_over_5pct.toFixed(1)}% shift ≥5%
            </span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={vis.histogram} margin={{ top: 5, right: 12, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7dfd7" vertical={false} />
            <XAxis dataKey="mid" tick={{ fontSize: 10, fill: "#7d6f63" }} tickFormatter={(v) => `${v > 0 ? "+" : ""}${Math.round(v)}%`} />
            <YAxis tick={{ fontSize: 11, fill: "#7d6f63" }} />
            <ReferenceLine x={0} stroke="#5a4b41" strokeDasharray="3 3" />
            <Tooltip
              contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }}
              labelFormatter={(v) => `Photo valuation shift: ${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(1)}%`}
              formatter={(c) => [`${Number(c).toLocaleString()} homes`, "Count"]}
            />
            <Bar dataKey="count" radius={[3, 3, 0, 0]} isAnimationActive={false}>
              {vis.histogram.map((b, i) => (
                <Cell key={i} fill={b.mid >= 0 ? C_PRIMARY : C_WARN} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 5. Cross-Modal JEPA Photo Honesty & Mixed-Use/Investment Regimes Tab       */
/* -------------------------------------------------------------------------- */
function JepaLabTab({ data }: { data: MarketInsights }) {
  const jepa = data.jepa_forensics;
  type RegimeSortKey = "label" | "count" | "median_asking_eur" | "mdape" | "p80_err" | "avg_duration_days" | "median_jepa_energy";
  const { sorted: sortedRegimes, sort: regSort, toggle: toggleRegSort } = useTableSort<PropertyRegimeRow, RegimeSortKey>(
    jepa?.regime_breakdown ?? [],
    "count",
    "desc",
  );

  if (!jepa) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center text-sm text-muted-foreground">
        Cross-Modal JEPA forensics not built yet. Run{" "}
        <code className="text-foreground">PYTHONPATH=src .venv-api/bin/python -u src/research_cross_modal_jepa.py</code>{" "}
        followed by <code className="text-foreground">src/build_market_insights.py</code>.
      </div>
    );
  }

  const fair = jepa.fair_priced_controlled;
  const hon = jepa.condition_honesty;

  return (
    <div className="space-y-6">
      {/* Intro Banner */}
      <div className="rounded-xl border border-primary/25 bg-primary/5 p-5 space-y-2">
        <p className="text-[11px] font-semibold uppercase tracking-widest text-primary">
          Cross-Modal JEPA Latent-Residual Surprise · n = {jepa.n_scored.toLocaleString()} Ghent Homes
        </p>
        <h3 className="font-display text-xl font-semibold text-foreground">
          Predicting what a listing&apos;s photos <em>should</em> look like from its specs — then measuring the surprise
        </h3>
        <p className="text-xs text-muted-foreground leading-relaxed max-w-4xl">
          Instead of scoring photos in isolation, a strictly fold-safe predictor{" "}
          <code className="text-foreground font-mono">f(Tabular + Spatial + Text) → ẑ_photo</code> estimates the expected
          SigLIP2 + SO400M embedding of a Ghent home with those exact specs, while the reverse predictor{" "}
          <code className="text-foreground font-mono">g(Specs + Photo) → ẑ_text</code> checks description consistency.
          The residual <code className="text-foreground font-mono">Δz = z_actual − ẑ_expected</code> isolates pure visual
          honesty and photo-vs-specs contradiction.
        </p>
      </div>

      {/* Headline Controlled Findings */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl border border-border bg-card p-5 space-y-1.5">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Fairly Priced (±5%) · High vs Low Energy
          </p>
          <p className="font-display text-2xl font-bold tabular-nums text-foreground">
            {fair.q1_duration_days}d → <span className="text-negative">{fair.q5_duration_days}d</span>
          </p>
          <p className="text-xs text-muted-foreground">
            Even when asking price is fair (n={fair.count.toLocaleString()}), high photo-vs-specs contradiction
            sits <strong>+{(fair.q5_duration_days - fair.q1_duration_days).toFixed(1)} days longer</strong> and has{" "}
            <strong>{(fair.q5_price_cut_pct / Math.max(fair.q1_price_cut_pct, 0.1)).toFixed(1)}×</strong> the price-cut rate
            ({fair.q1_price_cut_pct}% → {fair.q5_price_cut_pct}%).
          </p>
        </div>

        <div className="rounded-xl border border-border bg-card p-5 space-y-1.5">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Condition Honesty (Claimed As New / Renovated)
          </p>
          <p className="font-display text-2xl font-bold tabular-nums text-foreground">
            {hon.top15_looks_sharp.pct_price_cut}% vs{" "}
            <span className="text-negative">{hon.bottom15_looks_dated.pct_price_cut}% cuts</span>
          </p>
          <p className="text-xs text-muted-foreground">
            Among {hon.total_claimed_top.toLocaleString()} top-condition listings, the bottom 15% by{" "}
            <code className="text-foreground">Δz_quality</code> get{" "}
            <strong>{hon.bottom15_looks_dated.median_vision_shift_pct}%</strong> visual penalty (vs{" "}
            <strong>+{hon.top15_looks_sharp.median_vision_shift_pct}%</strong>) and{" "}
            <strong>2×</strong> as many price cuts.
          </p>
        </div>

        <div className="rounded-xl border border-border bg-card p-5 space-y-1.5">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Intra-Listing Room Cherry-Picking
          </p>
          <p className="font-display text-2xl font-bold tabular-nums text-foreground">
            {jepa.cherry_quintiles[0]?.mdape}% →{" "}
            <span className="text-warning">{jepa.cherry_quintiles[jepa.cherry_quintiles.length - 1]?.mdape}%</span> MdAPE
          </p>
          <p className="text-xs text-muted-foreground">
            Listings with the widest gap between best photo finish and worst room shown carry{" "}
            <strong>+{(
              (jepa.cherry_quintiles[jepa.cherry_quintiles.length - 1]?.mdape ?? 0) -
              (jepa.cherry_quintiles[0]?.mdape ?? 0)
            ).toFixed(2)} pt</strong> higher valuation uncertainty.
          </p>
        </div>
      </div>

      {/* Charts row: Energy Quintiles (Duration/Cuts & Error) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-border bg-card p-5 space-y-3">
          <div>
            <h3 className="font-display text-lg font-semibold">Contradiction Energy vs Market Friction</h3>
            <p className="text-xs text-muted-foreground">
              Bars: average days on market · Line: price-cut rate (%) across Symmetric JEPA Energy quintiles
            </p>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={jepa.energy_quintiles} margin={{ top: 10, right: 16, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4ddd5" vertical={false} />
              <XAxis dataKey="quintile" tick={{ fontSize: 11, fill: "#5f554c" }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11, fill: "#7a6f66" }} unit="d" />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: "#7a6f66" }} unit="%" />
              <Tooltip contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }} />
              <Bar yAxisId="left" dataKey="avg_duration_days" name="Avg stay (days)" fill={C_PRIMARY} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              <Line yAxisId="right" type="monotone" dataKey="pct_price_cut" name="Price cut rate (%)" stroke={C_NEGATIVE} strokeWidth={2.5} dot={{ r: 4 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-xl border border-border bg-card p-5 space-y-3">
          <div>
            <h3 className="font-display text-lg font-semibold">Contradiction Energy vs Valuation Uncertainty</h3>
            <p className="text-xs text-muted-foreground">
              P80 absolute error (%) and MdAPE (%) rise monotonically when photos diverge from specs
            </p>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={jepa.energy_quintiles} margin={{ top: 10, right: 16, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4ddd5" vertical={false} />
              <XAxis dataKey="quintile" tick={{ fontSize: 11, fill: "#5f554c" }} />
              <YAxis tick={{ fontSize: 11, fill: "#7a6f66" }} unit="%" />
              <Tooltip contentStyle={{ backgroundColor: "#faf8f5", border: "1px solid #d6cdc4", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="p80_err" name="P80 Abs Error (%)" fill={C_WARN} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              <Line type="monotone" dataKey="mdape" name="MdAPE (%)" stroke={C_PRIMARY} strokeWidth={2.5} dot={{ r: 4 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Concrete Real Ghent Case Studies: Over-Claimed vs Hidden Gems */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-negative/25 bg-card overflow-hidden">
          <div className="p-4 border-b border-border bg-negative/5">
            <h3 className="font-display text-base font-semibold text-negative">
              Flagged: Claimed &ldquo;As New / Renovated&rdquo; · Photos Look Dated
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Lowest <code className="text-foreground">Δz_quality</code> among top-condition listings — inspect finishes closely
            </p>
          </div>
          <div className="divide-y divide-border/60">
            {jepa.examples.overclaimed.map((ex) => (
              <div key={ex.property_id} className="p-3.5 flex items-center justify-between gap-3 text-xs">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="font-semibold text-foreground">{ex.neighbourhood}</span>
                    <span className="px-1.5 py-0.5 rounded bg-negative/10 text-negative font-semibold tabular-nums">
                      Δz {ex.dz_quality.toFixed(2)}
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground tabular-nums">
                      vis {formatPctSigned(ex.vision_shift_pct)}
                    </span>
                    {ex.had_price_cut && (
                      <span className="px-1.5 py-0.5 rounded bg-warning/15 text-warning font-semibold">Cut</span>
                    )}
                  </div>
                  <p className="text-muted-foreground mt-0.5 truncate">
                    {ex.property_type} · {ex.livable_area} m² · EPC {ex.energy_label ?? "?"} · Claimed {formatCondition(ex.condition)}
                    {ex.title ? ` — ${ex.title}` : ""}
                  </p>
                </div>
                <div className="text-right shrink-0 tabular-nums">
                  <p className="font-semibold text-foreground">{formatEurCompact(ex.asking_eur)}</p>
                  <a href={ex.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline inline-flex items-center gap-0.5">
                    #{ex.property_id} <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-positive/25 bg-card overflow-hidden">
          <div className="p-4 border-b border-border bg-positive/5">
            <h3 className="font-display text-base font-semibold text-positive">
              Flagged: Visual Hidden Gems (Photos Outperform Specs)
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Listed as Good / To Refresh / To Renovate, priced below fair value, yet highest <code className="text-foreground">Δz_quality</code>
            </p>
          </div>
          <div className="divide-y divide-border/60">
            {jepa.examples.hidden_gems.map((ex) => (
              <div key={ex.property_id} className="p-3.5 flex items-center justify-between gap-3 text-xs">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="font-semibold text-foreground">{ex.neighbourhood}</span>
                    <span className="px-1.5 py-0.5 rounded bg-positive/10 text-positive font-semibold tabular-nums">
                      Δz +{ex.dz_quality.toFixed(2)}
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-positive/10 text-positive tabular-nums">
                      vis {formatPctSigned(ex.vision_shift_pct)}
                    </span>
                  </div>
                  <p className="text-muted-foreground mt-0.5 truncate">
                    {ex.property_type} · {ex.livable_area} m² · EPC {ex.energy_label ?? "?"} · Claimed {formatCondition(ex.condition)}
                  </p>
                </div>
                <div className="text-right shrink-0 tabular-nums">
                  <p className="font-semibold text-foreground">
                    {formatEurCompact(ex.asking_eur)} → <span className="text-positive">{formatEurCompact(ex.fair_eur)}</span>
                  </p>
                  <a href={ex.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline inline-flex items-center gap-0.5">
                    #{ex.property_id} <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Mixed-Use / Store / Investment Regime Performance Table */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="p-5 border-b border-border">
          <h3 className="font-display text-lg font-semibold">
            How Phase-7 Handles Mixed-Use, Store-Attached &amp; Investment Properties
          </h3>
          <p className="text-xs text-muted-foreground mt-1 max-w-4xl">
            Immoweb often tags a house with a ground-floor shop (<em>handelspand</em>), practice space (<em>praktijkruimte</em>),
            or multi-unit rental (<em>opbrengsteigendom</em>) as a standard <code className="text-foreground">HOUSE</code>.
            Phase-7 still prices them with low bias using text + vision cues, but valuation spread is systematically wider
            (~8.5–14% MdAPE vs 7.2% pure residential) — so live predictions now display an explicit regime advisory.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/60 border-b border-border text-xs text-muted-foreground">
              <tr>
                {(
                  [
                    ["label", "Property Regime", "left", "asc"],
                    ["count", "Listings", "right", "desc"],
                    ["median_asking_eur", "Median Ask", "right", "desc"],
                    ["mdape", "Phase-7 MdAPE", "right", "desc"],
                    ["p80_err", "P80 Error", "right", "desc"],
                    ["avg_duration_days", "Avg Stay", "right", "desc"],
                    ["median_jepa_energy", "Med. JEPA Energy", "right", "desc"],
                  ] as const
                ).map(([key, label, align, defDir], idx, arr) => (
                  <th
                    key={key}
                    onClick={() => toggleRegSort(key, defDir)}
                    className={`${align === "left" ? "text-left" : "text-right"} ${
                      idx === 0 || idx === arr.length - 1 ? "px-4" : "px-3"
                    } py-2.5 font-medium cursor-pointer select-none hover:text-foreground whitespace-nowrap`}
                  >
                    {label}{" "}
                    <span className={`ml-0.5 text-[10px] ${regSort.key === key ? "text-primary" : "opacity-35"}`}>
                      {sortGlyph(regSort.key === key, regSort.dir)}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 tabular-nums">
              {sortedRegimes.map((r) => {
                const isStd = r.code === "residential";
                return (
                  <tr key={r.code} className={isStd ? "bg-primary/5 font-medium" : "hover:bg-muted/40"}>
                    <td className="px-4 py-2.5 text-foreground">{r.label}</td>
                    <td className="px-3 py-2.5 text-right text-muted-foreground">{r.count.toLocaleString()}</td>
                    <td className="px-3 py-2.5 text-right">{formatEurCompact(r.median_asking_eur)}</td>
                    <td className={`px-3 py-2.5 text-right font-semibold ${r.mdape >= 9 ? "text-negative" : r.mdape >= 8 ? "text-warning" : "text-positive"}`}>
                      {r.mdape.toFixed(2)}%
                    </td>
                    <td className="px-3 py-2.5 text-right text-muted-foreground">{r.p80_err.toFixed(1)}%</td>
                    <td className="px-3 py-2.5 text-right text-muted-foreground">{r.avg_duration_days}d</td>
                    <td className="px-4 py-2.5 text-right text-muted-foreground">
                      {r.median_jepa_energy != null ? r.median_jepa_energy.toFixed(3) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
