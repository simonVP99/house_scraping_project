import { useState } from "react";
import { Sparkles, RotateCcw, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatEur, formatEurSigned, formatPctSigned, formatCondition } from "../lib/format";

interface Props {
  baseFormData: Record<string, unknown>;
  basePrediction: number;
  currentPrediction: number;
  onSimulate: (nextFormData: Record<string, unknown>) => void;
  onReset: () => void;
  isPending: boolean;
}

const EPC_OPTIONS = ["A++", "A+", "A", "B", "C", "D", "E", "F"];
const CONDITION_OPTIONS = [
  { value: "AS_NEW", label: "As new" },
  { value: "JUST_RENOVATED", label: "Just renovated" },
  { value: "GOOD", label: "Good" },
  { value: "TO_BE_DONE_UP", label: "To refresh" },
  { value: "TO_RENOVATE", label: "To renovate" },
  { value: "TO_RESTORE", label: "To restore" },
];

/**
 * Interactive Phase-7 What-If / Renovation ROI Simulator.
 * Lets users tweak high-leverage upgrades (EPC, condition, livable area, garden,
 * solar panels, heat pump) and immediately re-score against the real Lean-5 ensemble.
 */
export default function WhatIfSimulator({
  baseFormData,
  basePrediction,
  currentPrediction,
  onSimulate,
  onReset,
  isPending,
}: Props) {
  const [prevBase, setPrevBase] = useState(baseFormData);
  const [epc, setEpc] = useState<string>(String(baseFormData.energy_label ?? "C"));
  const [condition, setCondition] = useState<string>(String(baseFormData.building_condition ?? "GOOD"));
  const [area, setArea] = useState<number>(Number(baseFormData.livable_area ?? 120));
  const [solar, setSolar] = useState<boolean>(Boolean(baseFormData.has_photovoltaic_panels));
  const [heatPump, setHeatPump] = useState<boolean>(Boolean(baseFormData.has_heat_pump));
  const [doubleGlazing, setDoubleGlazing] = useState<boolean>(
    baseFormData.has_double_glazing == null ? true : Boolean(baseFormData.has_double_glazing),
  );

  if (baseFormData !== prevBase) {
    setPrevBase(baseFormData);
    setEpc(String(baseFormData.energy_label ?? "C"));
    setCondition(String(baseFormData.building_condition ?? "GOOD"));
    setArea(Number(baseFormData.livable_area ?? 120));
    setSolar(Boolean(baseFormData.has_photovoltaic_panels));
    setHeatPump(Boolean(baseFormData.has_heat_pump));
    setDoubleGlazing(baseFormData.has_double_glazing == null ? true : Boolean(baseFormData.has_double_glazing));
  }

  const baseEpc = String(baseFormData.energy_label ?? "C");
  const baseCond = String(baseFormData.building_condition ?? "GOOD");
  const baseArea = Number(baseFormData.livable_area ?? 120);
  const baseSolar = Boolean(baseFormData.has_photovoltaic_panels);
  const baseHp = Boolean(baseFormData.has_heat_pump);
  const baseDg = baseFormData.has_double_glazing == null ? true : Boolean(baseFormData.has_double_glazing);

  const hasTweaks =
    epc !== baseEpc ||
    condition !== baseCond ||
    Math.round(area) !== Math.round(baseArea) ||
    solar !== baseSolar ||
    heatPump !== baseHp ||
    doubleGlazing !== baseDg;

  const delta = currentPrediction - basePrediction;
  const deltaPct = basePrediction > 0 ? (delta / basePrediction) * 100 : 0;

  const runSimulation = () => {
    onSimulate({
      ...baseFormData,
      energy_label: epc,
      building_condition: condition,
      livable_area: area,
      has_photovoltaic_panels: solar,
      has_heat_pump: heatPump,
      has_double_glazing: doubleGlazing,
    });
  };

  const applyPreset = (preset: "epc_b" | "full_reno") => {
    const next =
      preset === "epc_b"
        ? {
            energy_label: "B",
            building_condition: condition,
            livable_area: area,
            has_photovoltaic_panels: true,
            has_heat_pump: true,
            has_double_glazing: true,
          }
        : {
            energy_label: "A",
            building_condition: "JUST_RENOVATED",
            livable_area: area,
            has_photovoltaic_panels: true,
            has_heat_pump: true,
            has_double_glazing: true,
          };
    setEpc(next.energy_label);
    setCondition(next.building_condition);
    setSolar(next.has_photovoltaic_panels);
    setHeatPump(next.has_heat_pump);
    setDoubleGlazing(next.has_double_glazing);
    onSimulate({ ...baseFormData, ...next });
  };

  return (
    <section aria-label="Renovation what-if simulator" className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-primary" />
            <h3 className="font-display text-lg font-semibold">Renovation & What-If Simulator</h3>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            Simulate EPC upgrades, renovation, or extension live through the Phase-7 ensemble
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => applyPreset("epc_b")}
            disabled={isPending}
            className="text-xs px-2.5 py-1 rounded-full border border-border bg-muted/50 hover:bg-muted text-foreground transition-colors cursor-pointer"
          >
            Preset: Upgrade to EPC B + Solar
          </button>
          <button
            type="button"
            onClick={() => applyPreset("full_reno")}
            disabled={isPending}
            className="text-xs px-2.5 py-1 rounded-full border border-border bg-muted/50 hover:bg-muted text-foreground transition-colors cursor-pointer"
          >
            Preset: Full Renovation (EPC A)
          </button>
        </div>
      </div>

      {/* Before / After Delta Banner when simulated */}
      {Math.abs(delta) >= 500 && (
        <div
          className={`flex flex-wrap items-center justify-between gap-3 rounded-lg border px-4 py-3 text-sm ${
            delta > 0
              ? "border-positive/25 bg-positive/5"
              : "border-negative/25 bg-negative/5"
          }`}
        >
          <div className="flex items-center gap-2.5 flex-wrap tabular-nums">
            <span className="text-muted-foreground">Baseline {formatEur(basePrediction)}</span>
            <ArrowRight className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="font-semibold text-foreground">Simulated {formatEur(currentPrediction)}</span>
            <span className={`font-bold ${delta > 0 ? "text-positive" : "text-negative"}`}>
              {formatEurSigned(delta)} ({formatPctSigned(deltaPct)})
            </span>
          </div>
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Reset to original
          </button>
        </div>
      )}

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1.5">
            Energy Label (EPC) <span className="text-foreground/60">· was {baseEpc}</span>
          </label>
          <div className="flex flex-wrap gap-1">
            {EPC_OPTIONS.map((opt) => {
              const active = epc === opt;
              const isCliff = opt === "E" || opt === "F";
              return (
                <button
                  key={opt}
                  type="button"
                  onClick={() => setEpc(opt)}
                  className={`px-2 py-1 text-xs font-semibold rounded-md border transition-colors cursor-pointer ${
                    active
                      ? "bg-primary text-primary-foreground border-primary"
                      : isCliff
                        ? "border-negative/30 text-negative/90 bg-negative/5 hover:bg-negative/10"
                        : "border-border bg-background text-foreground hover:bg-muted"
                  }`}
                >
                  {opt}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <label htmlFor="sim-condition" className="block text-xs font-medium text-muted-foreground mb-1.5">
            Condition <span className="text-foreground/60">· was {formatCondition(baseCond)}</span>
          </label>
          <select
            id="sim-condition"
            value={condition}
            onChange={(e) => setCondition(e.target.value)}
            className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          >
            {CONDITION_OPTIONS.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>

        <div>
          <div className="flex items-center justify-between text-xs mb-1.5">
            <label htmlFor="sim-area" className="font-medium text-muted-foreground">
              Livable Area
            </label>
            <span className="font-semibold tabular-nums text-foreground">
              {Math.round(area)} m²
              {Math.round(area) !== Math.round(baseArea) && (
                <span className="text-primary ml-1">
                  ({area > baseArea ? "+" : ""}{Math.round(area - baseArea)} m²)
                </span>
              )}
            </span>
          </div>
          <input
            id="sim-area"
            type="range"
            min={Math.max(30, Math.round(baseArea * 0.65))}
            max={Math.min(450, Math.round(baseArea * 1.5))}
            step={5}
            value={area}
            onChange={(e) => setArea(Number(e.target.value))}
            className="w-full accent-[var(--color-primary)] cursor-pointer mt-1"
          />
        </div>
      </div>

      {/* Tech toggles + Run */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-border/70">
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="inline-flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={solar}
              onChange={(e) => setSolar(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)]"
            />
            <span>Solar panels</span>
          </label>
          <label className="inline-flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={heatPump}
              onChange={(e) => setHeatPump(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)]"
            />
            <span>Heat pump</span>
          </label>
          <label className="inline-flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={doubleGlazing}
              onChange={(e) => setDoubleGlazing(e.target.checked)}
              className="rounded border-input accent-[var(--color-primary)]"
            />
            <span>Double glazing</span>
          </label>
        </div>

        <Button
          type="button"
          size="sm"
          onClick={runSimulation}
          disabled={isPending || !hasTweaks}
        >
          {isPending ? "Scoring..." : "Simulate Value"}
        </Button>
      </div>
    </section>
  );
}
