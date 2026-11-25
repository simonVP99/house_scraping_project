import { formatEur, formatEurCompact, formatEurSigned, formatPctSigned } from "../lib/format";

interface Props {
  predicted: number;
  low80: number;
  high80: number;
  low90?: number | null;
  high90?: number | null;
  asking?: number | null;
  halfwidthPct?: number | null;
  bandPosition?: "below_band" | "inside_band" | "above_band" | null;
}

/**
 * Visual heteroscedastic conformal prediction gauge.
 * Shows the 90% outer range, 80% inner confidence band, point prediction,
 * and (when provided) the asking price marker directly on the same scale.
 */
export default function ConformalBandGauge({
  predicted,
  low80,
  high80,
  low90,
  high90,
  asking,
  halfwidthPct,
  bandPosition,
}: Props) {
  const outerLow = low90 ?? low80 * 0.92;
  const outerHigh = high90 ?? high80 * 1.08;

  // Domain covers outer 90% band + asking price (if any) with 6% visual breathing room
  const rawMin = Math.min(outerLow, asking ?? outerLow);
  const rawMax = Math.max(outerHigh, asking ?? outerHigh);
  const pad = Math.max((rawMax - rawMin) * 0.08, 15_000);
  const minVal = rawMin - pad;
  const maxVal = rawMax + pad;
  const span = Math.max(maxVal - minVal, 1);

  const pct = (v: number) => Math.max(1, Math.min(99, ((v - minVal) / span) * 100));

  const left90 = pct(outerLow);
  const width90 = Math.max(2, pct(outerHigh) - left90);
  const left80 = pct(low80);
  const width80 = Math.max(2, pct(high80) - left80);
  const posPred = pct(predicted);
  const posAsking = asking != null ? pct(asking) : null;

  const isUnderpriced = asking != null && asking < predicted;
  const deltaEur = asking != null ? predicted - asking : null;
  const deltaPct = asking != null && asking > 0 ? ((predicted - asking) / asking) * 100 : null;

  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
            Conformal Uncertainty Band
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            Calibrated out-of-fold intervals — wider for unusual homes, tighter for standard Ghent stock
          </p>
        </div>
        {halfwidthPct != null && (
          <span className="text-xs font-medium text-muted-foreground tabular-nums bg-muted px-2.5 py-1 rounded-full">
            80% half-width: ±{halfwidthPct.toFixed(1)}%
          </span>
        )}
      </div>

      {/* Gauge track */}
      <div className="pt-7 pb-6 relative select-none">
        {/* Asking price callout above track */}
        {posAsking != null && asking != null && (
          <div
            className="absolute top-0 -translate-x-1/2 flex flex-col items-center z-20"
            style={{ left: `${posAsking}%` }}
          >
            <span
              className={`text-[11px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap shadow-sm tabular-nums ${
                isUnderpriced
                  ? "bg-positive text-white"
                  : "bg-negative text-white"
              }`}
            >
              Asking {formatEurCompact(asking)}
            </span>
          </div>
        )}

        {/* Base rail */}
        <div className="relative h-5 rounded-full bg-muted/70 border border-border/70">
          {/* 90% outer band */}
          <div
            className="absolute top-0 bottom-0 rounded-full bg-primary/15"
            style={{ left: `${left90}%`, width: `${width90}%` }}
            title={`90% range: ${formatEur(outerLow)} – ${formatEur(outerHigh)}`}
          />

          {/* 80% inner band */}
          <div
            className="absolute top-0.5 bottom-0.5 rounded-full bg-primary/35 border border-primary/40"
            style={{ left: `${left80}%`, width: `${width80}%` }}
            title={`80% range: ${formatEur(low80)} – ${formatEur(high80)}`}
          />

          {/* Point fair-value marker */}
          <div
            className="absolute -top-1 -bottom-1 w-1 rounded-full bg-primary shadow-sm -translate-x-1/2 z-10"
            style={{ left: `${posPred}%` }}
            title={`Fair value: ${formatEur(predicted)}`}
          />

          {/* Asking price pin */}
          {posAsking != null && (
            <div
              className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full border-2 border-card shadow-md z-20 ${
                isUnderpriced ? "bg-positive" : "bg-negative"
              }`}
              style={{ left: `${posAsking}%` }}
            />
          )}
        </div>

        {/* Fair value label below */}
        <div
          className="absolute bottom-0 -translate-x-1/2 text-[11px] font-semibold text-primary whitespace-nowrap tabular-nums"
          style={{ left: `${posPred}%` }}
        >
          Fair {formatEurCompact(predicted)}
        </div>
      </div>

      {/* Numeric summary grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 border-t border-border/70 text-xs">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">90% Low – High</p>
          <p className="font-medium text-foreground tabular-nums mt-0.5">
            {formatEurCompact(outerLow)} – {formatEurCompact(outerHigh)}
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">80% Low – High</p>
          <p className="font-semibold text-foreground tabular-nums mt-0.5">
            {formatEur(low80)} – {formatEur(high80)}
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Point Estimate</p>
          <p className="font-semibold text-primary tabular-nums mt-0.5">{formatEur(predicted)}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Asking vs Band</p>
          {asking != null && deltaEur != null && deltaPct != null ? (
            <p
              className={`font-semibold tabular-nums mt-0.5 ${
                bandPosition === "below_band"
                  ? "text-positive"
                  : bandPosition === "above_band"
                    ? "text-negative"
                    : "text-foreground"
              }`}
            >
              {formatEurSigned(deltaEur)} ({formatPctSigned(deltaPct)})
            </p>
          ) : (
            <p className="text-muted-foreground mt-0.5">No asking price</p>
          )}
        </div>
      </div>
    </div>
  );
}
