import { useState, useEffect, useMemo, useId } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { FEATURE_FIELDS } from "../types/prediction";
import type { FeatureFieldMeta } from "../types/prediction";

interface Props {
  formData: Record<string, unknown>;
  onRepredict: (data: Record<string, unknown>) => void;
  isPending: boolean;
}

export default function FeatureForm({ formData, onRepredict, isPending }: Props) {
  const [values, setValues] = useState<Record<string, unknown>>({ ...formData });
  // All groups collapsed by default for progressive disclosure
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // M4 fix: sync when formData changes (new prediction)
  useEffect(() => {
    setValues({ ...formData });
  }, [formData]);

  const groups = useMemo(() => {
    const map = new Map<string, FeatureFieldMeta[]>();
    for (const field of FEATURE_FIELDS) {
      const list = map.get(field.group) || [];
      list.push(field);
      map.set(field.group, list);
    }
    return map;
  }, []);

  const isModified = (key: string) => {
    const orig = formData[key];
    const curr = values[key];
    if (orig == null && curr == null) return false;
    if (orig == null || curr == null) return true;
    return String(orig) !== String(curr);
  };

  const hasChanges = FEATURE_FIELDS.some((f) => isModified(f.key));
  const modifiedCount = FEATURE_FIELDS.filter((f) => isModified(f.key)).length;

  const setValue = (key: string, raw: string | boolean) => {
    setValues((prev) => {
      const field = FEATURE_FIELDS.find((f) => f.key === key);
      let val: unknown = raw;
      if (field?.type === "number") {
        val = raw === "" ? null : Number(raw);
      }
      return { ...prev, [key]: val };
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onRepredict(values);
  };

  const toggleGroup = (group: string) => {
    setExpanded((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  return (
    <section aria-label="Edit features">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-display text-lg font-semibold">All Features</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Edit values to correct parsing or add info, then re-predict
          </p>
        </div>
        <Button
          size="sm"
          onClick={handleSubmit}
          disabled={isPending || !hasChanges}
          aria-label={`Re-predict with ${modifiedCount} changed features`}
        >
          {isPending ? "Predicting..." : hasChanges ? `Re-predict (${modifiedCount})` : "Re-predict"}
        </Button>
      </div>
      <form onSubmit={handleSubmit} className="space-y-2">
        {Array.from(groups.entries()).map(([group, fields]) => {
          const modCount = fields.filter((f) => isModified(f.key)).length;
          const isOpen = expanded[group] === true;
          return (
            <div key={group} className="border border-border rounded-lg overflow-hidden">
              <button
                type="button"
                onClick={() => toggleGroup(group)}
                aria-expanded={isOpen}
                className="flex w-full items-center justify-between px-4 py-2.5 text-sm font-medium text-foreground hover:bg-muted/50 transition-colors"
              >
                <span className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground select-none w-4">
                    {isOpen ? "\u25BC" : "\u25B6"}
                  </span>
                  {group}
                </span>
                <span className="flex items-center gap-2">
                  {modCount > 0 && (
                    <span className="text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">
                      {modCount} changed
                    </span>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {fields.length} fields
                  </span>
                </span>
              </button>

              {isOpen && (
                <div className="px-4 pb-4 pt-2 border-t border-border bg-muted/20">
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-2">
                    {fields.map((field) => (
                      <FieldInput
                        key={field.key}
                        field={field}
                        value={values[field.key]}
                        modified={isModified(field.key)}
                        onChange={(v) => setValue(field.key, v)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </form>
    </section>
  );
}

function FieldInput({
  field,
  value,
  modified,
  onChange,
}: {
  field: FeatureFieldMeta;
  value: unknown;
  modified: boolean;
  onChange: (v: string | boolean) => void;
}) {
  const id = useId();
  const borderClass = modified ? "border-l-2 border-l-primary pl-2" : "pl-2";

  if (field.type === "boolean") {
    const checked = value === true || value === "True" || value === 1;
    return (
      <label
        htmlFor={id}
        className={`flex items-center gap-2 text-sm py-1 cursor-pointer ${borderClass}`}
      >
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="rounded border-input"
        />
        <span className={modified ? "text-primary font-medium" : ""}>
          {field.label}
        </span>
      </label>
    );
  }

  if (field.type === "select") {
    const strVal = value != null ? String(value) : "";
    return (
      <div className={borderClass}>
        <label htmlFor={id} className="text-xs text-muted-foreground">
          {field.label}
        </label>
        <select
          id={id}
          value={strVal}
          onChange={(e) => onChange(e.target.value)}
          className="flex h-8 w-full rounded-md border border-input bg-transparent px-2 py-1 text-xs shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <option value="">--</option>
          {field.options?.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  const strVal = value != null ? String(value) : "";
  return (
    <div className={borderClass}>
      <label htmlFor={id} className="text-xs text-muted-foreground">
        {field.label}
      </label>
      <Input
        id={id}
        type={field.type === "number" ? "number" : "text"}
        step={field.type === "number" ? "any" : undefined}
        value={strVal}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 text-xs"
      />
    </div>
  );
}
