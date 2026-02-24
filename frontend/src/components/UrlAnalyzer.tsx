import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useUrlPrediction } from "../hooks/usePrediction";
import PredictionResultCard from "./PredictionResult";

export default function UrlAnalyzer() {
  const [url, setUrl] = useState("");
  const mutation = useUrlPrediction();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) mutation.mutate(url.trim());
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Check a Property</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex gap-3">
            <Input
              type="url"
              placeholder="https://www.immoweb.be/en/classified/house/for-sale/gent/9000/..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="flex-1"
            />
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? "Analyzing..." : "Check Price"}
            </Button>
          </form>
          {mutation.isError && (
            <p className="mt-3 text-sm text-red-600">{mutation.error.message}</p>
          )}
        </CardContent>
      </Card>

      {mutation.data && <PredictionResultCard data={mutation.data} />}
    </div>
  );
}
