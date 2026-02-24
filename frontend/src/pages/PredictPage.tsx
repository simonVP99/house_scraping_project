import UrlAnalyzer from "../components/UrlAnalyzer";

export default function PredictPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Price Predictor</h1>
        <p className="text-muted-foreground">
          Paste an Immoweb URL to get an instant price prediction
        </p>
      </div>
      <UrlAnalyzer />
    </div>
  );
}
