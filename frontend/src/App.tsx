import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const PredictPage = lazy(() => import("./pages/PredictPage"));
const AffordabilityPage = lazy(() => import("./pages/AffordabilityPage"));
const NeighbourhoodsPage = lazy(() => import("./pages/NeighbourhoodsPage"));
const InsightsPage = lazy(() => import("./pages/InsightsPage"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      retry: 1,
    },
  },
});

const NAV_ITEMS = [
  { to: "/predict", label: "Predict" },
  { to: "/explore", label: "Affordability" },
  { to: "/neighbourhoods", label: "Neighbourhoods" },
  { to: "/insights", label: "Insights & EDA" },
];

function PageFallback() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-4">
      <div className="h-8 w-64 bg-muted rounded animate-warm-pulse" />
      <div className="h-4 w-96 bg-muted/70 rounded animate-warm-pulse" />
      <div className="h-72 bg-muted/50 rounded-xl animate-warm-pulse" />
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-background text-foreground flex flex-col">
          <nav
            className="border-b border-border bg-card/90 backdrop-blur-md sticky top-0 z-50"
            aria-label="Main navigation"
          >
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-4">
              <NavLink to="/predict" className="flex items-baseline gap-1.5 group shrink-0">
                <span className="font-display text-lg font-bold tracking-tight text-foreground">
                  Ghent RE
                </span>
                <span className="text-[11px] font-semibold tracking-widest uppercase text-primary">
                  Intelligence
                </span>
              </NavLink>

              <div className="flex items-center gap-1 sm:gap-2 overflow-x-auto">
                {NAV_ITEMS.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={({ isActive }) =>
                      `px-3 py-1.5 rounded-md text-sm transition-colors whitespace-nowrap ${
                        isActive
                          ? "bg-primary/10 text-primary font-semibold"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ))}
              </div>
            </div>
          </nav>

          <main className="flex-1">
            <Suspense fallback={<PageFallback />}>
              <Routes>
                <Route path="/" element={<Navigate to="/predict" replace />} />
                <Route path="/predict" element={<PredictPage />} />
                <Route path="/explore" element={<AffordabilityPage />} />
                <Route path="/neighbourhoods" element={<NeighbourhoodsPage />} />
                <Route path="/insights" element={<InsightsPage />} />
                <Route path="/eda" element={<Navigate to="/insights" replace />} />
              </Routes>
            </Suspense>
          </main>

          <footer className="border-t border-border/70 bg-card/50 mt-12">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
              <span>Ghent Real Estate Intelligence · Phase-7 Lean-5 Agency-Blind Valuation Engine</span>
              <span>CPI+HPI Deflated · Anti-Chaining Spatial+Text CV · Conformal 80% / 90%</span>
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
