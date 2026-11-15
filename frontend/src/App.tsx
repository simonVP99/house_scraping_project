import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PredictPage from "./pages/PredictPage";
import NeighbourhoodsPage from "./pages/NeighbourhoodsPage";
import EdaPage from "./pages/EdaPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-background">
          <nav
            className="sticky top-0 z-40 border-b border-border bg-card/80 backdrop-blur-sm"
            role="navigation"
            aria-label="Main navigation"
          >
            <div className="mx-auto max-w-6xl px-6 flex h-14 items-center gap-8">
              <NavLink to="/" className="flex items-baseline gap-1.5 group">
                <span className="font-display text-lg font-bold tracking-tight text-foreground group-hover:text-primary transition-colors">
                  Ghent RE
                </span>
                <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                  Intelligence
                </span>
              </NavLink>
              <div className="flex gap-0.5 ml-auto">
                <NavLink
                  to="/predict"
                  className={({ isActive }) =>
                    `relative px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      isActive
                        ? "text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      Predict
                      {isActive && (
                        <span className="absolute bottom-0 left-3 right-3 h-0.5 bg-primary rounded-full" />
                      )}
                    </>
                  )}
                </NavLink>
                <NavLink
                  to="/neighbourhoods"
                  className={({ isActive }) =>
                    `relative px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      isActive
                        ? "text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      Neighbourhoods
                      {isActive && (
                        <span className="absolute bottom-0 left-3 right-3 h-0.5 bg-primary rounded-full" />
                      )}
                    </>
                  )}
                </NavLink>
                <NavLink
                  to="/eda"
                  className={({ isActive }) =>
                    `relative px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      isActive
                        ? "text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      Dataset EDA
                      {isActive && (
                        <span className="absolute bottom-0 left-3 right-3 h-0.5 bg-primary rounded-full" />
                      )}
                    </>
                  )}
                </NavLink>
              </div>
            </div>
          </nav>
          <main className="mx-auto max-w-6xl px-6 py-10">
            <Routes>
              <Route path="/" element={<PredictPage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/neighbourhoods" element={<NeighbourhoodsPage />} />
              <Route path="/eda" element={<EdaPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
