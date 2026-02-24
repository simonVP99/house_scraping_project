import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PredictPage from "./pages/PredictPage";
import NeighbourhoodsPage from "./pages/NeighbourhoodsPage";

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
          <nav className="border-b bg-white">
            <div className="mx-auto max-w-7xl px-4 flex h-14 items-center gap-6">
              <span className="font-bold text-lg">Ghent RE Intelligence</span>
              <NavLink
                to="/predict"
                className={({ isActive }) =>
                  `text-sm font-medium ${isActive ? "text-blue-600 border-b-2 border-blue-600" : "text-muted-foreground"}`
                }
              >
                Price Predictor
              </NavLink>
              <NavLink
                to="/neighbourhoods"
                className={({ isActive }) =>
                  `text-sm font-medium ${isActive ? "text-blue-600 border-b-2 border-blue-600" : "text-muted-foreground"}`
                }
              >
                Neighbourhoods
              </NavLink>
            </div>
          </nav>
          <main className="mx-auto max-w-7xl px-4 py-6">
            <Routes>
              <Route path="/" element={<PredictPage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/neighbourhoods" element={<NeighbourhoodsPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
