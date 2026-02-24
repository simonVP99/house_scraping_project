import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../services/api";

export function useLiveListings(neighbourhood: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ["liveListings", neighbourhood],
    queryFn: () => api.getLiveListings(neighbourhood ?? undefined),
    enabled: !!neighbourhood && enabled,
    staleTime: 5 * 60 * 1000,
  });
}

export function useRefreshListings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: api.refreshListings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["liveListings"] });
    },
  });
}
