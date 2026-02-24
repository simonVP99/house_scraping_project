import { useMutation } from "@tanstack/react-query";
import { api } from "../services/api";

export function useUrlPrediction() {
  return useMutation({ mutationFn: api.predictFromUrl });
}

export function useManualPrediction() {
  return useMutation({ mutationFn: api.predictManual });
}
