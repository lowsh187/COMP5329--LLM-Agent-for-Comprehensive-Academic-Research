import { demoResult } from "./mockData";
import type { ResearchRequest, ResearchResult } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function runResearch(request: ResearchRequest, demoMode: boolean): Promise<ResearchResult> {
  if (demoMode) {
    await delay(500);
    return {
      ...demoResult,
      topic: request.topic
    };
  }

  const response = await fetch(`${API_BASE_URL}/api/research/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Backend request failed");
  }

  return response.json();
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
