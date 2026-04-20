import type { ChatTurn } from "../src/agent/client.js";
import type { RetrievedItem } from "../src/types.js";

export async function createOcbAdapter(): Promise<{
  route(
    history: ChatTurn[],
    query: string,
  ): Promise<{
    fire: boolean;
    retrieved: RetrievedItem[];
    injected_text: string;
    gate_score: number;
    gate_threshold: number;
  }>;
}> {
  return {
    async route(_history: ChatTurn[], _query: string) {
      throw new Error(
        "OCB adapter not implemented. Wire this to OCB's real gate and retrieval path, and forward gate_score plus gate_threshold for each turn.",
      );
    },
  };
}
