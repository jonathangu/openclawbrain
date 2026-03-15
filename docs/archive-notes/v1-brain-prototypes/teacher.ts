/**
 * Off-path async teacher for episode evaluation.
 *
 * CRITICAL RULE: Teacher sees ONLY what the router saw.
 * It evaluates the routing decision, not the overall task outcome.
 * No cheating with extra context.
 */

import type { Episode } from "./types.js";
import type { BrainGraph } from "./graph.js";
import type { CompleteFn, ResolveModelFn, GetApiKeyFn } from "../types.js";

const TEACHER_SYSTEM_PROMPT =
  "You are evaluating a context routing decision. Score the quality of the selected context for the given query. Return ONLY a JSON object: {\"score\": <number from -1.0 to 1.0>, \"reason\": \"<brief explanation>\"}";

export class BrainTeacher {
  constructor(
    private complete: CompleteFn,
    private resolveModel: ResolveModelFn,
    private getApiKey: GetApiKeyFn,
    private graph: BrainGraph,
    private log: { info: (msg: string) => void; error: (msg: string) => void },
  ) {}

  async evaluate(episode: Episode): Promise<{ score: number; reason: string }> {
    // Build prompt showing only what the router saw
    const candidateDescriptions: string[] = [];
    for (const step of episode.trajectory) {
      for (const candidate of step.candidates) {
        if (candidate.action.type === "traverse") {
          const node = this.graph.getNode(candidate.action.targetNodeId);
          if (node) {
            candidateDescriptions.push(
              `- [${node.kind}] ${node.content.slice(0, 200)}${node.content.length > 200 ? "..." : ""} (prob: ${candidate.probability.toFixed(3)})`,
            );
          }
        }
      }
    }

    const firedDescriptions = episode.firedNodes.map((id) => {
      const node = this.graph.getNode(id);
      return node
        ? `- [${node.kind}] ${node.content.slice(0, 200)}${node.content.length > 200 ? "..." : ""}`
        : `- ${id} (not found)`;
    });

    const prompt = `Query: "${episode.queryText}"

Candidate nodes the router could have chosen:
${candidateDescriptions.join("\n") || "(none)"}

Nodes actually selected (fired):
${firedDescriptions.join("\n") || "(none)"}

Was this the right context for the query? Consider relevance, completeness, and conciseness.`;

    try {
      const { provider, model } = this.resolveModel();
      const apiKey = await this.getApiKey(provider, model);

      const result = await this.complete({
        provider,
        model,
        apiKey,
        system: TEACHER_SYSTEM_PROMPT,
        messages: [{ role: "user", content: prompt }],
        maxTokens: 200,
        temperature: 0.1,
      });

      const text = result.content
        ?.map((b: { text?: string }) => b.text ?? "")
        .join("") ?? "";

      const jsonMatch = text.match(/\{[\s\S]*"score"[\s\S]*\}/);
      if (!jsonMatch) {
        this.log.error(`[brain] Teacher returned non-JSON: ${text.slice(0, 100)}`);
        return { score: 0, reason: "failed to parse teacher response" };
      }

      const parsed = JSON.parse(jsonMatch[0]);
      const score = Math.max(-1, Math.min(1, Number(parsed.score) || 0));
      const reason = String(parsed.reason || "teacher evaluation");

      this.log.info(`[brain] Teacher scored episode ${episode.id}: ${score.toFixed(2)} (${reason})`);
      return { score, reason };
    } catch (err) {
      this.log.error(`[brain] Teacher evaluation failed: ${(err as Error).message}`);
      return { score: 0, reason: "teacher evaluation failed" };
    }
  }
}
