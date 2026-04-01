import { Type } from "@sinclair/typebox";
import crypto from "node:crypto";
import type { LcmContextEngine } from "../engine.js";
import {
  createDelegatedExpansionGrant,
  revokeDelegatedExpansionGrantForSession,
} from "../expansion-auth.js";
import type { LcmDependencies } from "../types.js";
import { jsonResult, type AnyAgentTool } from "./common.js";
import { resolveLcmConversationScope } from "./lcm-conversation-scope.js";
import {
  normalizeSummaryIds,
  resolveRequesterConversationScopeId,
} from "./lcm-expand-tool.delegation.js";
import {
  clearDelegatedExpansionContext,
  evaluateExpansionRecursionGuard,
  recordExpansionDelegationTelemetry,
  resolveExpansionRequestId,
  resolveNextExpansionDepth,
  stampDelegatedExpansionContext,
} from "./lcm-expansion-recursion-guard.js";

const DELEGATED_WAIT_TIMEOUT_MS = 120_000;
const GATEWAY_TIMEOUT_MS = 10_000;
const DEFAULT_MAX_ANSWER_TOKENS = 2_000;

const LcmExpandQuerySchema = Type.Object({
  summaryIds: Type.Optional(
    Type.Array(Type.String(), {
      description:
        "Summary IDs to expand (sum_xxx). Marble IDs (mar_xxx) are also accepted as seeds and resolve to source refs. Required when query is not provided.",
    }),
  ),
  query: Type.Optional(
    Type.String({
      description:
        "Text query used to find summaries via grep before expansion. Required when summaryIds is not provided.",
    }),
  ),
  prompt: Type.String({
    description: "Question to answer using expanded context.",
  }),
  conversationId: Type.Optional(
    Type.Number({
      description:
        "Conversation ID to scope expansion to. If omitted, uses the current session conversation.",
    }),
  ),
  allConversations: Type.Optional(
    Type.Boolean({
      description:
        "Set true to explicitly allow cross-conversation lookup. Ignored when conversationId is provided.",
    }),
  ),
  maxTokens: Type.Optional(
    Type.Number({
      description: `Maximum answer tokens to target (default: ${DEFAULT_MAX_ANSWER_TOKENS}).`,
      minimum: 1,
    }),
  ),
  tokenCap: Type.Optional(
    Type.Number({
      description:
        "Expansion retrieval token budget across all delegated lcm_expand calls for this query.",
      minimum: 1,
    }),
  ),
});

type ExpandQueryReply = {
  answer: string;
  citedIds: string[];
  expandedSummaryCount: number;
  totalSourceTokens: number;
  truncated: boolean;
};

type MarbleSeedSnapshot = {
  marbleId: string;
  marbleKind: string;
  freshnessState: string;
  provenanceRef: string;
  sourceFingerprint: string;
  sourceCount: number;
  sourceArtifactTokenCount: number;
  sourceRefs: string[];
  sourceSummaryIds: string[];
};

type ExpansionCandidate = {
  seedId: string;
  seedType: "summary" | "marble";
  conversationId: number;
  summaryIds: string[];
  marble?: MarbleSeedSnapshot;
};

/**
 * Build the sub-agent task message for delegated expansion and prompt answering.
 */
function buildDelegatedExpandQueryTask(params: {
  summaryIds: string[];
  marbleSeeds: MarbleSeedSnapshot[];
  conversationId: number;
  query?: string;
  prompt: string;
  maxTokens: number;
  tokenCap: number;
  requestId: string;
  expansionDepth: number;
  originSessionKey: string;
}) {
  const seedSummaryIds = params.summaryIds.length > 0 ? params.summaryIds.join(", ") : "(none)";
  const marbleSeedLines =
    params.marbleSeeds.length > 0
      ? params.marbleSeeds
          .map(
            (marble) =>
              `- ${marble.marbleId} (${marble.marbleKind}, ${marble.freshnessState}) ` +
              `prov=${marble.provenanceRef} src=${marble.sourceFingerprint} ` +
              `links=${marble.sourceCount} tokens=${marble.sourceArtifactTokenCount} ` +
              `refs=${marble.sourceRefs.join(", ") || "(none)"}`,
          )
          .join("\n")
      : "- (none)";
  return [
    "You are an autonomous LCM retrieval navigator. Plan and execute retrieval before answering.",
    "",
    "Available tools: lcm_describe, lcm_expand, lcm_grep",
    `Conversation scope: ${params.conversationId}`,
    `Expansion token budget (total across this run): ${params.tokenCap}`,
    `Seed summary IDs: ${seedSummaryIds}`,
    "Marble seeds:",
    marbleSeedLines,
    params.query ? `Routing query: ${params.query}` : undefined,
    "",
    "Strategy:",
    "1. Start with `lcm_describe` on seed summaries and any marble seeds to inspect subtree manifests, provenance, and branch costs.",
    "2. If additional candidates are needed, use `lcm_grep` scoped to summaries and marbles.",
    "3. For marble seeds, inspect the marble provenance and then follow the underlying source refs; do not treat a marble as final proof when exactness matters.",
    "4. Select branches that fit remaining budget; prefer high-signal paths first.",
    "5. Call `lcm_expand` selectively (do not expand everything blindly).",
    "6. Keep includeMessages=false by default; use includeMessages=true only for specific leaf evidence.",
    `7. Stay within ${params.tokenCap} total expansion tokens across all lcm_expand calls.`,
    "",
    "User prompt to answer:",
    params.prompt,
    "",
    "Delegated expansion metadata (for tracing):",
    `- requestId: ${params.requestId}`,
    `- expansionDepth: ${params.expansionDepth}`,
    `- originSessionKey: ${params.originSessionKey}`,
    "",
    "Return ONLY JSON with this shape:",
    "{",
    '  "answer": "string",',
    '  "citedIds": ["sum_xxx"],',
    '  "expandedSummaryCount": 0,',
    '  "totalSourceTokens": 0,',
    '  "truncated": false',
    "}",
    "",
    "Rules:",
    "- In delegated context, call `lcm_expand` directly for source retrieval.",
    "- DO NOT call `lcm_expand_query` from this delegated session.",
    "- If a marble seed is stale or precision-sensitive, expand to the marble's source refs before answering.",
    "- Synthesize the final answer from retrieved evidence, not assumptions.",
    `- Keep answer concise and focused (target <= ${params.maxTokens} tokens).`,
    "- citedIds must be unique summary IDs.",
    "- expandedSummaryCount should reflect how many summaries were expanded/used.",
    "- totalSourceTokens should estimate total tokens consumed from expansion calls.",
    "- truncated should indicate whether source expansion appears truncated.",
  ].join("\n");
}

/**
 * Parse the child reply; accepts plain JSON or fenced JSON.
 */
function parseDelegatedExpandQueryReply(
  rawReply: string | undefined,
  fallbackExpandedSummaryCount: number,
): ExpandQueryReply {
  const fallback: ExpandQueryReply = {
    answer: (rawReply ?? "").trim(),
    citedIds: [],
    expandedSummaryCount: fallbackExpandedSummaryCount,
    totalSourceTokens: 0,
    truncated: false,
  };

  const reply = rawReply?.trim();
  if (!reply) {
    return fallback;
  }

  const candidates: string[] = [reply];
  const fenced = reply.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    candidates.unshift(fenced[1].trim());
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as {
        answer?: unknown;
        citedIds?: unknown;
        expandedSummaryCount?: unknown;
        totalSourceTokens?: unknown;
        truncated?: unknown;
      };
      const answer = typeof parsed.answer === "string" ? parsed.answer.trim() : "";
      const citedIds = normalizeSummaryIds(
        Array.isArray(parsed.citedIds)
          ? parsed.citedIds.filter((value): value is string => typeof value === "string")
          : undefined,
      );
      const expandedSummaryCount =
        typeof parsed.expandedSummaryCount === "number" &&
        Number.isFinite(parsed.expandedSummaryCount)
          ? Math.max(0, Math.floor(parsed.expandedSummaryCount))
          : fallbackExpandedSummaryCount;
      const totalSourceTokens =
        typeof parsed.totalSourceTokens === "number" && Number.isFinite(parsed.totalSourceTokens)
          ? Math.max(0, Math.floor(parsed.totalSourceTokens))
          : 0;
      const truncated = parsed.truncated === true;

      return {
        answer: answer || fallback.answer,
        citedIds,
        expandedSummaryCount,
        totalSourceTokens,
        truncated,
      };
    } catch {
      // Try next candidate.
    }
  }

  return fallback;
}

/**
 * Resolve a single source conversation for delegated expansion.
 */
function resolveSourceConversationId(params: {
  scopedConversationId?: number;
  allConversations: boolean;
  candidates: ExpansionCandidate[];
}): number {
  if (typeof params.scopedConversationId === "number") {
    const mismatched = params.candidates
      .filter((candidate) => candidate.conversationId !== params.scopedConversationId)
      .map((candidate) => candidate.seedId);
    if (mismatched.length > 0) {
      throw new Error(
        `Some seed IDs are outside conversation ${params.scopedConversationId}: ${mismatched.join(", ")}`,
      );
    }
    return params.scopedConversationId;
  }

  const conversationIds = Array.from(
    new Set(params.candidates.map((candidate) => candidate.conversationId)),
  );
  if (conversationIds.length === 1 && typeof conversationIds[0] === "number") {
    return conversationIds[0];
  }

  if (params.allConversations && conversationIds.length > 1) {
    throw new Error(
      "Query matched seeds from multiple conversations. Provide conversationId or narrow the query.",
    );
  }

  throw new Error(
    "Unable to resolve a single conversation scope. Provide conversationId or set a narrower summary scope.",
  );
}

/**
 * Resolve summary candidates from explicit IDs and/or query matches.
 */
async function resolveExpansionCandidates(params: {
  lcm: LcmContextEngine;
  explicitSummaryIds: string[];
  query?: string;
  conversationId?: number;
}): Promise<ExpansionCandidate[]> {
  const retrieval = params.lcm.getRetrieval();
  const candidates = new Map<string, ExpansionCandidate>();

  const addSummaryCandidate = (summaryId: string, conversationId: number): void => {
    candidates.set(`sum:${summaryId}`, {
      seedId: summaryId,
      seedType: "summary",
      conversationId,
      summaryIds: [summaryId],
    });
  };

  const addMarbleCandidate = (seed: MarbleSeedSnapshot, conversationId: number): void => {
    candidates.set(`mar:${seed.marbleId}`, {
      seedId: seed.marbleId,
      seedType: "marble",
      conversationId,
      summaryIds: seed.sourceSummaryIds,
      marble: seed,
    });
  };

  for (const summaryId of params.explicitSummaryIds) {
    const described = await retrieval.describe(summaryId);
    if (!described) {
      throw new Error(`Seed not found: ${summaryId}`);
    }
    if (described.type === "summary" && described.summary) {
      addSummaryCandidate(summaryId, described.summary.conversationId);
      continue;
    }
    if (described.type === "marble" && described.marble) {
      const marble = described.marble;
      addMarbleCandidate(
        {
          marbleId: marble.marbleId,
          marbleKind: marble.marbleKind,
          freshnessState: marble.freshnessState,
          provenanceRef: marble.provenanceRef,
          sourceFingerprint: marble.sourceFingerprint,
          sourceCount: marble.sourceCount,
          sourceArtifactTokenCount: marble.sourceArtifactTokenCount,
          sourceRefs: marble.sourceRefs,
          sourceSummaryIds: marble.sourceRefs.filter((ref) => ref.startsWith("sum_")),
        },
        marble.conversationId,
      );
      continue;
    }
    throw new Error(`Seed not expandable: ${summaryId}`);
  }

  if (params.query) {
    const grepResult = await retrieval.grep({
      query: params.query,
      mode: "full_text",
      scope: "both",
      conversationId: params.conversationId,
    });
    for (const summary of grepResult.summaries) {
      addSummaryCandidate(summary.summaryId, summary.conversationId);
    }
    for (const marble of grepResult.marbles) {
      const described = await retrieval.describe(marble.marbleId);
      if (!described || described.type !== "marble" || !described.marble) {
        continue;
      }
      addMarbleCandidate(
        {
          marbleId: described.marble.marbleId,
          marbleKind: described.marble.marbleKind,
          freshnessState: described.marble.freshnessState,
          provenanceRef: described.marble.provenanceRef,
          sourceFingerprint: described.marble.sourceFingerprint,
          sourceCount: described.marble.sourceCount,
          sourceArtifactTokenCount: described.marble.sourceArtifactTokenCount,
          sourceRefs: described.marble.sourceRefs,
          sourceSummaryIds: described.marble.sourceRefs.filter((ref) => ref.startsWith("sum_")),
        },
        described.marble.conversationId,
      );
    }
  }

  return Array.from(candidates.values());
}

export function createLcmExpandQueryTool(input: {
  deps: LcmDependencies;
  lcm: LcmContextEngine;
  /** Session id used for LCM conversation scoping. */
  sessionId?: string;
  /** Requester agent session key used for delegated child session/auth scoping. */
  requesterSessionKey?: string;
  /** Session key for scope fallback when sessionId is unavailable. */
  sessionKey?: string;
}): AnyAgentTool {
  return {
    name: "lcm_expand_query",
    label: "LCM Expand Query",
    description:
      "Answer a focused question using delegated LCM expansion. " +
      "Find candidate summaries or marbles (by IDs or query), expand them in a delegated sub-agent, " +
      "and return a compact prompt-focused answer with cited summary IDs.",
    parameters: LcmExpandQuerySchema,
    async execute(_toolCallId, params) {
      const p = params as Record<string, unknown>;
      const explicitSummaryIds = normalizeSummaryIds(p.summaryIds as string[] | undefined);
      const query = typeof p.query === "string" ? p.query.trim() : "";
      const prompt = typeof p.prompt === "string" ? p.prompt.trim() : "";
      const requestedMaxTokens =
        typeof p.maxTokens === "number" ? Math.trunc(p.maxTokens) : undefined;
      const maxTokens =
        typeof requestedMaxTokens === "number" && Number.isFinite(requestedMaxTokens)
          ? Math.max(1, requestedMaxTokens)
          : DEFAULT_MAX_ANSWER_TOKENS;
      const requestedTokenCap = typeof p.tokenCap === "number" ? Math.trunc(p.tokenCap) : undefined;
      const expansionTokenCap =
        typeof requestedTokenCap === "number" && Number.isFinite(requestedTokenCap)
          ? Math.max(1, requestedTokenCap)
          : Math.max(1, Math.trunc(input.deps.config.maxExpandTokens));

      if (!prompt) {
        return jsonResult({
          error: "prompt is required.",
        });
      }

      if (explicitSummaryIds.length === 0 && !query) {
        return jsonResult({
          error: "Either summaryIds or query must be provided.",
        });
      }

      const callerSessionKey =
        (typeof input.requesterSessionKey === "string"
          ? input.requesterSessionKey
          : input.sessionId
        )?.trim() ?? "";
      const requestId = resolveExpansionRequestId(callerSessionKey);
      const recursionCheck = evaluateExpansionRecursionGuard({
        sessionKey: callerSessionKey,
        requestId,
      });
      recordExpansionDelegationTelemetry({
        deps: input.deps,
        component: "lcm_expand_query",
        event: "start",
        requestId,
        sessionKey: callerSessionKey,
        expansionDepth: recursionCheck.expansionDepth,
        originSessionKey: recursionCheck.originSessionKey,
      });
      if (recursionCheck.blocked) {
        recordExpansionDelegationTelemetry({
          deps: input.deps,
          component: "lcm_expand_query",
          event: "block",
          requestId,
          sessionKey: callerSessionKey,
          expansionDepth: recursionCheck.expansionDepth,
          originSessionKey: recursionCheck.originSessionKey,
          reason: recursionCheck.reason,
        });
        return jsonResult({
          errorCode: recursionCheck.code,
          error: recursionCheck.message,
          requestId: recursionCheck.requestId,
          expansionDepth: recursionCheck.expansionDepth,
          originSessionKey: recursionCheck.originSessionKey,
          reason: recursionCheck.reason,
        });
      }

      const conversationScope = await resolveLcmConversationScope({
        lcm: input.lcm,
        deps: input.deps,
        sessionId: input.sessionId,
        sessionKey: input.sessionKey,
        params: p,
      });
      let scopedConversationId = conversationScope.conversationId;
      if (
        !conversationScope.allConversations &&
        scopedConversationId == null &&
        callerSessionKey
      ) {
        scopedConversationId = await resolveRequesterConversationScopeId({
          deps: input.deps,
          requesterSessionKey: callerSessionKey,
          lcm: input.lcm,
        });
      }

      if (!conversationScope.allConversations && scopedConversationId == null) {
        return jsonResult({
          error:
            "No LCM conversation found for this session. Provide conversationId or set allConversations=true.",
        });
      }

      let childSessionKey = "";
      let grantCreated = false;

      try {
        const candidates = await resolveExpansionCandidates({
          lcm: input.lcm,
          explicitSummaryIds,
          query: query || undefined,
          conversationId: scopedConversationId,
        });

        if (candidates.length === 0) {
          if (typeof scopedConversationId !== "number") {
            return jsonResult({
              error: "No matching summaries or marbles found.",
            });
          }
          return jsonResult({
            answer: "No matching summaries or marbles found for this scope.",
            citedIds: [],
            sourceConversationId: scopedConversationId,
            expandedSummaryCount: 0,
            totalSourceTokens: 0,
            truncated: false,
          });
        }

        const sourceConversationId = resolveSourceConversationId({
          scopedConversationId,
          allConversations: conversationScope.allConversations,
          candidates,
        });
        const summaryIds = normalizeSummaryIds(
          candidates
            .filter((candidate) => candidate.conversationId === sourceConversationId)
            .flatMap((candidate) => candidate.summaryIds),
        );
        const marbleSeeds = candidates
          .filter((candidate) => candidate.conversationId === sourceConversationId)
          .flatMap((candidate) => (candidate.marble ? [candidate.marble] : []));

        const requesterAgentId = input.deps.normalizeAgentId(
          input.deps.parseAgentSessionKey(callerSessionKey)?.agentId,
        );
        childSessionKey = `agent:${requesterAgentId}:subagent:${crypto.randomUUID()}`;
        const childExpansionDepth = resolveNextExpansionDepth(callerSessionKey);
        const originSessionKey = recursionCheck.originSessionKey || callerSessionKey || "main";

        createDelegatedExpansionGrant({
          delegatedSessionKey: childSessionKey,
          issuerSessionId: callerSessionKey || "main",
          allowedConversationIds: [sourceConversationId],
          tokenCap: expansionTokenCap,
          ttlMs: DELEGATED_WAIT_TIMEOUT_MS + 30_000,
        });
        stampDelegatedExpansionContext({
          sessionKey: childSessionKey,
          requestId,
          expansionDepth: childExpansionDepth,
          originSessionKey,
          stampedBy: "lcm_expand_query",
        });
        grantCreated = true;

        const task = buildDelegatedExpandQueryTask({
          summaryIds,
          marbleSeeds,
          conversationId: sourceConversationId,
          query: query || undefined,
          prompt,
          maxTokens,
          tokenCap: expansionTokenCap,
          requestId,
          expansionDepth: childExpansionDepth,
          originSessionKey,
        });

        const childIdem = crypto.randomUUID();
        const response = (await input.deps.callGateway({
          method: "agent",
          params: {
            message: task,
            sessionKey: childSessionKey,
            deliver: false,
            lane: input.deps.agentLaneSubagent,
            idempotencyKey: childIdem,
            extraSystemPrompt: input.deps.buildSubagentSystemPrompt({
              depth: 1,
              maxDepth: 8,
              taskSummary: "Run lcm_expand and return prompt-focused JSON answer",
            }),
          },
          timeoutMs: GATEWAY_TIMEOUT_MS,
        })) as { runId?: string };

        const runId = typeof response?.runId === "string" ? response.runId.trim() : "";
        if (!runId) {
          return jsonResult({
            error: "Delegated expansion did not return a runId.",
          });
        }

        const wait = (await input.deps.callGateway({
          method: "agent.wait",
          params: {
            runId,
            timeoutMs: DELEGATED_WAIT_TIMEOUT_MS,
          },
          timeoutMs: DELEGATED_WAIT_TIMEOUT_MS,
        })) as { status?: string; error?: string };
        const status = typeof wait?.status === "string" ? wait.status : "error";
        if (status === "timeout") {
          recordExpansionDelegationTelemetry({
            deps: input.deps,
            component: "lcm_expand_query",
            event: "timeout",
            requestId,
            sessionKey: callerSessionKey,
            expansionDepth: childExpansionDepth,
            originSessionKey,
            runId,
          });
          return jsonResult({
            error: "lcm_expand_query timed out waiting for delegated expansion (120s).",
          });
        }
        if (status !== "ok") {
          return jsonResult({
            error:
              typeof wait?.error === "string" && wait.error.trim()
                ? wait.error
                : "Delegated expansion query failed.",
          });
        }

        const replyPayload = (await input.deps.callGateway({
          method: "sessions.get",
          params: { key: childSessionKey, limit: 80 },
          timeoutMs: GATEWAY_TIMEOUT_MS,
        })) as { messages?: unknown[] };
        const reply = input.deps.readLatestAssistantReply(
          Array.isArray(replyPayload.messages) ? replyPayload.messages : [],
        );
        const parsed = parseDelegatedExpandQueryReply(reply, summaryIds.length);
        recordExpansionDelegationTelemetry({
          deps: input.deps,
          component: "lcm_expand_query",
          event: "success",
          requestId,
          sessionKey: callerSessionKey,
          expansionDepth: childExpansionDepth,
          originSessionKey,
          runId,
        });

        return jsonResult({
          answer: parsed.answer,
          citedIds: parsed.citedIds,
          sourceConversationId,
          expandedSummaryCount: parsed.expandedSummaryCount,
          totalSourceTokens: parsed.totalSourceTokens,
          truncated: parsed.truncated,
        });
      } catch (error) {
        return jsonResult({
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        if (childSessionKey) {
          try {
            await input.deps.callGateway({
              method: "sessions.delete",
              params: { key: childSessionKey, deleteTranscript: true },
              timeoutMs: GATEWAY_TIMEOUT_MS,
            });
          } catch {
            // Cleanup is best-effort.
          }
        }
        if (grantCreated && childSessionKey) {
          revokeDelegatedExpansionGrantForSession(childSessionKey, { removeBinding: true });
        }
        if (childSessionKey) {
          clearDelegatedExpansionContext(childSessionKey);
        }
      }
    },
  };
}
