import { mkdirSync, rmSync } from "node:fs";
import path from "node:path";

import {
  buildNormalizedEventExport,
  createExplicitEventRange,
  CONTRACT_IDS,
  type ArtifactManifestV1,
  type FeedbackEventV1,
  type InteractionEventV1,
  type NormalizedEventExportV1,
  type NormalizedEventV1,
  type PackContextBlockRecordV1,
  type PackGraphPayloadV1,
  type PackVectorEntryV1,
  type PackVectorsPayloadV1,
  type RouterArtifactV1,
  validateNormalizedEventExport
} from "@openclawbrain/contracts";
import { computePayloadChecksum, loadPack, PACK_LAYOUT, type PackDescriptor, writePackFile } from "@openclawbrain/pack-format";

export interface CandidatePackEventExports {
  interactionEvents: InteractionEventV1[];
  feedbackEvents: FeedbackEventV1[];
}

export interface CandidatePackBuildInput {
  packLabel: string;
  workspaceSnapshot: string;
  eventRange: {
    start: number;
    end: number;
  };
  eventExports?: CandidatePackEventExports;
  learnedRouting: boolean;
  builtAt?: string;
  offlineArtifacts?: string[];
  structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
}

export interface CandidatePackFromNormalizedEventExportInput {
  packLabel: string;
  workspaceSnapshot: string;
  normalizedEventExport: NormalizedEventExportV1;
  learnedRouting: boolean;
  builtAt?: string;
  offlineArtifacts?: string[];
  structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
}

export interface CandidatePackPayloads {
  graph: PackGraphPayloadV1;
  vectors: PackVectorsPayloadV1;
  router: RouterArtifactV1 | null;
}

export interface CandidatePackBuildResult {
  manifest: ArtifactManifestV1;
  payloads: CandidatePackPayloads;
  summary: {
    packId: string;
    immutable: true;
    routePolicy: ArtifactManifestV1["routePolicy"];
    eventRange: ArtifactManifestV1["provenance"]["eventRange"];
    eventExportDigest: string | null;
  };
}

function stableHash(value: string): string {
  let hash = 0;
  for (const char of value) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function structuralOpsSummary(input: CandidatePackBuildInput): Required<ArtifactManifestV1["graphDynamics"]["structuralOps"]> {
  return {
    split: input.structuralOps?.split ?? 0,
    merge: input.structuralOps?.merge ?? 0,
    prune: input.structuralOps?.prune ?? 0,
    connect: input.structuralOps?.connect ?? 0
  };
}

function keywordTokens(value: string): string[] {
  return [...new Set(value.toLowerCase().split(/[^a-z0-9]+/u).filter((token) => token.length >= 3 && /[a-z]/u.test(token)))].slice(0, 12);
}

function eventPriority(event: NormalizedEventV1): number {
  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    switch (event.kind) {
      case "correction":
      case "teaching":
        return 5;
      case "approval":
      case "suppression":
        return 4;
      default:
        return 4;
    }
  }

  switch (event.kind) {
    case "operator_override":
      return 5;
    case "memory_compiled":
      return 4;
    case "message_delivered":
      return 3;
    default:
      return 3;
  }
}

function sentenceCase(value: string): string {
  return value.length === 0 ? value : `${value[0]?.toUpperCase() ?? ""}${value.slice(1)}`;
}

function summarizeEvent(event: NormalizedEventV1): string {
  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    const relation = event.relatedInteractionId === undefined ? "" : ` Related interaction: ${event.relatedInteractionId}.`;
    return `${sentenceCase(event.kind)} feedback on ${event.channel} session ${event.sessionId}: ${event.content}.${relation}`;
  }

  const messagePart = event.messageId === undefined ? "" : ` Message: ${event.messageId}.`;
  const packPart = event.packId === undefined ? "" : ` Pack: ${event.packId}.`;
  return `Interaction ${event.kind} on ${event.channel} session ${event.sessionId}.${packPart}${messagePart}`;
}

function eventBlock(packId: string, event: NormalizedEventV1): PackContextBlockRecordV1 {
  const text = summarizeEvent(event);
  return {
    id: `${packId}:event:${event.eventId}`,
    source: `${event.source.stream}:${event.kind}`,
    text,
    keywords: keywordTokens(`${event.kind} ${event.channel} ${event.source.stream} ${text}`),
    priority: eventPriority(event)
  };
}

function staticLifecycleBlocks(packId: string, input: CandidatePackBuildInput): PackContextBlockRecordV1[] {
  const structuralOps = structuralOpsSummary(input);

  return [
    {
      id: `${packId}:feedback-scanner`,
      source: "memory/2026-03-05-openclawbrain-vnext-roadmap.md",
      text: "Unified feedback scanner is validated on local session logs with Ollama qwen3.5:9b-q4_K_M, checkpointed resumes, and deduplicated feedback events.",
      keywords: ["feedback", "scanner", "local", "session", "logs", "ollama", "qwen", "checkpoint", "dedup"],
      priority: 5
    },
    {
      id: `${packId}:runtime-compile`,
      source: "docs/openclawbrain-openclaw-rearchitecture-execution-plan.md",
      text: "runtime_compile.v1 gives OpenClaw a narrow compile boundary over promoted immutable packs and manifest-gated learned routing.",
      keywords: ["runtime", "compile", "contract", "pack", "manifest", "learned", "routing", "openclaw"],
      priority: 4
    },
    {
      id: `${packId}:structural-ops`,
      source: "docs/openclawbrain-openclaw-rearchitecture-plan.md",
      text: `Structural graph ops remain first-class: split=${structuralOps.split}, merge=${structuralOps.merge}, prune=${structuralOps.prune}, connect=${structuralOps.connect}.`,
      keywords: ["structural", "split", "merge", "prune", "connect", "graph", "memory"],
      priority: 3
    }
  ];
}

function eventExportBlocks(packId: string, eventExport: NormalizedEventExportV1): PackContextBlockRecordV1[] {
  const allEvents = [...eventExport.interactionEvents, ...eventExport.feedbackEvents];
  const summaryKeywords = keywordTokens(
    `normalized event export ${eventExport.provenance.sourceStreams.join(" ")} interaction ${eventExport.provenance.interactionCount} feedback ${eventExport.provenance.feedbackCount}`
  );

  return [
    {
      id: `${packId}:event-export`,
      source: `contracts/${CONTRACT_IDS.interactionEvents}+${CONTRACT_IDS.feedbackEvents}`,
      text: `Normalized event export covers ${eventExport.provenance.interactionCount} interaction events and ${eventExport.provenance.feedbackCount} feedback events across sequences ${eventExport.range.start}-${eventExport.range.end}.`,
      keywords: summaryKeywords,
      priority: 5
    },
    ...allEvents.map((event) => eventBlock(packId, event))
  ];
}

function createGraphPayload(packId: string, input: CandidatePackBuildInput, eventExport: NormalizedEventExportV1 | null): PackGraphPayloadV1 {
  return {
    packId,
    blocks: [...staticLifecycleBlocks(packId, input), ...(eventExport === null ? [] : eventExportBlocks(packId, eventExport))]
  };
}

function vectorEntryFromBlock(block: PackContextBlockRecordV1): PackVectorEntryV1 {
  const weights = Object.fromEntries(
    block.keywords.map((keyword, index) => [keyword, Math.max(1, block.priority - Math.min(index, block.priority - 1))])
  );

  return {
    blockId: block.id,
    keywords: [...block.keywords],
    boost: Math.max(1, Math.ceil(block.priority / 2)),
    weights
  };
}

function createVectorsPayload(graph: PackGraphPayloadV1): PackVectorsPayloadV1 {
  return {
    packId: graph.packId,
    entries: graph.blocks.map((block) => vectorEntryFromBlock(block))
  };
}

function createRouterArtifact(packId: string, builtAt: string): RouterArtifactV1 {
  return {
    routerIdentity: `${packId}:route_fn`,
    strategy: "keyword_overlap_v1",
    trainedAt: builtAt,
    requiresLearnedRouting: true
  };
}

function resolveEventExport(input: CandidatePackBuildInput): NormalizedEventExportV1 | null {
  if (input.eventExports === undefined) {
    return null;
  }

  const eventExport = buildNormalizedEventExport(input.eventExports);
  const validationErrors = validateNormalizedEventExport(eventExport);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
  }

  if (eventExport.range.start !== input.eventRange.start || eventExport.range.end !== input.eventRange.end) {
    throw new Error(
      `event export range ${eventExport.range.start}-${eventExport.range.end} does not match requested range ${input.eventRange.start}-${input.eventRange.end}`
    );
  }

  return eventExport;
}

export function buildCandidatePack(input: CandidatePackBuildInput): CandidatePackBuildResult {
  const builtAt = input.builtAt ?? "2026-03-06T00:00:00.000Z";
  const routePolicy = input.learnedRouting ? "requires_learned_routing" : "heuristic_allowed";
  const eventExport = resolveEventExport(input);
  const eventRange = eventExport?.range ?? createExplicitEventRange(input.eventRange);
  const seed = JSON.stringify({
    packLabel: input.packLabel,
    workspaceSnapshot: input.workspaceSnapshot,
    eventRange,
    learnedRouting: input.learnedRouting,
    offlineArtifacts: input.offlineArtifacts ?? [],
    eventExportDigest: eventExport?.provenance.exportDigest ?? null
  });
  const packId = `pack-${stableHash(seed)}`;

  const graph = createGraphPayload(packId, input, eventExport);
  const vectors = createVectorsPayload(graph);
  const router = input.learnedRouting ? createRouterArtifact(packId, builtAt) : null;

  const payloads: CandidatePackPayloads = {
    graph,
    vectors,
    router
  };

  const manifest: ArtifactManifestV1 = {
    contract: CONTRACT_IDS.artifactManifest,
    packId,
    immutable: true,
    routePolicy,
    runtimeAssets: {
      graphPath: PACK_LAYOUT.graph,
      vectorPath: PACK_LAYOUT.vectors,
      router: input.learnedRouting
        ? {
            kind: "artifact",
            identity: payloads.router?.routerIdentity ?? null,
            artifactPath: PACK_LAYOUT.router
          }
        : {
            kind: "none",
            identity: null,
            artifactPath: null
          }
    },
    payloadChecksums: {
      graph: computePayloadChecksum(payloads.graph),
      vector: computePayloadChecksum(payloads.vectors),
      router: payloads.router === null ? null : computePayloadChecksum(payloads.router)
    },
    modelFingerprints: input.learnedRouting
      ? ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", payloads.router?.routerIdentity ?? "router:missing"]
      : ["BAAI/bge-large-en-v1.5"],
    provenance: {
      workspaceSnapshot: input.workspaceSnapshot,
      eventRange,
      eventExports: eventExport?.provenance ?? null,
      builtAt,
      offlineArtifacts: input.offlineArtifacts ?? []
    },
    graphDynamics: {
      hebbian: {
        enabled: true,
        learningRate: 0.1
      },
      decay: {
        enabled: true,
        halfLifeDays: 30
      },
      structuralOps: structuralOpsSummary(input)
    }
  };

  return {
    manifest,
    payloads,
    summary: {
      packId,
      immutable: true,
      routePolicy,
      eventRange,
      eventExportDigest: eventExport?.provenance.exportDigest ?? null
    }
  };
}

export function buildCandidatePackFromNormalizedEventExport(
  input: CandidatePackFromNormalizedEventExportInput
): CandidatePackBuildResult {
  const validationErrors = validateNormalizedEventExport(input.normalizedEventExport);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
  }

  const candidateInput: CandidatePackBuildInput = {
    packLabel: input.packLabel,
    workspaceSnapshot: input.workspaceSnapshot,
    eventRange: {
      start: input.normalizedEventExport.range.start,
      end: input.normalizedEventExport.range.end
    },
    eventExports: {
      interactionEvents: [...input.normalizedEventExport.interactionEvents],
      feedbackEvents: [...input.normalizedEventExport.feedbackEvents]
    },
    learnedRouting: input.learnedRouting,
    ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
    ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
    ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {})
  };

  return buildCandidatePack(candidateInput);
}

export function materializeCandidatePack(rootDir: string, input: CandidatePackBuildInput): PackDescriptor {
  const result = buildCandidatePack(input);
  rmSync(rootDir, { recursive: true, force: true });
  mkdirSync(rootDir, { recursive: true });

  writePackFile(rootDir, PACK_LAYOUT.graph, result.payloads.graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, result.payloads.vectors);
  if (result.payloads.router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, result.payloads.router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, result.manifest);

  return loadPack(path.resolve(rootDir));
}

export function materializeCandidatePackFromNormalizedEventExport(
  rootDir: string,
  input: CandidatePackFromNormalizedEventExportInput
): PackDescriptor {
  const result = buildCandidatePackFromNormalizedEventExport(input);
  rmSync(rootDir, { recursive: true, force: true });
  mkdirSync(rootDir, { recursive: true });

  writePackFile(rootDir, PACK_LAYOUT.graph, result.payloads.graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, result.payloads.vectors);
  if (result.payloads.router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, result.payloads.router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, result.manifest);

  return loadPack(path.resolve(rootDir));
}
