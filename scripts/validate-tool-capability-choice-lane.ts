#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { Value } from "@sinclair/typebox/value";
import { materializeRouteDecisionRowsFromTraceV1, RouteDecisionRowSchemaV1, validateRouteDecisionRowV1, type RouteDecisionRowV1 } from "../src/brain-core/route-rows.ts";
import { recordTrace } from "../src/brain-core/trace.ts";
import type { BrainNode, SeedScore, TrajectoryExpansion } from "../src/brain-core/types.ts";
import type { TraverseResult } from "../src/brain-core/traverse.ts";

export const WEATHER_CAPABILITY_FAMILY = "weather.current_conditions";
export const WEATHER_CAPABILITY_NODE_ID = "toolcap_weather_current_conditions";
export const WEATHER_INSTANCE_NODE_ID = "toolinst_weather_provider_local";
export const WEATHER_STATIC_DOC_NODE_ID = "doc_weather_chance_of_rain";
export const WEATHER_INTENT_NODE_ID = "intent_weather_freshness";

export type WeatherCapabilityCaseId = "must_fire_current_weather" | "must_not_fire_weather_definition";

export interface WeatherCapabilityLaneResult {
  family: string;
  generatedAt: string;
  pass: boolean;
  must_fire_pass: boolean;
  must_not_fire_pass: boolean;
  rows: Array<{
    case_id: WeatherCapabilityCaseId;
    prompt: string;
    chosen_action_kind: RouteDecisionRowV1["chosen_action_kind"];
    chosen_tool_capability_id: string | null;
    chosen_tool_instance_id: string | null;
    stop_label: RouteDecisionRowV1["stop_label"];
    hard_negatives: string[];
    local_action_kinds: string[];
    pass: boolean;
  }>;
}

function promptForCaseId(caseId: WeatherCapabilityCaseId): string {
  return caseId === "must_fire_current_weather"
    ? "Is it going to rain in San Francisco this afternoon? Give the current answer."
    : "In one sentence, explain what chance of rain means.";
}

function makeNode(id: string, content = `node ${id}`): BrainNode {
  return {
    id,
    kind: "chunk",
    content,
    embedding: new Float32Array([1, 0, 0]),
    sourceUri: "tool-capability-choice-weather-v1.fixture.md",
    trust: "scanner",
    tags: [],
    tokenCount: 12,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeToolNode(id: string, role: "capability" | "instance"): BrainNode {
  return {
    ...makeNode(id, role === "capability" ? "current weather/rain lookup capability" : "local weather provider instance"),
    kind: "toolcard",
    metadata: {
      toolName: role === "capability" ? WEATHER_CAPABILITY_FAMILY : "weather.provider.local",
      toolArgsShape: "location,date_or_time",
      toolRole: role,
      ...(role === "instance" ? { toolCapabilityId: WEATHER_CAPABILITY_FAMILY } : {}),
    },
  };
}

function makeSeedScore(nodeId: string, selected: boolean, selectionSubstepIndex: number | null): SeedScore {
  return {
    nodeId,
    priorScore: 1,
    learnedSeedWeight: 0,
    initialPolicyScore: 1,
    initialProbability: selected ? 0.7 : 0.1,
    latestPolicyScore: 1,
    latestProbability: selected ? 0.7 : 0.1,
    selected,
    selectionSubstepIndex,
  };
}

function makeStateSnapshot(sourceNodeId: string, expansionIndex: number, selectionIndex: number) {
  return {
    sourceNodeId,
    expansionIndex,
    selectionIndex,
    budgetRemaining: 100,
    initialBudget: 100,
    reservedTokenCost: 0,
    maxHops: 3,
    frontierSize: 3,
    frontierNodeIds: [WEATHER_STATIC_DOC_NODE_ID, WEATHER_CAPABILITY_NODE_ID, WEATHER_INSTANCE_NODE_ID],
    visitedCount: 1,
    firedCount: 1,
    maxFrontierSize: 4,
  };
}

function lookupWeatherNode(nodeId: string): BrainNode | null {
  if (nodeId === WEATHER_INTENT_NODE_ID) {
    return makeNode(nodeId, "fresh/location/time weather intent");
  }
  if (nodeId === WEATHER_STATIC_DOC_NODE_ID) {
    return makeNode(nodeId, "static definition of chance of rain");
  }
  if (nodeId === WEATHER_CAPABILITY_NODE_ID) {
    return makeToolNode(nodeId, "capability");
  }
  if (nodeId === WEATHER_INSTANCE_NODE_ID) {
    return makeToolNode(nodeId, "instance");
  }
  return null;
}

function makeTrace(params: {
  caseId: WeatherCapabilityCaseId;
  prompt: string;
  chosenAction: { type: "traverse"; targetNodeId: string } | { type: "stop_local" };
  chosenActionProbability: number;
  stopProbability: number;
  selectedTargets: string[];
}) {
  const sourceNodeId = WEATHER_INTENT_NODE_ID;
  const trajectory: TrajectoryExpansion[] = [
    {
      sourceNodeId,
      expansionIndex: 0,
      frontierBefore: [sourceNodeId],
      frontierAfter: [],
      budgetBefore: 100,
      budgetAfter: 90,
      substeps: [
        {
          stateSnapshot: makeStateSnapshot(sourceNodeId, 0, 0),
          candidates: [
            {
              action: { type: "traverse" as const, targetNodeId: WEATHER_STATIC_DOC_NODE_ID },
              score: params.caseId === "must_not_fire_weather_definition" ? 2.2 : 0.55,
              probability: params.caseId === "must_not_fire_weather_definition" ? 0.36 : 0.16,
              scoreBreakdown: { totalScore: params.caseId === "must_not_fire_weather_definition" ? 2.2 : 0.55, seedPrior: 0.1 },
            },
            {
              action: { type: "traverse" as const, targetNodeId: WEATHER_CAPABILITY_NODE_ID },
              score: params.caseId === "must_fire_current_weather" ? 3.2 : 0.2,
              probability: params.caseId === "must_fire_current_weather" ? 0.52 : 0.08,
              scoreBreakdown: { totalScore: params.caseId === "must_fire_current_weather" ? 3.2 : 0.2, toolActionPrior: params.caseId === "must_fire_current_weather" ? 1.8 : -0.4 },
            },
            {
              action: { type: "traverse" as const, targetNodeId: WEATHER_INSTANCE_NODE_ID },
              score: 0.3,
              probability: 0.07,
              scoreBreakdown: { totalScore: 0.3, toolActionPrior: -0.2 },
            },
            {
              action: { type: "stop_local" as const },
              score: params.caseId === "must_not_fire_weather_definition" ? 2.8 : 0.1,
              probability: params.caseId === "must_not_fire_weather_definition" ? 0.49 : 0.25,
              scoreBreakdown: { totalScore: params.caseId === "must_not_fire_weather_definition" ? 2.8 : 0.1, learnedStopWeight: params.caseId === "must_not_fire_weather_definition" ? 1.4 : -0.2 },
            },
          ],
          chosenAction: params.chosenAction,
          chosenActionProbability: params.chosenActionProbability,
          stopProbability: params.stopProbability,
        },
      ],
      selectedTargets: params.selectedTargets,
      acceptedTargets: params.selectedTargets,
      vetoedTargets: [],
      proposalOutcomes: params.selectedTargets.map((targetNodeId) => ({ targetNodeId, outcome: "accepted" as const, reason: "accepted" })),
      terminationReason: "policy_stop",
    },
  ];

  const traversalResult: TraverseResult = {
    firedNodes: params.selectedTargets.map((nodeId) => ({
      nodeId,
      kind: lookupWeatherNode(nodeId)?.kind ?? "chunk",
      content: lookupWeatherNode(nodeId)?.content ?? nodeId,
      tokenCount: 12,
    })),
    vetoedNodes: [],
    trajectory,
    seedScores: [makeSeedScore(sourceNodeId, true, 0)],
    contextChars: params.selectedTargets.length * 12,
    footer: "Brain · weather capability-choice fixture",
    interruption: null,
  };

  return recordTrace({
    traversalResult,
    queryText: params.prompt,
    episodeId: `ep_${params.caseId}`,
    conversationId: params.caseId === "must_fire_current_weather" ? 301 : 302,
    packVersion: 1,
    budgetChars: 100,
    maxHops: 3,
    maxFanoutPerNode: 4,
    maxFrontierSize: 4,
    embeddingMs: 1,
    routeSelectionMs: 2,
    totalQueryMs: 3,
    queryEmbeddingSource: "provided",
    selectedNodes: params.selectedTargets.map((nodeId) => lookupWeatherNode(nodeId)).filter((node): node is BrainNode => node !== null),
    lookupNode: lookupWeatherNode,
    persistRawSurfaces: false,
  });
}

export function buildWeatherCapabilityChoiceRows(): RouteDecisionRowV1[] {
  const traces = [
    makeTrace({
      caseId: "must_fire_current_weather",
      prompt: "Is it going to rain in San Francisco this afternoon? Give the current answer.",
      chosenAction: { type: "traverse", targetNodeId: WEATHER_CAPABILITY_NODE_ID },
      chosenActionProbability: 0.52,
      stopProbability: 0.25,
      selectedTargets: [WEATHER_CAPABILITY_NODE_ID],
    }),
    makeTrace({
      caseId: "must_not_fire_weather_definition",
      prompt: "In one sentence, explain what chance of rain means.",
      chosenAction: { type: "stop_local" },
      chosenActionProbability: 0.49,
      stopProbability: 0.49,
      selectedTargets: [],
    }),
  ];

  return traces.flatMap((trace) => materializeRouteDecisionRowsFromTraceV1({ trace, routeFnVersion: "tool-capability-choice-weather-v1" }));
}

export function validateWeatherCapabilityChoiceLane(rows = buildWeatherCapabilityChoiceRows()): WeatherCapabilityLaneResult {
  if (rows.length !== 2) {
    throw new Error(`expected exactly 2 route rows, got ${rows.length}`);
  }

  for (const row of rows) {
    if (!Value.Check(RouteDecisionRowSchemaV1, row)) {
      throw new Error(`invalid route row schema at index ${rows.indexOf(row)}`);
    }
    const validation = validateRouteDecisionRowV1(row);
    if (!validation.valid) {
      throw new Error(`invalid route row ${row.row_id}: ${validation.issues.join("; ")}`);
    }
  }

  const mustFire = rows.find((row) => row.episode_id === "ep_must_fire_current_weather");
  const mustNotFire = rows.find((row) => row.episode_id === "ep_must_not_fire_weather_definition");
  if (!mustFire || !mustNotFire) {
    throw new Error("missing weather capability proof rows");
  }

  const mustFirePass = mustFire.chosen_action_kind === "tool_capability"
    && mustFire.chosen_tool_capability_id === WEATHER_CAPABILITY_FAMILY
    && mustFire.chosen_tool_instance_id === null
    && mustFire.stop_label === "CONTINUE"
    && mustFire.local_action_set.some((candidate) => candidate.action_kind === "tool_instance" && candidate.node_id === WEATHER_INSTANCE_NODE_ID)
    && mustFire.hard_negatives.includes(WEATHER_INSTANCE_NODE_ID)
    && mustFire.hard_negatives.includes(WEATHER_STATIC_DOC_NODE_ID);

  const mustNotFirePass = mustNotFire.chosen_action_kind === "stop_local"
    && mustNotFire.chosen_tool_capability_id === null
    && mustNotFire.chosen_tool_instance_id === null
    && mustNotFire.stop_label === "STOP_LOCAL"
    && mustNotFire.local_action_set.some((candidate) => candidate.tool_capability_id === WEATHER_CAPABILITY_FAMILY)
    && mustNotFire.local_action_set.some((candidate) => candidate.action_kind === "tool_instance")
    && mustNotFire.hard_negatives.includes(WEATHER_CAPABILITY_NODE_ID)
    && mustNotFire.hard_negatives.includes(WEATHER_INSTANCE_NODE_ID);

  const summarize = (row: RouteDecisionRowV1): WeatherCapabilityLaneResult["rows"][number] => {
    const caseId: WeatherCapabilityCaseId = row.episode_id === "ep_must_fire_current_weather" ? "must_fire_current_weather" : "must_not_fire_weather_definition";
    return {
    case_id: caseId,
    prompt: promptForCaseId(caseId),
    chosen_action_kind: row.chosen_action_kind,
    chosen_tool_capability_id: row.chosen_tool_capability_id,
    chosen_tool_instance_id: row.chosen_tool_instance_id,
    stop_label: row.stop_label,
    hard_negatives: row.hard_negatives,
    local_action_kinds: row.local_action_set.map((candidate) => candidate.action_kind),
    pass: row === mustFire ? mustFirePass : mustNotFirePass,
  };
  };

  return {
    family: WEATHER_CAPABILITY_FAMILY,
    generatedAt: new Date().toISOString(),
    pass: mustFirePass && mustNotFirePass,
    must_fire_pass: mustFirePass,
    must_not_fire_pass: mustNotFirePass,
    rows: [summarize(mustFire), summarize(mustNotFire)],
  };
}

function parseArgs(argv: string[]): { family: string; outputDir: string } {
  let family = WEATHER_CAPABILITY_FAMILY;
  let outputDir = path.resolve("artifacts", "tool-capability-choice-weather-v1");
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--family") {
      family = argv[++i] ?? "";
    } else if (arg === "--output-dir") {
      outputDir = argv[++i] ?? "";
    }
  }
  if (family !== WEATHER_CAPABILITY_FAMILY) {
    throw new Error(`unsupported family ${family}; expected ${WEATHER_CAPABILITY_FAMILY}`);
  }
  if (!outputDir) {
    throw new Error("--output-dir is required");
  }
  return { family, outputDir: path.resolve(outputDir) };
}

function writeArtifacts(outputDir: string, rows: RouteDecisionRowV1[], summary: WeatherCapabilityLaneResult): void {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(path.join(outputDir, "rows.jsonl"), `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { outputDir } = parseArgs(process.argv.slice(2));
  const rows = buildWeatherCapabilityChoiceRows();
  const summary = validateWeatherCapabilityChoiceLane(rows);
  writeArtifacts(outputDir, rows, summary);
  if (!summary.pass) {
    console.error(JSON.stringify(summary, null, 2));
    process.exit(1);
  }
  console.log(`tool capability choice proof: ok (${WEATHER_CAPABILITY_FAMILY})`);
  console.log(`summary: ${path.join(outputDir, "summary.json")}`);
  console.log(`rows: ${path.join(outputDir, "rows.jsonl")}`);
}
