#!/usr/bin/env node

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

export const LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT = "learned_route_eval_tranche_manifest.v1" as const;
export const LEARNED_ROUTE_HARD_NEGATIVE_MINING_SPEC_CONTRACT = "learned_route_hard_negative_mining_spec.v1" as const;

const DEFAULT_SOURCE_TASK_ID = "T-20260415-257";
const DEFAULT_OUTPUT_TASK_ID = "T-20260419-267";
const DEFAULT_MUST_FIRE_MANIFEST = path.join(workspaceRoot, "task-artifacts", DEFAULT_SOURCE_TASK_ID, "must-fire-30.manifest.json");
const DEFAULT_MUST_NOT_FIRE_MANIFEST = path.join(workspaceRoot, "task-artifacts", DEFAULT_SOURCE_TASK_ID, "must-not-fire-100.manifest.json");
const DEFAULT_TRAP_MANIFEST = path.join(workspaceRoot, "task-artifacts", DEFAULT_SOURCE_TASK_ID, "vector-only-trap-50.manifest.json");
const DEFAULT_HARD_NEGATIVE_SPEC = path.join(workspaceRoot, "task-artifacts", DEFAULT_SOURCE_TASK_ID, "hard-negative-mining-spec.v1.json");
const DEFAULT_OUTPUT_DIR = path.join(workspaceRoot, "task-artifacts", DEFAULT_OUTPUT_TASK_ID, "binary-gate-v2-tranches");

export type MustFireSplitTrancheId =
  | "must_fire_exact_artifact"
  | "must_fire_resume_state"
  | "must_fire_recent_decision"
  | "must_fire_stale_summary_repair";

export type TrapSplitTrancheId =
  | "trap_wrapper_system"
  | "trap_operator_artifact"
  | "trap_user_visible_resume";

export type HardNegativeClass =
  | "unnecessary_activation"
  | "tie_with_cost"
  | "graph_prior_preferred"
  | "stale_memory"
  | "wrapper_heavy"
  | "no_outcome_change";

export interface TrancheAnchor {
  traceId: string;
  sourcePath: string;
  bucket?: string;
  whyIncluded?: string;
  preview?: string;
}

export interface TrancheManifest {
  contract: typeof LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT;
  trancheId: string;
  taskId?: string;
  builtAt?: string;
  status?: string;
  purpose?: string;
  targetTraceCount?: number;
  anchorTraceCount?: number;
  remainingToMine?: number;
  sourceManifests?: string[];
  selectionRules?: string[];
  notes?: string[];
  priorityBuckets?: string[];
  anchors: TrancheAnchor[];
  [key: string]: unknown;
}

export interface HardNegativeSpec {
  contract?: string;
  taskId?: string;
  builtAt?: string;
  status?: string;
  purpose?: string;
  primaryGoal?: string;
  missionTestCommand?: string;
  weightFloorsByClass?: Partial<Record<HardNegativeClass, number>>;
  bucketToHardNegativeClass?: Record<string, HardNegativeClass>;
  miningRules?: Array<Record<string, unknown>>;
  seedAssignments?: Record<string, unknown>;
  nextMiningPass?: string[];
  [key: string]: unknown;
}

interface ParsedArgs {
  sourceTaskId: string;
  outputTaskId: string;
  mustFireManifest: string;
  mustNotFireManifest: string;
  trapManifest: string;
  hardNegativeSpec: string;
  outputDir: string;
  generatedAt: string;
}

const MUST_FIRE_SPLIT_BY_TRACE_ID: Record<string, MustFireSplitTrancheId> = {
  "live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003": "must_fire_recent_decision",
  "live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002": "must_fire_exact_artifact",
  "live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002": "must_fire_stale_summary_repair",
  "live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002": "must_fire_recent_decision",
  "live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003": "must_fire_resume_state",
  "live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002": "must_fire_exact_artifact",
  "live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006": "must_fire_stale_summary_repair",
  "live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054": "must_fire_recent_decision",
  "live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002": "must_fire_resume_state",
  "live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002": "must_fire_exact_artifact",
};

const MUST_FIRE_PURPOSE_BY_TRANCHE: Record<MustFireSplitTrancheId, string> = {
  must_fire_exact_artifact:
    "Positive audit lane for exact artifact, proof, config, or repo-state references where activation should recover the concrete local object instead of answering from generic semantics.",
  must_fire_resume_state:
    "Positive audit lane for short follow-up and relaunch turns whose correct response depends on recent session continuity.",
  must_fire_recent_decision:
    "Positive audit lane for owner-direction, approval, and recent-decision recall where the active decision surface is not fully restated inline.",
  must_fire_stale_summary_repair:
    "Positive audit lane for runtime-truth and blocker-diagnosis repairs where stale summary state must be corrected with fresher local evidence.",
};

const TRAP_PURPOSE_BY_TRANCHE: Record<TrapSplitTrancheId, string> = {
  trap_wrapper_system:
    "Abstention audit lane for wrapper-heavy, system-relay, and automation-shell traces that should teach a strong veto before retrieval activates.",
  trap_operator_artifact:
    "Abstention audit lane for operator/artifact instruction turns that look memory-adjacent but are still self-contained enough to avoid retrieval.",
  trap_user_visible_resume:
    "Abstention audit lane for user-visible resume-style turns where the current binary gate must still prove restraint instead of collapsing to vector-only retrieval.",
};

const TRAP_BUCKET_TO_HARD_NEGATIVE_CLASS: Record<TrapSplitTrancheId, HardNegativeClass> = {
  trap_wrapper_system: "wrapper_heavy",
  trap_operator_artifact: "unnecessary_activation",
  trap_user_visible_resume: "graph_prior_preferred",
};

function usage(): void {
  process.stderr.write(
    [
      "Usage: node --experimental-transform-types scripts/build-binary-gate-v2-tranches.ts [options]",
      "",
      "Options:",
      `  --source-task-id <id>         Defaults to ${DEFAULT_SOURCE_TASK_ID}`,
      `  --output-task-id <id>         Defaults to ${DEFAULT_OUTPUT_TASK_ID}`,
      `  --must-fire-manifest <path>   Defaults to ${DEFAULT_MUST_FIRE_MANIFEST}`,
      `  --must-not-fire-manifest <path> Defaults to ${DEFAULT_MUST_NOT_FIRE_MANIFEST}`,
      `  --trap-manifest <path>        Defaults to ${DEFAULT_TRAP_MANIFEST}`,
      `  --hard-negative-spec <path>   Defaults to ${DEFAULT_HARD_NEGATIVE_SPEC}`,
      `  --output-dir <path>           Defaults to ${DEFAULT_OUTPUT_DIR}`,
      "  --generated-at <iso>          Override generated timestamp",
      "  --help                        Show this help",
    ].join("\n") + "\n",
  );
}

function normalizeCliString(value: string | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function parseArgs(argv: string[]): ParsedArgs {
  const parsed: ParsedArgs = {
    sourceTaskId: DEFAULT_SOURCE_TASK_ID,
    outputTaskId: DEFAULT_OUTPUT_TASK_ID,
    mustFireManifest: DEFAULT_MUST_FIRE_MANIFEST,
    mustNotFireManifest: DEFAULT_MUST_NOT_FIRE_MANIFEST,
    trapManifest: DEFAULT_TRAP_MANIFEST,
    hardNegativeSpec: DEFAULT_HARD_NEGATIVE_SPEC,
    outputDir: DEFAULT_OUTPUT_DIR,
    generatedAt: new Date().toISOString(),
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--source-task-id":
        parsed.sourceTaskId = normalizeCliString(argv[index + 1]) ?? parsed.sourceTaskId;
        index += 1;
        break;
      case "--output-task-id":
        parsed.outputTaskId = normalizeCliString(argv[index + 1]) ?? parsed.outputTaskId;
        index += 1;
        break;
      case "--must-fire-manifest":
        parsed.mustFireManifest = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--must-not-fire-manifest":
        parsed.mustNotFireManifest = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--trap-manifest":
        parsed.trapManifest = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--hard-negative-spec":
        parsed.hardNegativeSpec = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--output-dir":
        parsed.outputDir = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--generated-at":
        parsed.generatedAt = normalizeCliString(argv[index + 1]) ?? parsed.generatedAt;
        index += 1;
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function readJson<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function writeJson(filePath: string, value: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function assertTrancheManifest(value: TrancheManifest, filePath: string): TrancheManifest {
  if (value?.contract !== LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT) {
    throw new Error(`Expected ${LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT} at ${filePath}`);
  }
  if (!Array.isArray(value.anchors) || value.anchors.length === 0) {
    throw new Error(`Tranche manifest has no anchors: ${filePath}`);
  }
  return value;
}

function normalizeBucket(value: string | undefined): string | undefined {
  return value?.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function deriveTraceFamily(traceId: string): string | null {
  const match = traceId.match(/^live-([a-z]+)-/i);
  return match?.[1]?.toLowerCase() ?? null;
}

function canonicalizeAnchorSourcePath(anchor: TrancheAnchor): string {
  const originalPath = resolveWorkspacePath(anchor.sourcePath);
  if (existsSync(originalPath)) {
    return anchor.sourcePath;
  }

  const traceFamily = deriveTraceFamily(anchor.traceId);
  const fallbackCandidates = traceFamily === null
    ? []
    : [
      path.join(workspaceRoot, "task-artifacts", "T-20260415-250", "stratified-rich-live-535-extracted", "traces", traceFamily, `${anchor.traceId}.json`),
      path.join(workspaceRoot, "task-artifacts", "T-20260415-250", "semantic-rich-live-535-extracted", "traces", traceFamily, `${anchor.traceId}.json`),
      path.join(workspaceRoot, "task-artifacts", "T-20260415-250", "primary-mixed-live-1045-extracted", "traces", traceFamily, `${anchor.traceId}.json`),
      path.join(workspaceRoot, "task-artifacts", "T-20260415-250", "primary-mixed-live-1045-extracted-v2", "traces", traceFamily, `${anchor.traceId}.json`),
    ];
  const resolvedFallback = fallbackCandidates.find((candidatePath) => existsSync(candidatePath));
  if (!resolvedFallback) {
    return anchor.sourcePath;
  }
  return path.relative(workspaceRoot, resolvedFallback).split(path.sep).join("/");
}

function withUpdatedAnchor(anchor: TrancheAnchor, bucket: string): TrancheAnchor {
  return {
    ...anchor,
    sourcePath: canonicalizeAnchorSourcePath(anchor),
    bucket,
  };
}

export function splitMustFireAnchors(anchors: readonly TrancheAnchor[]): Record<MustFireSplitTrancheId, TrancheAnchor[]> {
  const split: Record<MustFireSplitTrancheId, TrancheAnchor[]> = {
    must_fire_exact_artifact: [],
    must_fire_resume_state: [],
    must_fire_recent_decision: [],
    must_fire_stale_summary_repair: [],
  };
  for (const anchor of anchors) {
    const trancheId = MUST_FIRE_SPLIT_BY_TRACE_ID[anchor.traceId];
    if (!trancheId) {
      throw new Error(`No must-fire v2 split mapping defined for trace ${anchor.traceId}`);
    }
    split[trancheId].push(withUpdatedAnchor(anchor, trancheId));
  }
  return split;
}

function includesAny(text: string, phrases: readonly string[]): boolean {
  return phrases.some((phrase) => text.includes(phrase));
}

interface RecordedSessionTraceLike {
  turns?: Array<{
    userMessage?: string;
  }>;
}

function resolveWorkspacePath(relativeOrAbsolutePath: string): string {
  return path.isAbsolute(relativeOrAbsolutePath)
    ? relativeOrAbsolutePath
    : path.resolve(workspaceRoot, relativeOrAbsolutePath);
}

function stripMetadataWrapper(text: string): string {
  return text
    .replace(/Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi, "")
    .replace(/Sender \(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi, "")
    .replace(/Replied message \(untrusted, for context\):\s*```json[\s\S]*?```\s*/gi, "")
    .replace(/```json[\s\S]*?```\s*/gi, "")
    .trim();
}

function extractTrapUserVisibleText(anchor: TrancheAnchor): string {
  try {
    const tracePath = resolveWorkspacePath(canonicalizeAnchorSourcePath(anchor));
    const trace = readJson<RecordedSessionTraceLike>(tracePath);
    const raw = trace.turns?.[0]?.userMessage;
    if (typeof raw !== "string") {
      return "";
    }
    return stripMetadataWrapper(raw);
  } catch {
    return "";
  }
}

export function classifyTrapAnchor(anchor: TrancheAnchor, userVisibleText?: string): TrapSplitTrancheId {
  const preview = (anchor.preview ?? anchor.whyIncluded ?? "").toLowerCase();
  const visible = (userVisibleText ?? "").toLowerCase();
  const normalizedBucket = normalizeBucket(anchor.bucket);
  const combined = [preview, visible].filter((value) => value.length > 0).join("\n");
  const wrapperPhrases = [
    "pre-compaction memory flush",
    "system:",
    "an async command you ran earlier has completed",
    "a cron job",
    "boot check",
    "exec denied",
    "heartbeat.md",
    "follow boot.md instructions exactly",
    "run the following periodic tasks",
    "store durable memories now",
    "conversation info (untrusted metadata)",
    "sender (untrusted metadata)",
    "replied message",
    "has_reply_context",
    "sessionid:",
  ] as const;
  const operatorPhrases = [
    "additional requirement from jon",
    "important contract clarification",
    "must treat explicit nonstandard openclaw homes",
    "docs/help/examples",
    "multi-horizon training lane",
    "three-surface model",
    "deliverables",
    "contract clarification",
    "rule change is specified directly",
    "explicit remember-this policy",
    "proof surface",
    "artifact",
  ] as const;

  if (normalizedBucket === "main_vector_collapse" || normalizedBucket === "pelican_vector_collapse" || normalizedBucket === "bountiful_vector_collapse") {
    if (visible.length > 0) {
      if (includesAny(visible, wrapperPhrases)) {
        return "trap_wrapper_system";
      }
      if (includesAny(visible, operatorPhrases)) {
        return "trap_operator_artifact";
      }
      return "trap_user_visible_resume";
    }
    if (includesAny(preview, wrapperPhrases)) {
      return "trap_wrapper_system";
    }
    if (includesAny(combined, operatorPhrases)) {
      return "trap_operator_artifact";
    }
  }

  return "trap_user_visible_resume";
}

export function splitTrapAnchors(anchors: readonly TrancheAnchor[]): Record<TrapSplitTrancheId, TrancheAnchor[]> {
  const split: Record<TrapSplitTrancheId, TrancheAnchor[]> = {
    trap_wrapper_system: [],
    trap_operator_artifact: [],
    trap_user_visible_resume: [],
  };
  for (const anchor of anchors) {
    const trancheId = classifyTrapAnchor(anchor, extractTrapUserVisibleText(anchor));
    split[trancheId].push(withUpdatedAnchor(anchor, trancheId));
  }
  return split;
}

function buildSplitManifest(params: {
  outputTaskId: string;
  trancheId: string;
  purpose: string;
  sourceManifest: TrancheManifest;
  anchors: TrancheAnchor[];
  notes: string[];
  selectionRules: string[];
  priorityBuckets?: string[];
  targetTraceCount?: number;
  generatedAt: string;
}): TrancheManifest {
  return {
    contract: LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT,
    trancheId: params.trancheId,
    taskId: params.outputTaskId,
    builtAt: params.generatedAt,
    status: "anchor_set_complete",
    purpose: params.purpose,
    targetTraceCount: params.targetTraceCount ?? params.anchors.length,
    anchorTraceCount: params.anchors.length,
    remainingToMine: Math.max((params.targetTraceCount ?? params.anchors.length) - params.anchors.length, 0),
    sourceManifests: uniqueStrings([
      ...((params.sourceManifest.sourceManifests ?? []) as string[]),
    ]),
    selectionRules: params.selectionRules,
    priorityBuckets: params.priorityBuckets,
    anchors: params.anchors,
    notes: params.notes,
  };
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const value of values) {
    if (typeof value !== "string") {
      continue;
    }
    const trimmed = value.trim();
    if (trimmed.length === 0 || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    ordered.push(trimmed);
  }
  return ordered;
}

function buildMergedPositiveManifest(params: {
  outputTaskId: string;
  sourceManifest: TrancheManifest;
  split: Record<MustFireSplitTrancheId, TrancheAnchor[]>;
  generatedAt: string;
}): TrancheManifest {
  const anchors = [
    ...params.split.must_fire_exact_artifact,
    ...params.split.must_fire_resume_state,
    ...params.split.must_fire_recent_decision,
    ...params.split.must_fire_stale_summary_repair,
  ];
  return buildSplitManifest({
    outputTaskId: params.outputTaskId,
    trancheId: "must_fire_binary_gate_v2",
    purpose:
      "Merged positive v2 audit lane for the binary activate-vs-abstain gate, composed only of exact-artifact, resume-state, recent-decision, and stale-summary-repair anchors.",
    sourceManifest: params.sourceManifest,
    anchors,
    targetTraceCount: anchors.length,
    selectionRules: [
      "Train the binary gate to activate only when exact recent-state recall should materially change the answer.",
      "Preserve per-trace split buckets so later audits can isolate the first true win clusters.",
    ],
    notes: [
      "This merged positive lane is derived from the narrower v2 split manifests and is intended for the first binary-gate training pass.",
      "Use the split per-bucket manifests for diagnosis; use this merged lane for the initial trainer input.",
    ],
    priorityBuckets: Object.keys(params.split),
    generatedAt: params.generatedAt,
  });
}

function buildMergedAbstentionManifest(params: {
  outputTaskId: string;
  mustNotFireManifest: TrancheManifest;
  trapManifest: TrancheManifest;
  trapSplit: Record<TrapSplitTrancheId, TrancheAnchor[]>;
  generatedAt: string;
}): TrancheManifest {
  const normalizedOriginalAnchors = params.mustNotFireManifest.anchors.map((anchor) => {
    const bucket = normalizeBucket(anchor.bucket) ?? "must_not_fire";
    return withUpdatedAnchor(anchor, bucket);
  });
  const trapAnchors = [
    ...params.trapSplit.trap_wrapper_system,
    ...params.trapSplit.trap_operator_artifact,
    ...params.trapSplit.trap_user_visible_resume,
  ];
  return {
    contract: LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT,
    trancheId: "must_not_fire_binary_gate_v2",
    taskId: params.outputTaskId,
    builtAt: params.generatedAt,
    status: "anchor_set_complete",
    purpose:
      "Merged abstention lane for binary-gate v2, combining the original must-not-fire anchors with the cleaned vector-collapse trap tranches.",
    targetTraceCount: normalizedOriginalAnchors.length + trapAnchors.length,
    anchorTraceCount: normalizedOriginalAnchors.length + trapAnchors.length,
    remainingToMine: 0,
    sourceManifests: uniqueStrings([
      ...((params.mustNotFireManifest.sourceManifests ?? []) as string[]),
      ...((params.trapManifest.sourceManifests ?? []) as string[]),
    ]),
    selectionRules: [
      "Keep the original semantic abstention anchors.",
      "Add cleaned vector-collapse trap anchors so the binary gate learns explicit vetoes instead of replaying vector_only behavior.",
      "Do not let wrapper/system-only cases completely dominate the abstention curriculum; keep user-visible trap cases present.",
    ],
    priorityBuckets: [
      "wrapper_or_continuation",
      "self_contained_semantic",
      "graph_prior_preferred_operational",
      "tie_with_cost",
      "trap_wrapper_system",
      "trap_operator_artifact",
      "trap_user_visible_resume",
    ],
    anchors: [...normalizedOriginalAnchors, ...trapAnchors],
    notes: [
      "This merged abstention lane keeps the original must-not-fire anchor set intact and appends cleaned vector-collapse traps for the first binary-gate retune.",
      "The split trap manifests should still be used as separate audit lanes when grading candidate restraint.",
    ],
  };
}

export function buildBinaryGateV2HardNegativeSpec(baseSpec: HardNegativeSpec, params?: { outputTaskId?: string; generatedAt?: string }): HardNegativeSpec {
  const bucketToHardNegativeClass = {
    wrapper_or_continuation: "wrapper_heavy",
    self_contained_semantic: "unnecessary_activation",
    graph_prior_preferred_operational: "graph_prior_preferred",
    tie_with_cost: "tie_with_cost",
    trap_wrapper_system: "wrapper_heavy",
    trap_operator_artifact: "unnecessary_activation",
    trap_user_visible_resume: "graph_prior_preferred",
  } as const satisfies Record<string, HardNegativeClass>;

  return {
    contract: LEARNED_ROUTE_HARD_NEGATIVE_MINING_SPEC_CONTRACT,
    taskId: params?.outputTaskId ?? DEFAULT_OUTPUT_TASK_ID,
    builtAt: params?.generatedAt ?? new Date().toISOString(),
    status: "binary_gate_v2_owner_mined_spec",
    purpose:
      "Turn the binary-gate v2 abstention lanes into explicit hard-negative classes while preserving the original must-not-fire mining pressure.",
    primaryGoal: "Improve abstention precision on wrapper/system and vector-collapse trap cohorts without collapsing felt-resume wins.",
    missionTestCommand: baseSpec.missionTestCommand ?? "npm run test:learned-route-mission",
    weightFloorsByClass: {
      unnecessary_activation: 1.5,
      tie_with_cost: 2.0,
      graph_prior_preferred: 2.0,
      stale_memory: 1.75,
      wrapper_heavy: 1.75,
      no_outcome_change: 1.5,
      ...(baseSpec.weightFloorsByClass ?? {}),
    },
    bucketToHardNegativeClass,
    miningRules: [
      {
        class: "wrapper_heavy",
        source: "must_not_fire_100 bucket wrapper_or_continuation and trap_wrapper_system",
        why: "Wrapper-heavy shells and system relay turns should be vetoed before retrieval even looks tempting.",
      },
      {
        class: "unnecessary_activation",
        source: "must_not_fire_100 bucket self_contained_semantic and trap_operator_artifact",
        why: "Operator/artifact directives can look memory-adjacent but often still carry enough inline state to stay off.",
      },
      {
        class: "graph_prior_preferred",
        source: "must_not_fire_100 bucket graph_prior_preferred_operational and trap_user_visible_resume",
        why: "Some user-visible resume turns should still favor the approved baseline path instead of a vector-heavy intervention.",
      },
      {
        class: "tie_with_cost",
        source: "must_not_fire_100 bucket tie_with_cost",
        why: "Equal-quality answers with higher retrieval cost remain a direct abstention penalty.",
      },
      {
        class: "stale_memory",
        source: "reviewed stale sibling retrievals and false positives",
        why: "Keep stale-memory pressure separate so the later gate does not learn broad suppression instead of precision.",
      },
      {
        class: "no_outcome_change",
        source: "activated-but-no-outcome-change replay or reviewed traces",
        why: "Preserve a distinct penalty for interventions that changed nothing even when they were not obviously wrapper-heavy.",
      },
    ],
    seedAssignments: {
      must_not_fire_binary_gate_v2: {
        anchorCount: 65,
        assignByBucket: true,
      },
      must_fire_binary_gate_v2: {
        anchorCount: 10,
        assignOnlyWhenReplayShowsMaterialGain: true,
      },
    },
    nextMiningPass: [
      "grade the binary gate separately on trap_wrapper_system, trap_operator_artifact, and trap_user_visible_resume",
      "expand split must_fire buckets beyond the first 10 anchors only after the binary gate shows honest wins",
      "keep retrieval-regime selection and STOP_LOCAL frozen until the activate-vs-abstain loop is clearly positive",
    ],
  };
}

function buildSummaryMarkdown(params: {
  generatedAt: string;
  mustFireSplit: Record<MustFireSplitTrancheId, TrancheAnchor[]>;
  trapSplit: Record<TrapSplitTrancheId, TrancheAnchor[]>;
  mergedPositive: TrancheManifest;
  mergedAbstention: TrancheManifest;
}): string {
  return [
    "# Binary-gate v2 tranche build",
    "",
    `- generatedAt: ${params.generatedAt}`,
    `- merged positive lane: ${params.mergedPositive.trancheId} (${params.mergedPositive.anchorTraceCount} anchors)`,
    `- merged abstention lane: ${params.mergedAbstention.trancheId} (${params.mergedAbstention.anchorTraceCount} anchors)`,
    "",
    "## must_fire split",
    ...Object.entries(params.mustFireSplit).map(([trancheId, anchors]) => `- ${trancheId}: ${anchors.length}`),
    "",
    "## trap split",
    ...Object.entries(params.trapSplit).map(([trancheId, anchors]) => `- ${trancheId}: ${anchors.length}`),
    "",
    "## Notes",
    "- Use the merged lanes for the first binary-gate retune input.",
    "- Use the split per-bucket manifests as audit lanes during candidate grading.",
    "- The trap split is heuristic and should be reviewed if future trap previews change materially.",
    "",
  ].join("\n");
}

export function buildBinaryGateV2Tranches(params: {
  outputTaskId: string;
  mustFireManifest: TrancheManifest;
  mustNotFireManifest: TrancheManifest;
  trapManifest: TrancheManifest;
  baseHardNegativeSpec: HardNegativeSpec;
  generatedAt: string;
}) {
  const mustFireSplit = splitMustFireAnchors(params.mustFireManifest.anchors);
  const trapSplit = splitTrapAnchors(params.trapManifest.anchors);

  const mustFireSplitManifests = Object.fromEntries(
    (Object.keys(mustFireSplit) as MustFireSplitTrancheId[]).map((trancheId) => [
      trancheId,
      buildSplitManifest({
        outputTaskId: params.outputTaskId,
        trancheId,
        purpose: MUST_FIRE_PURPOSE_BY_TRANCHE[trancheId],
        sourceManifest: params.mustFireManifest,
        anchors: mustFireSplit[trancheId],
        targetTraceCount: mustFireSplit[trancheId].length,
        selectionRules: [
          "These anchors are a subset of must_fire_30, split only to isolate the binary gate's first real positive win clusters.",
          "Keep the downstream retrieval and stop rules frozen while grading these positive cases.",
        ],
        priorityBuckets: [trancheId],
        notes: [
          "This split is intentionally explicit and trace-id stable for the current reviewed 10-anchor set.",
        ],
        generatedAt: params.generatedAt,
      }),
    ]),
  ) as Record<MustFireSplitTrancheId, TrancheManifest>;

  const trapSplitManifests = Object.fromEntries(
    (Object.keys(trapSplit) as TrapSplitTrancheId[]).map((trancheId) => [
      trancheId,
      buildSplitManifest({
        outputTaskId: params.outputTaskId,
        trancheId,
        purpose: TRAP_PURPOSE_BY_TRANCHE[trancheId],
        sourceManifest: params.trapManifest,
        anchors: trapSplit[trancheId],
        targetTraceCount: trapSplit[trancheId].length,
        selectionRules: [
          "These anchors come from vector_only_trap_50 and are split to separate wrapper-system avoidance from the more product-facing abstention cases.",
          "Review this split when the trap source manifest changes; it is intentionally heuristic on preview text.",
        ],
        priorityBuckets: [trancheId],
        notes: [
          `Derived hard-negative class: ${TRAP_BUCKET_TO_HARD_NEGATIVE_CLASS[trancheId]}`,
        ],
        generatedAt: params.generatedAt,
      }),
    ]),
  ) as Record<TrapSplitTrancheId, TrancheManifest>;

  const mergedPositive = buildMergedPositiveManifest({
    outputTaskId: params.outputTaskId,
    sourceManifest: params.mustFireManifest,
    split: mustFireSplit,
    generatedAt: params.generatedAt,
  });
  const mergedAbstention = buildMergedAbstentionManifest({
    outputTaskId: params.outputTaskId,
    mustNotFireManifest: params.mustNotFireManifest,
    trapManifest: params.trapManifest,
    trapSplit,
    generatedAt: params.generatedAt,
  });
  const hardNegativeSpec = buildBinaryGateV2HardNegativeSpec(params.baseHardNegativeSpec, {
    outputTaskId: params.outputTaskId,
    generatedAt: params.generatedAt,
  });

  return {
    mustFireSplit,
    trapSplit,
    mustFireSplitManifests,
    trapSplitManifests,
    mergedPositive,
    mergedAbstention,
    hardNegativeSpec,
    summaryMarkdown: buildSummaryMarkdown({
      generatedAt: params.generatedAt,
      mustFireSplit,
      trapSplit,
      mergedPositive,
      mergedAbstention,
    }),
  };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(args.outputDir, { recursive: true });

  const mustFireManifest = assertTrancheManifest(readJson<TrancheManifest>(args.mustFireManifest), args.mustFireManifest);
  const mustNotFireManifest = assertTrancheManifest(readJson<TrancheManifest>(args.mustNotFireManifest), args.mustNotFireManifest);
  const trapManifest = assertTrancheManifest(readJson<TrancheManifest>(args.trapManifest), args.trapManifest);
  const baseHardNegativeSpec = readJson<HardNegativeSpec>(args.hardNegativeSpec);

  const built = buildBinaryGateV2Tranches({
    outputTaskId: args.outputTaskId,
    mustFireManifest,
    mustNotFireManifest,
    trapManifest,
    baseHardNegativeSpec,
    generatedAt: args.generatedAt,
  });

  for (const [trancheId, manifest] of Object.entries(built.mustFireSplitManifests)) {
    writeJson(path.join(args.outputDir, `${trancheId}.manifest.json`), manifest);
  }
  for (const [trancheId, manifest] of Object.entries(built.trapSplitManifests)) {
    writeJson(path.join(args.outputDir, `${trancheId}.manifest.json`), manifest);
  }
  writeJson(path.join(args.outputDir, "must_fire_binary_gate_v2.manifest.json"), built.mergedPositive);
  writeJson(path.join(args.outputDir, "must_not_fire_binary_gate_v2.manifest.json"), built.mergedAbstention);
  writeJson(path.join(args.outputDir, "binary-gate-v2-hard-negative-spec.v1.json"), built.hardNegativeSpec);
  writeFileSync(path.join(args.outputDir, "README.md"), built.summaryMarkdown, "utf8");

  process.stdout.write(
    [
      `Binary-gate v2 tranche build: ${args.outputDir}`,
      `mustFireExactArtifact: ${built.mustFireSplit.must_fire_exact_artifact.length}`,
      `mustFireResumeState: ${built.mustFireSplit.must_fire_resume_state.length}`,
      `mustFireRecentDecision: ${built.mustFireSplit.must_fire_recent_decision.length}`,
      `mustFireStaleSummaryRepair: ${built.mustFireSplit.must_fire_stale_summary_repair.length}`,
      `trapWrapperSystem: ${built.trapSplit.trap_wrapper_system.length}`,
      `trapOperatorArtifact: ${built.trapSplit.trap_operator_artifact.length}`,
      `trapUserVisibleResume: ${built.trapSplit.trap_user_visible_resume.length}`,
      `mergedPositive: ${built.mergedPositive.anchorTraceCount}`,
      `mergedAbstention: ${built.mergedAbstention.anchorTraceCount}`,
    ].join("\n") + "\n",
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  }
}
