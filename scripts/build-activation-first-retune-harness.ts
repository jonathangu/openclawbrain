#!/usr/bin/env tsx

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

const FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT = "frozen_recorded_session_eval_manifest.v1" as const;
const LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT = "learned_route_eval_tranche_manifest.v1" as const;
const ACTIVATION_FIRST_RETUNE_HARNESS_CONTRACT = "activation_first_retune_harness.v1" as const;
const LEARNED_ROUTE_LABEL_SCHEMA = "learned-route-labels.v1" as const;
const DEFAULT_TASK_ID = "T-20260415-257";
const DEFAULT_MUST_FIRE_MANIFEST = path.join(workspaceRoot, "task-artifacts", DEFAULT_TASK_ID, "must-fire-30.manifest.json");
const DEFAULT_MUST_NOT_FIRE_MANIFEST = path.join(workspaceRoot, "task-artifacts", DEFAULT_TASK_ID, "must-not-fire-100.manifest.json");
const DEFAULT_HARD_NEGATIVE_SPEC = path.join(workspaceRoot, "task-artifacts", DEFAULT_TASK_ID, "hard-negative-mining-spec.v1.json");
const DEFAULT_GUARDRAIL_MANIFEST = path.join(workspaceRoot, "task-artifacts", "T-20260415-250", "semantic-rich-live-535-extracted", "manifest.json");
const DEFAULT_OUTPUT_DIR = path.join(workspaceRoot, "task-artifacts", DEFAULT_TASK_ID, "activation-first-retune-harness");

type OracleBestMode = "learned_route" | "graph_prior_only" | "tie";
type CostSensitive = "low" | "medium" | "high";

type HardNegativeClass =
  | "unnecessary_activation"
  | "tie_with_cost"
  | "graph_prior_preferred"
  | "stale_memory"
  | "wrapper_heavy"
  | "no_outcome_change";

interface TrancheAnchor {
  traceId: string;
  sourcePath: string;
  bucket?: string;
  whyIncluded?: string;
}

interface TrancheManifest {
  contract: typeof LEARNED_ROUTE_TRANCHE_MANIFEST_CONTRACT;
  trancheId: string;
  taskId?: string;
  purpose?: string;
  targetTraceCount?: number;
  anchorTraceCount?: number;
  notes?: string[];
  selectionRules?: string[];
  sourceManifests?: string[];
  anchors: TrancheAnchor[];
}

interface HardNegativeSpec {
  missionTestCommand?: string;
  bucketToHardNegativeClass?: Record<string, HardNegativeClass>;
  weightFloorsByClass?: Partial<Record<HardNegativeClass, number>>;
}

interface GuardrailManifest {
  contract?: string;
  setId?: string;
  manifestId?: string;
  traceCount?: number;
  expectedTraceCount?: number;
  realTraceCoverage?: {
    summary?: string;
  };
}

interface FrozenRecordedSessionEvalManifestV1 {
  contract: typeof FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT;
  manifestId: string;
  generatedAt: string;
  expectedTraceCount: number;
  notes: string[];
  traces: Array<{
    traceId: string;
    tracePath: string;
    notes: string[];
  }>;
}

interface LabelRecord {
  schema_version: typeof LEARNED_ROUTE_LABEL_SCHEMA;
  status: string;
  trace_id: string;
  source_path: string;
  bucket: string | null;
  annotator: string;
  labeled_at: string;
  labels: {
    memory_needed: "yes" | "no";
    wrapper_noise: "yes" | "no";
    continuation_only: "yes" | "no";
    operational_recovery: "yes" | "no";
    human_semantic_task: "yes" | "no";
    oracle_best_mode: OracleBestMode;
    cost_sensitive: CostSensitive;
    focus_lane: string;
    hard_negative_class: HardNegativeClass | null;
  };
  notes: {
    memory_needed_reason: string;
    oracle_reason: string;
    unclear_reason: null;
  };
  prefill: {
    seed_set: string;
    rationale: string;
    projection_source: string;
  };
}

interface ParsedArgs {
  taskId: string;
  mustFireManifest: string;
  mustNotFireManifest: string;
  hardNegativeSpec: string;
  guardrailManifest: string;
  outputDir: string;
  generatedAt: string;
}

function usage(): void {
  process.stderr.write(
    [
      "Usage: tsx scripts/build-activation-first-retune-harness.ts [options]",
      "",
      "Options:",
      `  --task-id <id>                 Defaults to ${DEFAULT_TASK_ID}`,
      `  --must-fire-manifest <path>   Defaults to ${DEFAULT_MUST_FIRE_MANIFEST}`,
      `  --must-not-fire-manifest <path> Defaults to ${DEFAULT_MUST_NOT_FIRE_MANIFEST}`,
      `  --hard-negative-spec <path>   Defaults to ${DEFAULT_HARD_NEGATIVE_SPEC}`,
      `  --guardrail-manifest <path>   Defaults to ${DEFAULT_GUARDRAIL_MANIFEST}`,
      `  --output-dir <path>           Defaults to ${DEFAULT_OUTPUT_DIR}`,
      "  --generated-at <iso>          Override generated timestamp",
      "  --help                        Show this help",
      "",
      "Outputs:",
      "  <output-dir>/must-fire-anchor-eval.manifest.json",
      "  <output-dir>/must-not-fire-anchor-eval.manifest.json",
      "  <output-dir>/activation-first-retune.labels-template.jsonl",
      "  <output-dir>/activation-first-retune-harness.json",
      "  <output-dir>/README.md",
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
    taskId: DEFAULT_TASK_ID,
    mustFireManifest: DEFAULT_MUST_FIRE_MANIFEST,
    mustNotFireManifest: DEFAULT_MUST_NOT_FIRE_MANIFEST,
    hardNegativeSpec: DEFAULT_HARD_NEGATIVE_SPEC,
    guardrailManifest: DEFAULT_GUARDRAIL_MANIFEST,
    outputDir: DEFAULT_OUTPUT_DIR,
    generatedAt: new Date().toISOString(),
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--task-id":
        parsed.taskId = normalizeCliString(argv[index + 1]) ?? parsed.taskId;
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
      case "--hard-negative-spec":
        parsed.hardNegativeSpec = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--guardrail-manifest":
        parsed.guardrailManifest = path.resolve(argv[index + 1] ?? "");
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

function portableRelativePath(fromDir: string, toPath: string): string {
  return path.relative(fromDir, toPath).split(path.sep).join("/");
}

function writeJson(filePath: string, value: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function resolveWorkspacePath(relativeOrAbsolutePath: string): string {
  return path.isAbsolute(relativeOrAbsolutePath)
    ? relativeOrAbsolutePath
    : path.resolve(workspaceRoot, relativeOrAbsolutePath);
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

function buildFrozenManifest(params: {
  tranche: TrancheManifest;
  manifestId: string;
  outputDir: string;
  generatedAt: string;
}): FrozenRecordedSessionEvalManifestV1 {
  return {
    contract: FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT,
    manifestId: params.manifestId,
    generatedAt: params.generatedAt,
    expectedTraceCount: params.tranche.anchors.length,
    notes: [
      params.tranche.purpose ?? `${params.tranche.trancheId} anchor eval manifest`,
      ...(params.tranche.notes ?? []),
      ...(params.tranche.selectionRules ?? []).map((rule) => `selection rule: ${rule}`),
    ],
    traces: params.tranche.anchors.map((anchor) => {
      const absoluteTracePath = resolveWorkspacePath(anchor.sourcePath);
      return {
        traceId: anchor.traceId,
        tracePath: portableRelativePath(params.outputDir, absoluteTracePath),
        notes: [anchor.bucket ? `bucket: ${anchor.bucket}` : null, anchor.whyIncluded ?? null].filter(
          (value): value is string => typeof value === "string" && value.length > 0,
        ),
      };
    }),
  };
}

function mustFireDefaults(anchor: TrancheAnchor, trancheId: string, generatedAt: string): LabelRecord {
  const bucket = anchor.bucket ?? null;
  const operationalRecovery = bucket !== null && (
    bucket.includes("runtime")
    || bucket.includes("config")
    || bucket.includes("merge")
    || bucket.includes("deploy")
    || bucket.includes("blocker")
    || bucket.includes("approval")
    || bucket.includes("prevention")
  );
  const costSensitive: CostSensitive = operationalRecovery ? "medium" : "low";
  return {
    schema_version: LEARNED_ROUTE_LABEL_SCHEMA,
    status: "prefilled_from_tranche_manifest",
    trace_id: anchor.traceId,
    source_path: anchor.sourcePath,
    bucket,
    annotator: "activation-first-retune-harness",
    labeled_at: generatedAt,
    labels: {
      memory_needed: "yes",
      wrapper_noise: "no",
      continuation_only: "no",
      operational_recovery: operationalRecovery ? "yes" : "no",
      human_semantic_task: "yes",
      oracle_best_mode: "learned_route",
      cost_sensitive: costSensitive,
      focus_lane: trancheId,
      hard_negative_class: null,
    },
    notes: {
      memory_needed_reason: anchor.whyIncluded ?? "Anchor selected because prior state should materially change the answer.",
      oracle_reason: "This must-fire anchor is prefilled toward learned_route because retrieval is expected to produce a real unique-win opportunity.",
      unclear_reason: null,
    },
    prefill: {
      seed_set: trancheId,
      rationale: anchor.whyIncluded ?? "Must-fire anchor from tranche manifest.",
      projection_source: "activation-first-retune-harness",
    },
  };
}

function mustNotFireDefaults(
  anchor: TrancheAnchor,
  trancheId: string,
  bucketToHardNegativeClass: Record<string, HardNegativeClass>,
  generatedAt: string,
): LabelRecord {
  const bucket = anchor.bucket ?? null;
  const hardNegativeClass = bucket ? (bucketToHardNegativeClass[bucket] ?? null) : null;
  const oracleBestMode: OracleBestMode = hardNegativeClass === "tie_with_cost" ? "tie" : "graph_prior_only";
  const continuationOnly = bucket === "wrapper_or_continuation";
  const wrapperNoise = bucket === "wrapper_or_continuation";
  const costSensitive: CostSensitive = hardNegativeClass === "tie_with_cost" || continuationOnly
    ? "high"
    : hardNegativeClass === "graph_prior_preferred"
      ? "medium"
      : "medium";
  return {
    schema_version: LEARNED_ROUTE_LABEL_SCHEMA,
    status: "prefilled_from_tranche_manifest",
    trace_id: anchor.traceId,
    source_path: anchor.sourcePath,
    bucket,
    annotator: "activation-first-retune-harness",
    labeled_at: generatedAt,
    labels: {
      memory_needed: "no",
      wrapper_noise: wrapperNoise ? "yes" : "no",
      continuation_only: continuationOnly ? "yes" : "no",
      operational_recovery: hardNegativeClass === "graph_prior_preferred" ? "yes" : "no",
      human_semantic_task: continuationOnly ? "no" : "yes",
      oracle_best_mode: oracleBestMode,
      cost_sensitive: costSensitive,
      focus_lane: trancheId,
      hard_negative_class: hardNegativeClass,
    },
    notes: {
      memory_needed_reason: anchor.whyIncluded ?? "Anchor selected because the current turn should be answerable without retrieval.",
      oracle_reason: hardNegativeClass === "tie_with_cost"
        ? "This must-not-fire anchor is prefilled as a tie-with-cost restraint case, so learned retrieval should tie at best and stay off."
        : "This must-not-fire anchor is prefilled toward graph_prior_only because retrieval should not be necessary here.",
      unclear_reason: null,
    },
    prefill: {
      seed_set: trancheId,
      rationale: anchor.whyIncluded ?? "Must-not-fire anchor from tranche manifest.",
      projection_source: "activation-first-retune-harness",
    },
  };
}

function buildReadme(params: {
  taskId: string;
  generatedAt: string;
  mustFire: TrancheManifest;
  mustNotFire: TrancheManifest;
  guardrail: GuardrailManifest;
  hardNegativeSpec: HardNegativeSpec;
  outputs: {
    mustFireManifestPath: string;
    mustNotFireManifestPath: string;
    labelsPath: string;
    harnessJsonPath: string;
  };
}): string {
  const missionTestCommand = params.hardNegativeSpec.missionTestCommand ?? "npm run test:learned-route-mission";
  return [
    "# Activation-first retune harness",
    "",
    `- task: ${params.taskId}`,
    `- generatedAt: ${params.generatedAt}`,
    `- mission test: \`${missionTestCommand}\``,
    "",
    "## Generated inputs",
    `- must-fire anchor eval manifest: \`${portableRelativePath(path.dirname(params.outputs.harnessJsonPath), params.outputs.mustFireManifestPath)}\``,
    `- must-not-fire anchor eval manifest: \`${portableRelativePath(path.dirname(params.outputs.harnessJsonPath), params.outputs.mustNotFireManifestPath)}\``,
    `- prefilled labels template: \`${portableRelativePath(path.dirname(params.outputs.harnessJsonPath), params.outputs.labelsPath)}\``,
    `- harness descriptor: \`${portableRelativePath(path.dirname(params.outputs.harnessJsonPath), params.outputs.harnessJsonPath)}\``,
    "",
    "## Truth boundaries",
    `- must-fire anchors available now: ${params.mustFire.anchors.length}/${params.mustFire.targetTraceCount ?? params.mustFire.anchors.length}`,
    `- must-not-fire anchors available now: ${params.mustNotFire.anchors.length}/${params.mustNotFire.targetTraceCount ?? params.mustNotFire.anchors.length}`,
    `- guardrail lane: ${params.guardrail.setId ?? params.guardrail.manifestId ?? "semantic-rich-live"} (${params.guardrail.traceCount ?? params.guardrail.expectedTraceCount ?? "unknown"} traces)`,
    ...(params.guardrail.realTraceCoverage?.summary ? [`- guardrail truth boundary: ${params.guardrail.realTraceCoverage.summary}`] : []),
    "",
    "## Suggested execution order",
    `1. Review and tighten \`${path.basename(params.outputs.labelsPath)}\` for the current anchor tranche defaults.`,
    `2. Run comparative replay on \`${path.basename(params.outputs.mustFireManifestPath)}\` and \`${path.basename(params.outputs.mustNotFireManifestPath)}\`.`,
    "3. Keep the semantic-rich broad-live lane as a guardrail lane, not the optimize lane.",
    `4. Re-run \`${missionTestCommand}\` after each candidate retune.`,
    "",
  ].join("\n");
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(args.outputDir, { recursive: true });

  const mustFire = assertTrancheManifest(readJson<TrancheManifest>(args.mustFireManifest), args.mustFireManifest);
  const mustNotFire = assertTrancheManifest(readJson<TrancheManifest>(args.mustNotFireManifest), args.mustNotFireManifest);
  const hardNegativeSpec = readJson<HardNegativeSpec>(args.hardNegativeSpec);
  const guardrailManifest = readJson<GuardrailManifest>(args.guardrailManifest);
  const bucketToHardNegativeClass = hardNegativeSpec.bucketToHardNegativeClass ?? {};

  const mustFireEvalManifestPath = path.join(args.outputDir, "must-fire-anchor-eval.manifest.json");
  const mustNotFireEvalManifestPath = path.join(args.outputDir, "must-not-fire-anchor-eval.manifest.json");
  const labelsPath = path.join(args.outputDir, "activation-first-retune.labels-template.jsonl");
  const harnessJsonPath = path.join(args.outputDir, "activation-first-retune-harness.json");
  const readmePath = path.join(args.outputDir, "README.md");

  const mustFireEvalManifest = buildFrozenManifest({
    tranche: mustFire,
    manifestId: `${mustFire.trancheId}-anchor-eval`,
    outputDir: args.outputDir,
    generatedAt: args.generatedAt,
  });
  const mustNotFireEvalManifest = buildFrozenManifest({
    tranche: mustNotFire,
    manifestId: `${mustNotFire.trancheId}-anchor-eval`,
    outputDir: args.outputDir,
    generatedAt: args.generatedAt,
  });

  writeJson(mustFireEvalManifestPath, mustFireEvalManifest);
  writeJson(mustNotFireEvalManifestPath, mustNotFireEvalManifest);

  const labelRecords: LabelRecord[] = [
    ...mustFire.anchors.map((anchor) => mustFireDefaults(anchor, mustFire.trancheId, args.generatedAt)),
    ...mustNotFire.anchors.map((anchor) => mustNotFireDefaults(anchor, mustNotFire.trancheId, bucketToHardNegativeClass, args.generatedAt)),
  ];
  writeFileSync(labelsPath, `${labelRecords.map((record) => JSON.stringify(record)).join("\n")}\n`, "utf8");

  const harnessDescriptor = {
    contract: ACTIVATION_FIRST_RETUNE_HARNESS_CONTRACT,
    taskId: args.taskId,
    generatedAt: args.generatedAt,
    missionTestCommand: hardNegativeSpec.missionTestCommand ?? "npm run test:learned-route-mission",
    labelsTemplatePath: portableRelativePath(args.outputDir, labelsPath),
    mustFire: {
      trancheId: mustFire.trancheId,
      purpose: mustFire.purpose ?? null,
      anchorTraceCount: mustFire.anchors.length,
      targetTraceCount: mustFire.targetTraceCount ?? null,
      manifestPath: portableRelativePath(args.outputDir, mustFireEvalManifestPath),
      sourceManifestPath: portableRelativePath(args.outputDir, args.mustFireManifest),
      comparativeEvalCommand: `npx tsx scripts/eval/run-comparative-eval.ts --manifest ${portableRelativePath(repoRoot, mustFireEvalManifestPath)}`,
      replayLaneCommand: `npx tsx scripts/build-recorded-session-replay-lane.ts --trace-manifest ${portableRelativePath(repoRoot, mustFireEvalManifestPath)}`,
    },
    mustNotFire: {
      trancheId: mustNotFire.trancheId,
      purpose: mustNotFire.purpose ?? null,
      anchorTraceCount: mustNotFire.anchors.length,
      targetTraceCount: mustNotFire.targetTraceCount ?? null,
      manifestPath: portableRelativePath(args.outputDir, mustNotFireEvalManifestPath),
      sourceManifestPath: portableRelativePath(args.outputDir, args.mustNotFireManifest),
      comparativeEvalCommand: `npx tsx scripts/eval/run-comparative-eval.ts --manifest ${portableRelativePath(repoRoot, mustNotFireEvalManifestPath)}`,
      replayLaneCommand: `npx tsx scripts/build-recorded-session-replay-lane.ts --trace-manifest ${portableRelativePath(repoRoot, mustNotFireEvalManifestPath)}`,
    },
    broadLiveGuardrail: {
      manifestPath: portableRelativePath(args.outputDir, args.guardrailManifest),
      manifestContract: guardrailManifest.contract ?? null,
      manifestId: guardrailManifest.setId ?? guardrailManifest.manifestId ?? null,
      traceCount: guardrailManifest.traceCount ?? guardrailManifest.expectedTraceCount ?? null,
      comparativeEvalCommand: `npx tsx scripts/eval/run-comparative-eval.ts --manifest ${portableRelativePath(repoRoot, args.guardrailManifest)}`,
      replayLaneCommand: `npx tsx scripts/build-recorded-session-replay-lane.ts --trace-manifest ${portableRelativePath(repoRoot, args.guardrailManifest)}`,
    },
    hardNegativeMining: {
      specPath: portableRelativePath(args.outputDir, args.hardNegativeSpec),
      bucketToHardNegativeClass,
      weightFloorsByClass: hardNegativeSpec.weightFloorsByClass ?? {},
    },
  };
  writeJson(harnessJsonPath, harnessDescriptor);
  writeFileSync(readmePath, buildReadme({
    taskId: args.taskId,
    generatedAt: args.generatedAt,
    mustFire,
    mustNotFire,
    guardrail: guardrailManifest,
    hardNegativeSpec,
    outputs: {
      mustFireManifestPath: mustFireEvalManifestPath,
      mustNotFireManifestPath: mustNotFireEvalManifestPath,
      labelsPath,
      harnessJsonPath,
    },
  }), "utf8");

  process.stdout.write(
    [
      `Activation-first retune harness: ${args.outputDir}`,
      `mustFireManifest: ${mustFireEvalManifestPath}`,
      `mustNotFireManifest: ${mustNotFireEvalManifestPath}`,
      `labelsTemplate: ${labelsPath}`,
      `harnessDescriptor: ${harnessJsonPath}`,
      `readme: ${readmePath}`,
      `labelRecordCount: ${labelRecords.length}`,
    ].join("\n") + "\n",
  );
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
}
