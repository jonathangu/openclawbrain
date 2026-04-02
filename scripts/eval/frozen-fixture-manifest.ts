#!/usr/bin/env tsx

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");

export const CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT = "canonical_recorded_session_trace_set_manifest.v1";
export const FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_CONTRACT = "frozen_recorded_session_fixture_manifest.v1";
export const FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_VALIDATION_CONTRACT =
  "frozen_recorded_session_fixture_manifest_validation.v1";
export const RECORDED_SESSION_TRACE_CONTRACT = "recorded_session_trace.v1";
export const RECORDED_SESSION_FIXTURE_CONTRACT = "recorded_session_replay_fixture.v1";
export const DEFAULT_TRACE_MANIFEST_PATH = path.resolve(
  repoRoot,
  "evals",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.json",
);
export const DEFAULT_FIXTURE_MANIFEST_PATH = path.resolve(
  repoRoot,
  "artifacts",
  "fixtures",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.json",
);
export const DEFAULT_FIXTURE_SCHEMA_PATH = path.resolve(
  repoRoot,
  "artifacts",
  "fixtures",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.schema.json",
);

export const TRACE_FAMILY_ORDER = [
  "direct_answer",
  "plan_execution",
  "retrieval_memory_heavy",
  "correction_follow_up_heavy",
] as const;

export const TRACE_FAMILY_DIRS = {
  direct_answer: "direct-answer",
  plan_execution: "plan-execution",
  retrieval_memory_heavy: "retrieval-memory-heavy",
  correction_follow_up_heavy: "correction-follow-up-heavy",
} as const;

const DEFAULT_GENERATED_AT = "2026-04-02T00:00:00.000Z";
const HASH_ALGORITHM = "sha256";
const MANIFEST_STATUS = "scaffold_only";
const HASH_RULE_TRACE = "sha256(canonicalJson(trace.json))";
const HASH_RULE_SELECTION = "sha256(canonicalJson(slot-selection-metadata))";
const HASH_RULE_FIXTURE = "sha256(canonicalJson(recorded_session_replay_fixture.v1)) after fixture materialization";
const HASH_RULE_MANIFEST = "sha256(canonicalJson(manifest-without-self-hash))";

type TraceFamily = (typeof TRACE_FAMILY_ORDER)[number];
type TraceFamilyDir = (typeof TRACE_FAMILY_DIRS)[TraceFamily];
type RecordedSessionTraceV1 = Record<string, unknown> & {
  contract: string;
};

interface CanonicalTraceManifestEntry {
  slotId: string;
  title: string;
  category: TraceFamily;
  sourceKind: string;
  sourcePaths: string[];
  tags: string[];
  notes: string[];
  path: string;
  status: string;
  realTraceSourceAvailable: boolean;
  sanitization: {
    classification: string;
    redactionRequired: boolean;
    notes: string[];
  };
  shape: {
    turnCount: number;
    evalTurnCount: number;
    feedbackKinds: string[];
    followUpTurnCount: number;
    runtimeHintTurnCount: number;
  };
}

export interface CanonicalRecordedSessionTraceSetManifestV1 {
  contract: typeof CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT;
  setId: string;
  frozenAt: string;
  traceContract: typeof RECORDED_SESSION_TRACE_CONTRACT;
  root: string;
  traceCount: number;
  categoryOrder: TraceFamily[];
  categoryCounts: Record<TraceFamily, number>;
  sourceSummary: Record<string, number>;
  realTraceCoverage: {
    availableCount: number;
    missingCount: number;
    summary: string;
  };
  redactionPolicy: {
    additionalRedactionRequired: boolean;
    summary: string;
  };
  bundlePathTemplate: string;
  selectionPrinciples: string[];
  entries: CanonicalTraceManifestEntry[];
}

export interface FrozenRecordedSessionFixtureEntryMetadataV1 {
  sourceKind: string;
  sourcePaths: string[];
  tags: string[];
  notes: string[];
  sourceTraceStatus: string;
  realTraceSourceAvailable: boolean;
  sanitization: {
    classification: string;
    redactionRequired: boolean;
    notes: string[];
  };
  shape: {
    turnCount: number;
    evalTurnCount: number;
    feedbackKinds: string[];
    followUpTurnCount: number;
    runtimeHintTurnCount: number;
  };
}

export interface FrozenRecordedSessionFixtureManifestEntryV1 {
  fixtureId: string;
  slotId: string;
  title: string;
  traceFamily: TraceFamily;
  traceFamilyDir: TraceFamilyDir;
  tracePath: string;
  traceHash: string;
  selectionHash: string;
  fixturePath: string;
  fixtureHash: null;
  fixtureHashStatus: "pending_materialization";
  metadata: FrozenRecordedSessionFixtureEntryMetadataV1;
}

export interface FrozenRecordedSessionFixtureManifestV1 {
  contract: typeof FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_CONTRACT;
  manifestId: string;
  generatedAt: string;
  status: typeof MANIFEST_STATUS;
  root: string;
  traceManifest: {
    contract: typeof CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT;
    setId: string;
    path: string;
    frozenAt: string;
    traceContract: typeof RECORDED_SESSION_TRACE_CONTRACT;
    traceCount: number;
  };
  fixtureContract: typeof RECORDED_SESSION_FIXTURE_CONTRACT;
  traceFamilyOrder: TraceFamily[];
  traceFamilyCounts: Record<TraceFamily, number>;
  materialization: {
    fixtureFilesCheckedIn: number;
    pendingFixtureCount: number;
    summary: string;
  };
  hashing: {
    algorithm: typeof HASH_ALGORITHM;
    traceHashRule: string;
    selectionHashRule: string;
    fixtureHashRule: string;
    manifestHashRule: string;
    manifestCanonicalization: "canonicalJson";
    immutabilityRules: string[];
  };
  selectionRules: string[];
  fixturePathTemplate: string;
  entries: FrozenRecordedSessionFixtureManifestEntryV1[];
}

export interface FrozenRecordedSessionFixtureManifestValidationV1 {
  contract: typeof FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_VALIDATION_CONTRACT;
  ok: boolean;
  manifestPath: string;
  traceManifestPath: string | null;
  entryCount: number;
  errors: string[];
}

interface BuildFrozenRecordedSessionFixtureManifestInput {
  traceManifestPath?: string;
  outputManifestPath?: string;
  generatedAt?: string;
}

interface WriteFrozenRecordedSessionFixtureManifestInput extends BuildFrozenRecordedSessionFixtureManifestInput {
  schemaOutputPath?: string;
}

function usage(): void {
  process.stderr.write(
    [
      "Usage: tsx scripts/eval/frozen-fixture-manifest.ts <write|validate> [options]",
      "",
      "Commands:",
      "  write     Generate the checked-in frozen fixture scaffold manifest and schema.",
      "  validate  Validate the checked-in frozen fixture scaffold manifest against the canonical trace set.",
      "",
      "Options:",
      `  --trace-manifest <path>  Source trace manifest. Default ${DEFAULT_TRACE_MANIFEST_PATH}`,
      `  --manifest <path>        Output or validation manifest. Default ${DEFAULT_FIXTURE_MANIFEST_PATH}`,
      `  --schema <path>          Schema output path for write. Default ${DEFAULT_FIXTURE_SCHEMA_PATH}`,
      `  --generated-at <iso>     Generated-at timestamp for write. Default ${DEFAULT_GENERATED_AT}`,
      "  --help                   Show this help.",
    ].join("\n") + "\n",
  );
}

function normalizeCliString(value: string | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length === 0 ? null : trimmed;
}

function parseArgs(argv: string[]) {
  const [command, ...rest] = argv;
  if (!command || command === "--help" || command === "-h") {
    usage();
    process.exit(0);
  }
  if (command !== "write" && command !== "validate") {
    throw new Error(`Unknown command: ${command}`);
  }
  const parsed = {
    command,
    traceManifestPath: DEFAULT_TRACE_MANIFEST_PATH,
    manifestPath: DEFAULT_FIXTURE_MANIFEST_PATH,
    schemaPath: DEFAULT_FIXTURE_SCHEMA_PATH,
    generatedAt: DEFAULT_GENERATED_AT,
  };
  for (let index = 0; index < rest.length; index += 1) {
    const arg = rest[index];
    switch (arg) {
      case "--trace-manifest":
        parsed.traceManifestPath = path.resolve(rest[index + 1] ?? "");
        index += 1;
        break;
      case "--manifest":
        parsed.manifestPath = path.resolve(rest[index + 1] ?? "");
        index += 1;
        break;
      case "--schema":
        parsed.schemaPath = path.resolve(rest[index + 1] ?? "");
        index += 1;
        break;
      case "--generated-at":
        parsed.generatedAt = rest[index + 1] ?? "";
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

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function sortJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => sortJsonValue(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, sortJsonValue(nested)]),
    );
  }
  return value;
}

export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortJsonValue(value), null, 2);
}

export function checksumJsonPayload(value: unknown): string {
  const digest = createHash(HASH_ALGORITHM).update(canonicalJson(value)).digest("hex");
  return `${HASH_ALGORITHM}-${digest}`;
}

function writeJsonFile(filePath: string, payload: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${canonicalJson(payload)}\n`, "utf8");
}

function asTraceFamily(value: string): TraceFamily {
  if (TRACE_FAMILY_ORDER.includes(value as TraceFamily)) {
    return value as TraceFamily;
  }
  throw new Error(`Unsupported trace family: ${value}`);
}

function getTraceFamilyDir(family: TraceFamily): TraceFamilyDir {
  return TRACE_FAMILY_DIRS[family];
}

function buildFixturePath(slotId: string, family: TraceFamily): string {
  return path.posix.join("fixtures", getTraceFamilyDir(family), slotId, "fixture.json");
}

function buildFixtureId(slotId: string): string {
  return slotId;
}

function toPosixRelativePath(fromDir: string, targetPath: string): string {
  return path.relative(fromDir, targetPath).split(path.sep).join(path.posix.sep);
}

function loadCanonicalTraceManifest(traceManifestPath: string): CanonicalRecordedSessionTraceSetManifestV1 {
  const manifest = readJsonFile<CanonicalRecordedSessionTraceSetManifestV1>(traceManifestPath);
  if (manifest.contract !== CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT) {
    throw new Error(`Unsupported trace manifest contract at ${traceManifestPath}: ${manifest.contract}`);
  }
  return manifest;
}

function loadTrace(tracePath: string): RecordedSessionTraceV1 {
  const trace = readJsonFile<RecordedSessionTraceV1>(tracePath);
  if (trace.contract !== RECORDED_SESSION_TRACE_CONTRACT) {
    throw new Error(`Unsupported trace contract at ${tracePath}: ${trace.contract}`);
  }
  return trace;
}

function buildSelectionHash(entry: Omit<FrozenRecordedSessionFixtureManifestEntryV1, "selectionHash" | "fixtureHash" | "fixtureHashStatus">): string {
  return checksumJsonPayload(entry);
}

function buildManifestPayload(
  traceManifestPath: string,
  outputManifestPath: string,
  generatedAt: string,
): FrozenRecordedSessionFixtureManifestV1 {
  const sourceManifest = loadCanonicalTraceManifest(traceManifestPath);
  const traceManifestDir = path.dirname(traceManifestPath);
  const outputDir = path.dirname(outputManifestPath);
  const root = toPosixRelativePath(repoRoot, outputDir);
  const traceManifestRelativePath = toPosixRelativePath(outputDir, traceManifestPath);

  const entries = sourceManifest.entries.map<FrozenRecordedSessionFixtureManifestEntryV1>((entry) => {
    const family = asTraceFamily(entry.category);
    const absoluteTracePath = path.resolve(traceManifestDir, entry.path);
    const trace = loadTrace(absoluteTracePath);
    const traceHash = checksumJsonPayload(trace);
    const scaffoldEntry = {
      fixtureId: buildFixtureId(entry.slotId),
      slotId: entry.slotId,
      title: entry.title,
      traceFamily: family,
      traceFamilyDir: getTraceFamilyDir(family),
      tracePath: toPosixRelativePath(outputDir, absoluteTracePath),
      traceHash,
      fixturePath: buildFixturePath(entry.slotId, family),
      metadata: {
        sourceKind: entry.sourceKind,
        sourcePaths: [...entry.sourcePaths],
        tags: [...entry.tags],
        notes: [...entry.notes],
        sourceTraceStatus: entry.status,
        realTraceSourceAvailable: entry.realTraceSourceAvailable,
        sanitization: {
          classification: entry.sanitization.classification,
          redactionRequired: entry.sanitization.redactionRequired,
          notes: [...entry.sanitization.notes],
        },
        shape: {
          turnCount: entry.shape.turnCount,
          evalTurnCount: entry.shape.evalTurnCount,
          feedbackKinds: [...entry.shape.feedbackKinds],
          followUpTurnCount: entry.shape.followUpTurnCount,
          runtimeHintTurnCount: entry.shape.runtimeHintTurnCount,
        },
      },
    };
    return {
      ...scaffoldEntry,
      selectionHash: buildSelectionHash(scaffoldEntry),
      fixtureHash: null,
      fixtureHashStatus: "pending_materialization",
    };
  });

  return {
    contract: FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_CONTRACT,
    manifestId: sourceManifest.setId,
    generatedAt,
    status: MANIFEST_STATUS,
    root,
    traceManifest: {
      contract: sourceManifest.contract,
      setId: sourceManifest.setId,
      path: traceManifestRelativePath,
      frozenAt: sourceManifest.frozenAt,
      traceContract: sourceManifest.traceContract,
      traceCount: sourceManifest.traceCount,
    },
    fixtureContract: RECORDED_SESSION_FIXTURE_CONTRACT,
    traceFamilyOrder: [...sourceManifest.categoryOrder],
    traceFamilyCounts: { ...sourceManifest.categoryCounts },
    materialization: {
      fixtureFilesCheckedIn: 0,
      pendingFixtureCount: sourceManifest.traceCount,
      summary:
        "This scaffold freezes slot membership, trace metadata, slot-selection hashes, and the future fixture-hash rule without checking fixture.json payloads into the repo yet.",
    },
    hashing: {
      algorithm: HASH_ALGORITHM,
      traceHashRule: HASH_RULE_TRACE,
      selectionHashRule: HASH_RULE_SELECTION,
      fixtureHashRule: HASH_RULE_FIXTURE,
      manifestHashRule: HASH_RULE_MANIFEST,
      manifestCanonicalization: "canonicalJson",
      immutabilityRules: [
        "Do not change slot membership, trace family assignment, tracePath, or fixturePath in place. Any such change requires a new manifest version or manifestId.",
        "Do not rewrite the source trace content in place. A changed traceHash invalidates the frozen selection and requires a new manifest regeneration.",
        "selectionHash freezes the per-slot scaffold metadata and must change whenever slot-level provenance or path metadata changes.",
        "fixtureHash values are intentionally absent in this scaffold-only manifest. When fixtures are materialized later, their hashes must be derived exactly from the fixtureHashRule instead of backfilled by hand.",
        "sourceKind, sourcePaths, sanitization metadata, and real-trace availability are part of the provenance contract and must not drift silently.",
        "Materializing fixture.json files later must preserve fixturePath exactly for each slot and add fixture hashes without altering slot identity.",
      ],
    },
    selectionRules: [
      ...sourceManifest.selectionPrinciples,
      "Preserve the exact 5/5/5/5 trace-family split from the canonical trace manifest.",
      "Derive fixture metadata only from the checked-in canonical trace manifest and replayable trace files already in-repo.",
      "Do not add or substitute traces in this scaffold; future materialization must operate on these exact slots.",
    ],
    fixturePathTemplate: "fixtures/<family-dir>/<slot-id>/fixture.json",
    entries,
  };
}

export function buildFrozenRecordedSessionFixtureManifest(
  input: BuildFrozenRecordedSessionFixtureManifestInput = {},
): FrozenRecordedSessionFixtureManifestV1 {
  return buildManifestPayload(
    path.resolve(input.traceManifestPath ?? DEFAULT_TRACE_MANIFEST_PATH),
    path.resolve(input.outputManifestPath ?? DEFAULT_FIXTURE_MANIFEST_PATH),
    input.generatedAt ?? DEFAULT_GENERATED_AT,
  );
}

export function buildFrozenRecordedSessionFixtureManifestSchema(): Record<string, unknown> {
  return {
    $schema: "https://json-schema.org/draft/2020-12/schema",
    title: "Frozen Recorded Session Fixture Manifest",
    type: "object",
    additionalProperties: false,
    required: [
      "contract",
      "manifestId",
      "generatedAt",
      "status",
      "root",
      "traceManifest",
      "fixtureContract",
      "traceFamilyOrder",
      "traceFamilyCounts",
      "materialization",
      "hashing",
      "selectionRules",
      "fixturePathTemplate",
      "entries",
    ],
    properties: {
      contract: {
        const: FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_CONTRACT,
      },
      manifestId: {
        type: "string",
        minLength: 1,
      },
      generatedAt: {
        type: "string",
        minLength: 1,
      },
      status: {
        const: MANIFEST_STATUS,
      },
      root: {
        type: "string",
        minLength: 1,
      },
      traceManifest: {
        type: "object",
        additionalProperties: false,
        required: ["contract", "setId", "path", "frozenAt", "traceContract", "traceCount"],
        properties: {
          contract: {
            const: CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT,
          },
          setId: {
            type: "string",
            minLength: 1,
          },
          path: {
            type: "string",
            minLength: 1,
          },
          frozenAt: {
            type: "string",
            minLength: 1,
          },
          traceContract: {
            const: RECORDED_SESSION_TRACE_CONTRACT,
          },
          traceCount: {
            type: "integer",
            minimum: 1,
          },
        },
      },
      fixtureContract: {
        const: RECORDED_SESSION_FIXTURE_CONTRACT,
      },
      traceFamilyOrder: {
        type: "array",
        minItems: TRACE_FAMILY_ORDER.length,
        maxItems: TRACE_FAMILY_ORDER.length,
        items: {
          enum: [...TRACE_FAMILY_ORDER],
        },
      },
      traceFamilyCounts: {
        type: "object",
        additionalProperties: false,
        required: [...TRACE_FAMILY_ORDER],
        properties: {
          direct_answer: {
            type: "integer",
            minimum: 0,
          },
          plan_execution: {
            type: "integer",
            minimum: 0,
          },
          retrieval_memory_heavy: {
            type: "integer",
            minimum: 0,
          },
          correction_follow_up_heavy: {
            type: "integer",
            minimum: 0,
          },
        },
      },
      materialization: {
        type: "object",
        additionalProperties: false,
        required: ["fixtureFilesCheckedIn", "pendingFixtureCount", "summary"],
        properties: {
          fixtureFilesCheckedIn: {
            type: "integer",
            minimum: 0,
          },
          pendingFixtureCount: {
            type: "integer",
            minimum: 0,
          },
          summary: {
            type: "string",
            minLength: 1,
          },
        },
      },
      hashing: {
        type: "object",
        additionalProperties: false,
        required: [
          "algorithm",
          "traceHashRule",
          "selectionHashRule",
          "fixtureHashRule",
          "manifestHashRule",
          "manifestCanonicalization",
          "immutabilityRules",
        ],
        properties: {
          algorithm: {
            const: HASH_ALGORITHM,
          },
          traceHashRule: {
            type: "string",
            minLength: 1,
          },
          selectionHashRule: {
            type: "string",
            minLength: 1,
          },
          fixtureHashRule: {
            type: "string",
            minLength: 1,
          },
          manifestHashRule: {
            type: "string",
            minLength: 1,
          },
          manifestCanonicalization: {
            const: "canonicalJson",
          },
          immutabilityRules: {
            type: "array",
            minItems: 1,
            items: {
              type: "string",
              minLength: 1,
            },
          },
        },
      },
      selectionRules: {
        type: "array",
        minItems: 1,
        items: {
          type: "string",
          minLength: 1,
        },
      },
      fixturePathTemplate: {
        type: "string",
        minLength: 1,
      },
      entries: {
        type: "array",
        minItems: 20,
        items: {
          type: "object",
          additionalProperties: false,
          required: [
            "fixtureId",
            "slotId",
            "title",
            "traceFamily",
            "traceFamilyDir",
            "tracePath",
            "traceHash",
            "selectionHash",
            "fixturePath",
            "fixtureHash",
            "fixtureHashStatus",
            "metadata",
          ],
          properties: {
            fixtureId: {
              type: "string",
              minLength: 1,
            },
            slotId: {
              type: "string",
              minLength: 1,
            },
            title: {
              type: "string",
              minLength: 1,
            },
            traceFamily: {
              enum: [...TRACE_FAMILY_ORDER],
            },
            traceFamilyDir: {
              enum: Object.values(TRACE_FAMILY_DIRS),
            },
            tracePath: {
              type: "string",
              minLength: 1,
            },
            traceHash: {
              type: "string",
              pattern: "^sha256-[a-f0-9]{64}$",
            },
            selectionHash: {
              type: "string",
              pattern: "^sha256-[a-f0-9]{64}$",
            },
            fixturePath: {
              type: "string",
              minLength: 1,
            },
            fixtureHash: {
              type: "null",
            },
            fixtureHashStatus: {
              const: "pending_materialization",
            },
            metadata: {
              type: "object",
              additionalProperties: false,
              required: [
                "sourceKind",
                "sourcePaths",
                "tags",
                "notes",
                "sourceTraceStatus",
                "realTraceSourceAvailable",
                "sanitization",
                "shape",
              ],
              properties: {
                sourceKind: {
                  type: "string",
                  minLength: 1,
                },
                sourcePaths: {
                  type: "array",
                  minItems: 1,
                  items: {
                    type: "string",
                    minLength: 1,
                  },
                },
                tags: {
                  type: "array",
                  minItems: 1,
                  items: {
                    type: "string",
                    minLength: 1,
                  },
                },
                notes: {
                  type: "array",
                  minItems: 1,
                  items: {
                    type: "string",
                    minLength: 1,
                  },
                },
                sourceTraceStatus: {
                  type: "string",
                  minLength: 1,
                },
                realTraceSourceAvailable: {
                  type: "boolean",
                },
                sanitization: {
                  type: "object",
                  additionalProperties: false,
                  required: ["classification", "redactionRequired", "notes"],
                  properties: {
                    classification: {
                      type: "string",
                      minLength: 1,
                    },
                    redactionRequired: {
                      type: "boolean",
                    },
                    notes: {
                      type: "array",
                      minItems: 1,
                      items: {
                        type: "string",
                        minLength: 1,
                      },
                    },
                  },
                },
                shape: {
                  type: "object",
                  additionalProperties: false,
                  required: [
                    "turnCount",
                    "evalTurnCount",
                    "feedbackKinds",
                    "followUpTurnCount",
                    "runtimeHintTurnCount",
                  ],
                  properties: {
                    turnCount: {
                      type: "integer",
                      minimum: 1,
                    },
                    evalTurnCount: {
                      type: "integer",
                      minimum: 1,
                    },
                    feedbackKinds: {
                      type: "array",
                      items: {
                        type: "string",
                        minLength: 1,
                      },
                    },
                    followUpTurnCount: {
                      type: "integer",
                      minimum: 0,
                    },
                    runtimeHintTurnCount: {
                      type: "integer",
                      minimum: 0,
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
  };
}

function validateExpectedPath(entry: FrozenRecordedSessionFixtureManifestEntryV1): string | null {
  const expectedFixturePath = buildFixturePath(entry.slotId, entry.traceFamily);
  if (entry.fixturePath !== expectedFixturePath) {
    return `slot ${entry.slotId} fixturePath mismatch: expected ${expectedFixturePath}, received ${entry.fixturePath}`;
  }
  const expectedFamilyDir = getTraceFamilyDir(entry.traceFamily);
  if (entry.traceFamilyDir !== expectedFamilyDir) {
    return `slot ${entry.slotId} traceFamilyDir mismatch: expected ${expectedFamilyDir}, received ${entry.traceFamilyDir}`;
  }
  if (entry.fixtureId !== buildFixtureId(entry.slotId)) {
    return `slot ${entry.slotId} fixtureId mismatch: expected ${buildFixtureId(entry.slotId)}, received ${entry.fixtureId}`;
  }
  return null;
}

export function validateFrozenRecordedSessionFixtureManifest(
  manifestPath: string = DEFAULT_FIXTURE_MANIFEST_PATH,
): FrozenRecordedSessionFixtureManifestValidationV1 {
  const resolvedManifestPath = path.resolve(manifestPath);
  const errors: string[] = [];
  const manifest = readJsonFile<FrozenRecordedSessionFixtureManifestV1>(resolvedManifestPath);
  if (manifest.contract !== FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_CONTRACT) {
    errors.push(`unsupported manifest contract: ${manifest.contract}`);
  }
  if (manifest.status !== MANIFEST_STATUS) {
    errors.push(`unsupported manifest status: ${manifest.status}`);
  }
  if (manifest.fixtureContract !== RECORDED_SESSION_FIXTURE_CONTRACT) {
    errors.push(`unsupported fixture contract: ${manifest.fixtureContract}`);
  }
  const manifestDir = path.dirname(resolvedManifestPath);
  const traceManifestPath = path.resolve(manifestDir, manifest.traceManifest.path);
  const sourceManifest = loadCanonicalTraceManifest(traceManifestPath);

  if (manifest.traceManifest.contract !== CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT) {
    errors.push(`traceManifest.contract mismatch: ${manifest.traceManifest.contract}`);
  }
  if (manifest.traceManifest.setId !== sourceManifest.setId) {
    errors.push(`traceManifest.setId mismatch: expected ${sourceManifest.setId}, received ${manifest.traceManifest.setId}`);
  }
  if (manifest.traceManifest.traceContract !== sourceManifest.traceContract) {
    errors.push(
      `traceManifest.traceContract mismatch: expected ${sourceManifest.traceContract}, received ${manifest.traceManifest.traceContract}`,
    );
  }
  if (manifest.traceManifest.traceCount !== sourceManifest.traceCount) {
    errors.push(`traceManifest.traceCount mismatch: expected ${sourceManifest.traceCount}, received ${manifest.traceManifest.traceCount}`);
  }
  if (manifest.entries.length !== sourceManifest.entries.length) {
    errors.push(`entry count mismatch: expected ${sourceManifest.entries.length}, received ${manifest.entries.length}`);
  }

  const seenSlotIds = new Set<string>();
  const seenFixtureIds = new Set<string>();
  const seenTracePaths = new Set<string>();
  const seenFixturePaths = new Set<string>();
  const observedCounts = Object.fromEntries(TRACE_FAMILY_ORDER.map((family) => [family, 0])) as Record<TraceFamily, number>;
  const sourceEntries = new Map(sourceManifest.entries.map((entry) => [entry.slotId, entry]));

  for (const entry of manifest.entries) {
    if (seenSlotIds.has(entry.slotId)) {
      errors.push(`duplicate slotId: ${entry.slotId}`);
    }
    seenSlotIds.add(entry.slotId);
    if (seenFixtureIds.has(entry.fixtureId)) {
      errors.push(`duplicate fixtureId: ${entry.fixtureId}`);
    }
    seenFixtureIds.add(entry.fixtureId);
    if (seenTracePaths.has(entry.tracePath)) {
      errors.push(`duplicate tracePath: ${entry.tracePath}`);
    }
    seenTracePaths.add(entry.tracePath);
    if (seenFixturePaths.has(entry.fixturePath)) {
      errors.push(`duplicate fixturePath: ${entry.fixturePath}`);
    }
    seenFixturePaths.add(entry.fixturePath);

    observedCounts[entry.traceFamily] += 1;

    const pathError = validateExpectedPath(entry);
    if (pathError) {
      errors.push(pathError);
    }

    const sourceEntry = sourceEntries.get(entry.slotId);
    if (!sourceEntry) {
      errors.push(`slot ${entry.slotId} missing from source trace manifest`);
      continue;
    }
    if (entry.title !== sourceEntry.title) {
      errors.push(`slot ${entry.slotId} title mismatch`);
    }
    if (entry.traceFamily !== sourceEntry.category) {
      errors.push(`slot ${entry.slotId} traceFamily mismatch: expected ${sourceEntry.category}, received ${entry.traceFamily}`);
    }
    if (entry.metadata.sourceKind !== sourceEntry.sourceKind) {
      errors.push(`slot ${entry.slotId} sourceKind mismatch`);
    }
    if (canonicalJson(entry.metadata.sourcePaths) !== canonicalJson(sourceEntry.sourcePaths)) {
      errors.push(`slot ${entry.slotId} sourcePaths mismatch`);
    }
    if (canonicalJson(entry.metadata.tags) !== canonicalJson(sourceEntry.tags)) {
      errors.push(`slot ${entry.slotId} tags mismatch`);
    }
    if (canonicalJson(entry.metadata.notes) !== canonicalJson(sourceEntry.notes)) {
      errors.push(`slot ${entry.slotId} notes mismatch`);
    }
    if (entry.metadata.sourceTraceStatus !== sourceEntry.status) {
      errors.push(`slot ${entry.slotId} sourceTraceStatus mismatch`);
    }
    if (entry.metadata.realTraceSourceAvailable !== sourceEntry.realTraceSourceAvailable) {
      errors.push(`slot ${entry.slotId} realTraceSourceAvailable mismatch`);
    }
    if (
      entry.metadata.sanitization.classification !== sourceEntry.sanitization.classification
      || entry.metadata.sanitization.redactionRequired !== sourceEntry.sanitization.redactionRequired
      || canonicalJson(entry.metadata.sanitization.notes) !== canonicalJson(sourceEntry.sanitization.notes)
    ) {
      errors.push(`slot ${entry.slotId} sanitization metadata mismatch`);
    }
    if (canonicalJson(entry.metadata.shape) !== canonicalJson(sourceEntry.shape)) {
      errors.push(`slot ${entry.slotId} shape metadata mismatch`);
    }

    const absoluteTracePath = path.resolve(manifestDir, entry.tracePath);
    const trace = loadTrace(absoluteTracePath);
    const expectedTraceHash = checksumJsonPayload(trace);
    if (entry.traceHash !== expectedTraceHash) {
      errors.push(`slot ${entry.slotId} traceHash mismatch: expected ${expectedTraceHash}, received ${entry.traceHash}`);
    }
    const expectedSelectionHash = buildSelectionHash({
      fixtureId: entry.fixtureId,
      slotId: entry.slotId,
      title: entry.title,
      traceFamily: entry.traceFamily,
      traceFamilyDir: entry.traceFamilyDir,
      tracePath: entry.tracePath,
      traceHash: entry.traceHash,
      fixturePath: entry.fixturePath,
      metadata: entry.metadata,
    });
    if (entry.selectionHash !== expectedSelectionHash) {
      errors.push(
        `slot ${entry.slotId} selectionHash mismatch: expected ${expectedSelectionHash}, received ${entry.selectionHash}`,
      );
    }
    if (entry.fixtureHash !== null) {
      errors.push(`slot ${entry.slotId} fixtureHash must remain null for scaffold-only manifest`);
    }
    if (entry.fixtureHashStatus !== "pending_materialization") {
      errors.push(
        `slot ${entry.slotId} fixtureHashStatus mismatch: expected pending_materialization, received ${entry.fixtureHashStatus}`,
      );
    }
  }

  if (canonicalJson(manifest.traceFamilyOrder) !== canonicalJson(sourceManifest.categoryOrder)) {
    errors.push("traceFamilyOrder mismatch with source trace manifest");
  }
  for (const family of TRACE_FAMILY_ORDER) {
    const expectedCount = sourceManifest.categoryCounts[family];
    if (manifest.traceFamilyCounts[family] !== expectedCount) {
      errors.push(`traceFamilyCounts.${family} mismatch: expected ${expectedCount}, received ${manifest.traceFamilyCounts[family]}`);
    }
    if (observedCounts[family] !== expectedCount) {
      errors.push(`observed ${family} count mismatch: expected ${expectedCount}, received ${observedCounts[family]}`);
    }
  }
  if (manifest.materialization.fixtureFilesCheckedIn !== 0) {
    errors.push(`fixtureFilesCheckedIn must remain 0 for scaffold-only manifest, received ${manifest.materialization.fixtureFilesCheckedIn}`);
  }
  if (manifest.materialization.pendingFixtureCount !== sourceManifest.traceCount) {
    errors.push(
      `pendingFixtureCount mismatch: expected ${sourceManifest.traceCount}, received ${manifest.materialization.pendingFixtureCount}`,
    );
  }

  return {
    contract: FROZEN_RECORDED_SESSION_FIXTURE_MANIFEST_VALIDATION_CONTRACT,
    ok: errors.length === 0,
    manifestPath: resolvedManifestPath,
    traceManifestPath,
    entryCount: manifest.entries.length,
    errors,
  };
}

export function writeFrozenRecordedSessionFixtureManifest(
  input: WriteFrozenRecordedSessionFixtureManifestInput = {},
): {
  manifestPath: string;
  schemaPath: string;
  manifest: FrozenRecordedSessionFixtureManifestV1;
  schema: Record<string, unknown>;
} {
  const manifestPath = path.resolve(input.outputManifestPath ?? DEFAULT_FIXTURE_MANIFEST_PATH);
  const schemaPath = path.resolve(input.schemaOutputPath ?? DEFAULT_FIXTURE_SCHEMA_PATH);
  const manifest = buildFrozenRecordedSessionFixtureManifest({
    traceManifestPath: input.traceManifestPath,
    outputManifestPath: manifestPath,
    generatedAt: input.generatedAt,
  });
  const schema = buildFrozenRecordedSessionFixtureManifestSchema();
  writeJsonFile(manifestPath, manifest);
  writeJsonFile(schemaPath, schema);
  return {
    manifestPath,
    schemaPath,
    manifest,
    schema,
  };
}

function main(): void {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.command === "write") {
    const result = writeFrozenRecordedSessionFixtureManifest({
      traceManifestPath: parsed.traceManifestPath,
      outputManifestPath: parsed.manifestPath,
      schemaOutputPath: parsed.schemaPath,
      generatedAt: normalizeCliString(parsed.generatedAt) ?? DEFAULT_GENERATED_AT,
    });
    process.stdout.write(`wrote manifest ${result.manifestPath}\n`);
    process.stdout.write(`wrote schema ${result.schemaPath}\n`);
    return;
  }

  const validation = validateFrozenRecordedSessionFixtureManifest(parsed.manifestPath);
  process.stdout.write(canonicalJson(validation) + "\n");
  if (!validation.ok) {
    process.exitCode = 1;
  }
}

const entrypointPath = process.argv[1] ? path.resolve(process.argv[1]) : null;
if (entrypointPath && entrypointPath === __filename) {
  main();
}
