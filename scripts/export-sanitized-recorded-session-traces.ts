#!/usr/bin/env tsx
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

type Role = "user" | "assistant";

type SourceRef = {
  role: Role;
  contains: string;
  occurrence?: number;
};

type SeedCueSpec = SourceRef & {
  cueId: string;
  content: string;
  kind?: "teaching" | "approval";
};

type FeedbackSpec = SourceRef & {
  content: string;
  kind?: "teaching" | "approval";
};

type TurnSpec = {
  turnId: string;
  userSource: SourceRef;
  userMessage: string;
  runtimeHints?: string[];
  deliveredSource?: SourceRef;
  feedback?: FeedbackSpec[];
  expectedContextPhrases: string[];
  minimumPhraseHits?: number;
};

type TraceSpec = {
  traceId: string;
  sessionFile: string;
  sessionId: string;
  channel: string;
  sourceStream: string;
  agentId?: string;
  privacyNotes: string[];
  workspace: {
    workspaceId: string;
    snapshotId: string;
    rootDir: string;
    branch?: string;
    revision: string;
    labels?: string[];
  };
  evalTurnCount?: number;
  seedCues: SeedCueSpec[];
  turns: TurnSpec[];
  outputFile: string;
};

type ExportSpec = {
  contract: "sanitized_recorded_session_trace_export_spec.v1";
  description?: string;
  traces: TraceSpec[];
};

type SessionMessage = {
  role: Role;
  timestamp: string;
  text: string;
};

type MatchedRef = {
  role: Role;
  contains: string;
  occurrence: number;
  timestamp: string;
  excerpt: string;
};

function usage(): never {
  console.error("Usage: tsx scripts/export-sanitized-recorded-session-traces.ts [--spec <path>] [--out-dir <path>]");
  process.exit(1);
}

function parseArgs(argv: string[]) {
  const args = {
    specPath: path.resolve("artifacts/recorded-session-traces/2026-03-28/spec.json"),
    outDir: path.resolve("artifacts/recorded-session-traces/2026-03-28/generated")
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--spec") {
      args.specPath = path.resolve(argv[++i] ?? usage());
    } else if (arg === "--out-dir") {
      args.outDir = path.resolve(argv[++i] ?? usage());
    } else {
      usage();
    }
  }
  return args;
}

function readJson<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function loadSessionMessages(sessionFile: string): SessionMessage[] {
  const rows = readFileSync(sessionFile, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as any);
  const messages: SessionMessage[] = [];
  for (const row of rows) {
    if (row.type !== "message") continue;
    const message = row.message;
    const role = message?.role;
    if (role !== "user" && role !== "assistant") continue;
    const text = (message.content ?? [])
      .filter((part: any) => part?.type === "text")
      .map((part: any) => String(part.text ?? ""))
      .join("\n")
      .trim();
    if (text.length === 0) continue;
    messages.push({ role, timestamp: String(row.timestamp), text });
  }
  return messages;
}

function findMessage(messages: SessionMessage[], ref: SourceRef): MatchedRef {
  const occurrence = ref.occurrence ?? 1;
  let seen = 0;
  for (const message of messages) {
    if (message.role !== ref.role) continue;
    if (!message.text.includes(ref.contains)) continue;
    seen += 1;
    if (seen === occurrence) {
      return {
        role: ref.role,
        contains: ref.contains,
        occurrence,
        timestamp: message.timestamp,
        excerpt: message.text.slice(0, 220)
      };
    }
  }
  throw new Error(`Could not find ${ref.role} message containing ${JSON.stringify(ref.contains)} (occurrence ${occurrence})`);
}

function addMinutes(iso: string, minutes: number): string {
  return new Date(new Date(iso).getTime() + minutes * 60_000).toISOString();
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(value, null, 2) + "\n";
}

function main() {
  const { specPath, outDir } = parseArgs(process.argv.slice(2));
  const spec = readJson<ExportSpec>(specPath);
  if (spec.contract !== "sanitized_recorded_session_trace_export_spec.v1") {
    throw new Error("Unexpected export spec contract");
  }

  const sessionRoot = path.join(os.homedir(), ".openclaw/agents/main/sessions");
  mkdirSync(outDir, { recursive: true });

  const provenance = {
    contract: "sanitized_recorded_session_trace_export_report.v1",
    generatedAt: new Date().toISOString(),
    specPath,
    outDir,
    traces: [] as any[]
  };

  for (const traceSpec of spec.traces) {
    const sessionFile = path.join(sessionRoot, traceSpec.sessionFile);
    const messages = loadSessionMessages(sessionFile);

    const seedMatches = traceSpec.seedCues.map((cue) => ({ cue, match: findMessage(messages, cue) }));
    const turnMatches = traceSpec.turns.map((turn) => ({
      turn,
      user: findMessage(messages, turn.userSource),
      delivered: turn.deliveredSource ? findMessage(messages, turn.deliveredSource) : null,
      feedback: (turn.feedback ?? []).map((feedback) => ({ feedback, match: findMessage(messages, feedback) }))
    }));

    const allTimestamps = [
      ...seedMatches.map(({ match }) => match.timestamp),
      ...turnMatches.flatMap(({ user, delivered, feedback }) => [
        user.timestamp,
        ...(delivered ? [delivered.timestamp] : []),
        ...feedback.map(({ match }) => match.timestamp)
      ])
    ].sort();

    const recordedAt = allTimestamps[0];
    const lastObservedAt = allTimestamps[allTimestamps.length - 1];
    const earliestSeedAt = seedMatches.map(({ match }) => match.timestamp).sort()[0] ?? recordedAt;

    const trace = {
      contract: "recorded_session_trace.v1",
      traceId: traceSpec.traceId,
      source: "sanitized_recorded_session",
      recordedAt,
      bundleBuiltAt: addMinutes(lastObservedAt, 1),
      agentId: traceSpec.agentId ?? "main",
      sessionId: traceSpec.sessionId,
      channel: traceSpec.channel,
      sourceStream: traceSpec.sourceStream,
      privacy: {
        sanitized: true,
        notes: traceSpec.privacyNotes
      },
      workspace: {
        ...traceSpec.workspace,
        capturedAt: recordedAt
      },
      ...(traceSpec.evalTurnCount !== undefined ? { evalTurnCount: traceSpec.evalTurnCount } : {}),
      seedBuiltAt: earliestSeedAt,
      seedActivatedAt: addMinutes(earliestSeedAt, 1),
      seedCues: seedMatches.map(({ cue, match }) => ({
        cueId: cue.cueId,
        createdAt: match.timestamp,
        content: cue.content,
        ...(cue.kind ? { kind: cue.kind } : {})
      })),
      turns: turnMatches.map(({ turn, user, delivered, feedback }) => ({
        turnId: turn.turnId,
        createdAt: user.timestamp,
        ...(delivered ? { deliveredAt: delivered.timestamp } : {}),
        userMessage: turn.userMessage,
        ...(turn.runtimeHints ? { runtimeHints: turn.runtimeHints } : {}),
        ...(feedback.length > 0
          ? {
              feedback: feedback.map(({ feedback, match }) => ({
                createdAt: match.timestamp,
                content: feedback.content,
                ...(feedback.kind ? { kind: feedback.kind } : {})
              }))
            }
          : {}),
        expectedContextPhrases: turn.expectedContextPhrases,
        ...(turn.minimumPhraseHits !== undefined ? { minimumPhraseHits: turn.minimumPhraseHits } : {})
      }))
    };

    const outputPath = path.join(outDir, traceSpec.outputFile);
    mkdirSync(path.dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, canonicalJson(trace), "utf8");

    provenance.traces.push({
      traceId: traceSpec.traceId,
      sessionFile,
      outputPath,
      recordedAt,
      bundleBuiltAt: trace.bundleBuiltAt,
      seedCues: seedMatches.map(({ cue, match }) => ({ cueId: cue.cueId, content: cue.content, source: match })),
      turns: turnMatches.map(({ turn, user, delivered, feedback }) => ({
        turnId: turn.turnId,
        userMessage: turn.userMessage,
        userSource: user,
        deliveredSource: delivered,
        feedback: feedback.map(({ feedback, match }) => ({ content: feedback.content, kind: feedback.kind ?? null, source: match })),
        expectedContextPhrases: turn.expectedContextPhrases
      }))
    });
  }

  writeFileSync(path.join(outDir, "export-report.json"), canonicalJson(provenance), "utf8");
  console.log(`wrote ${provenance.traces.length} sanitized recorded-session trace(s) to ${outDir}`);
}

main();
