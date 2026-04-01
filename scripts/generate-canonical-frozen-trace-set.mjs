#!/usr/bin/env node

import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const outputRoot = path.join(repoRoot, "evals", "recorded-session-replay", "canonical-frozen-20");

const CATEGORY_ORDER = [
  "direct_answer",
  "plan_execution",
  "retrieval_memory_heavy",
  "correction_follow_up_heavy",
];

const CATEGORY_DIRS = {
  direct_answer: "direct-answer",
  plan_execution: "plan-execution",
  retrieval_memory_heavy: "retrieval-memory-heavy",
  correction_follow_up_heavy: "correction-follow-up-heavy",
};

function cue(cueId, createdAt, content, kind = "teaching") {
  return { cueId, createdAt, content, kind };
}

function feedback(createdAt, content, kind = "teaching") {
  return { createdAt, content, kind };
}

function turn(turnId, createdAt, deliveredAt, userMessage, expectedContextPhrases, extra = {}) {
  return {
    turnId,
    createdAt,
    deliveredAt,
    userMessage,
    ...(extra.runtimeHints ? { runtimeHints: extra.runtimeHints } : {}),
    ...(extra.feedback ? { feedback: extra.feedback } : {}),
    expectedContextPhrases,
    ...(extra.minimumPhraseHits != null ? { minimumPhraseHits: extra.minimumPhraseHits } : {}),
  };
}

function workspaceFor(traceId, capturedAt, opts = {}) {
  const slug = traceId.replace(/[^a-z0-9-]/gi, "-").toLowerCase();
  return {
    workspaceId: opts.workspaceId ?? `ws-${slug}`,
    snapshotId: opts.snapshotId ?? `snapshot-${slug}`,
    capturedAt,
    rootDir: opts.rootDir ?? `/tmp/openclawbrain-canonical-frozen-20/${slug}`,
    ...(opts.branch ? { branch: opts.branch } : {}),
    revision: opts.revision ?? `rev-${slug}`,
    ...(opts.labels ? { labels: opts.labels } : {}),
  };
}

function buildTrace({
  traceId,
  recordedAt,
  bundleBuiltAt,
  sessionId,
  channel = "cli",
  sourceStream,
  privacyNotes,
  workspace,
  seedBuiltAt,
  seedActivatedAt,
  seedCues,
  turns,
  evalTurnCount,
  agentId,
}) {
  return {
    contract: "recorded_session_trace.v1",
    traceId,
    source: "sanitized_recorded_session",
    recordedAt,
    bundleBuiltAt,
    ...(agentId ? { agentId } : {}),
    sessionId,
    channel,
    sourceStream: sourceStream ?? `openclaw/runtime/${channel}`,
    privacy: {
      sanitized: true,
      notes: privacyNotes,
    },
    workspace,
    ...(evalTurnCount != null ? { evalTurnCount } : {}),
    seedBuiltAt,
    seedActivatedAt,
    seedCues,
    turns,
  };
}

const exactTrainFreezeEval = buildTrace({
  traceId: "trace-train-freeze-eval",
  recordedAt: "2026-03-25T10:00:00.000Z",
  bundleBuiltAt: "2026-03-25T10:30:00.000Z",
  sessionId: "session-train-freeze-eval",
  channel: "chat",
  sourceStream: "recorded/session",
  privacyNotes: ["test"],
  workspace: {
    workspaceId: "workspace-train-freeze-eval",
    snapshotId: "snapshot-train-freeze-eval",
    capturedAt: "2026-03-25T09:55:00.000Z",
    rootDir: "/tmp/workspace-train-freeze-eval",
    revision: "rev-train-freeze-eval",
  },
  evalTurnCount: 2,
  seedBuiltAt: "2026-03-25T09:56:00.000Z",
  seedActivatedAt: "2026-03-25T09:57:00.000Z",
  seedCues: [
    cue("cue-1", "2026-03-25T09:50:00.000Z", "Always read README before editing code."),
  ],
  turns: [
    turn(
      "turn-1",
      "2026-03-25T10:01:00.000Z",
      "2026-03-25T10:01:30.000Z",
      "What should I read before editing?",
      ["readme before editing"],
      {
        feedback: [
          feedback("2026-03-25T10:01:45.000Z", "Correct: read README before editing code.", "approval"),
        ],
      },
    ),
    turn(
      "turn-2",
      "2026-03-25T10:05:00.000Z",
      "2026-03-25T10:05:30.000Z",
      "Before changing files, what is the rule?",
      ["readme before editing"],
    ),
    turn(
      "turn-3",
      "2026-03-25T10:10:00.000Z",
      "2026-03-25T10:10:30.000Z",
      "What must happen before editing code?",
      ["readme before editing"],
    ),
  ],
});

const exactOpenclawReplayFreezeIdentity = buildTrace({
  traceId: "trace-openclaw-replay-freeze-identity",
  recordedAt: "2026-03-25T10:00:00.000Z",
  bundleBuiltAt: "2026-03-25T10:30:00.000Z",
  sessionId: "session-openclaw-replay-freeze-identity",
  channel: "chat",
  sourceStream: "recorded/session",
  privacyNotes: ["test"],
  workspace: {
    workspaceId: "workspace-openclaw-replay-freeze-identity",
    snapshotId: "snapshot-openclaw-replay-freeze-identity",
    capturedAt: "2026-03-25T09:55:00.000Z",
    rootDir: "/tmp/workspace-openclaw-replay-freeze-identity",
    revision: "rev-openclaw-replay-freeze-identity",
  },
  evalTurnCount: 2,
  seedBuiltAt: "2026-03-25T09:56:00.000Z",
  seedActivatedAt: "2026-03-25T09:57:00.000Z",
  seedCues: [
    cue("cue-1", "2026-03-25T09:50:00.000Z", "Always read README before editing code."),
  ],
  turns: [
    turn(
      "turn-1",
      "2026-03-25T10:01:00.000Z",
      "2026-03-25T10:01:30.000Z",
      "What should I read before editing?",
      ["readme before editing"],
      {
        feedback: [
          feedback("2026-03-25T10:01:45.000Z", "Correct: read README before editing code.", "approval"),
        ],
      },
    ),
    turn(
      "turn-2",
      "2026-03-25T10:05:00.000Z",
      "2026-03-25T10:05:30.000Z",
      "Before changing files, what is the rule?",
      ["readme before editing"],
    ),
    turn(
      "turn-3",
      "2026-03-25T10:10:00.000Z",
      "2026-03-25T10:10:30.000Z",
      "What must happen before editing code?",
      ["readme before editing"],
    ),
  ],
});

const exactComparativeReplay = buildTrace({
  traceId: "trace-comparative-replay",
  recordedAt: "2026-03-25T00:00:00.000Z",
  bundleBuiltAt: "2026-03-25T00:10:00.000Z",
  agentId: "agent",
  sessionId: "session-comparative-replay",
  channel: "cli",
  sourceStream: "recorded/session",
  privacyNotes: ["test fixture"],
  workspace: {
    workspaceId: "ws-comparative-replay",
    snapshotId: "snapshot-comparative-replay",
    capturedAt: "2026-03-25T00:00:00.000Z",
    rootDir: "/tmp/workspace-comparative-replay",
    branch: "main",
    revision: "rev-comparative-replay",
    labels: ["test"],
  },
  seedBuiltAt: "2026-03-25T00:01:00.000Z",
  seedActivatedAt: "2026-03-25T00:02:00.000Z",
  seedCues: [
    cue("cue-routing-guide", "2026-03-25T00:00:30.000Z", "The routing guide lives here."),
  ],
  turns: [
    turn(
      "turn-1",
      "2026-03-25T00:03:00.000Z",
      "2026-03-25T00:03:30.000Z",
      "show the routing guide",
      ["routing guide"],
      {
        runtimeHints: ["routing", "guide"],
        feedback: [feedback("2026-03-25T00:03:45.000Z", "Keep the routing guide easy to find.")],
      },
    ),
    turn(
      "turn-2",
      "2026-03-25T00:05:00.000Z",
      "2026-03-25T00:05:30.000Z",
      "show the routing guide again",
      ["routing guide"],
      {
        runtimeHints: ["routing", "guide", "again"],
        feedback: [
          feedback("2026-03-25T00:05:45.000Z", "The routing guide is still the right answer.", "approval"),
        ],
      },
    ),
  ],
});

const normalizedScoreResolution = buildTrace({
  traceId: "trace-score-resolution",
  recordedAt: "2026-03-28T17:26:18.111Z",
  bundleBuiltAt: "2026-03-28T20:02:41.227Z",
  agentId: "main",
  sessionId: "sanitized-session-score-resolution",
  channel: "telegram",
  sourceStream: "telegram/direct/proof-plan",
  privacyNotes: ["normalized dynamic test fixture from packages/cli/dist/test/replay-score-resolution.test.js"],
  workspace: workspaceFor("trace-score-resolution", "2026-03-28T17:26:18.111Z", {
    branch: "main",
    labels: ["proof-plan"],
  }),
  evalTurnCount: 1,
  seedBuiltAt: "2026-03-28T17:26:18.111Z",
  seedActivatedAt: "2026-03-28T17:27:18.111Z",
  seedCues: [
    cue(
      "cue-proof-run",
      "2026-03-28T17:26:18.111Z",
      "T-20260328-040 is the real-trace learned-route proof run for OpenClawBrain.",
    ),
  ],
  turns: [
    turn(
      "plan-turn-1",
      "2026-03-28T19:54:42.389Z",
      "2026-03-28T19:54:57.546Z",
      "So what's next for us to work on?",
      ["learned-route proof run"],
      {
        runtimeHints: ["proof", "next", "plan"],
        feedback: [
          feedback(
            "2026-03-28T19:54:57.546Z",
            "Next steps: commit the rollout-evaluator scaffold, export a sanitized non-test real-trace replay corpus, and rerun the rollout evaluator.",
          ),
        ],
      },
    ),
    turn(
      "plan-turn-2",
      "2026-03-28T19:56:06.301Z",
      "2026-03-28T19:58:48.383Z",
      "Please make an end-to-end master plan for me.",
      ["real-trace learned-route proof run"],
      {
        runtimeHints: ["master-plan", "proof", "roadmap"],
        feedback: [
          feedback("2026-03-28T19:58:48.383Z", "The master plan ends with a rollout verdict: ready, limited, or blocked."),
        ],
      },
    ),
    turn(
      "plan-turn-3",
      "2026-03-28T20:01:14.399Z",
      "2026-03-28T20:01:41.227Z",
      "Please do this entire plan end to end.",
      ["sanitized non-test real-trace replay corpus", "ready, limited, or blocked"],
      {
        runtimeHints: ["execute", "proof", "end-to-end"],
      },
    ),
  ],
});

const normalizedSeedCarryForward = buildTrace({
  traceId: "trace-seed-carry-forward",
  recordedAt: "2026-03-29T00:00:00.000Z",
  bundleBuiltAt: "2026-03-29T00:10:00.000Z",
  agentId: "agent",
  sessionId: "session-seed-carry-forward",
  channel: "cli",
  sourceStream: "recorded/session",
  privacyNotes: [
    "normalized dynamic test fixture from packages/cli/dist/test/learned-route-seed-carry-forward.test.js",
  ],
  workspace: workspaceFor("trace-seed-carry-forward", "2026-03-29T00:00:00.000Z", {
    branch: "main",
    labels: ["real"],
  }),
  seedBuiltAt: "2026-03-29T00:01:00.000Z",
  seedActivatedAt: "2026-03-29T00:02:00.000Z",
  seedCues: [
    cue(
      "cue-proof-run",
      "2026-03-29T00:00:30.000Z",
      "T-20260329-048 is the real-trace learned-route proof run for OpenClawBrain.",
    ),
  ],
  turns: [
    turn(
      "turn-1",
      "2026-03-29T00:03:00.000Z",
      "2026-03-29T00:03:30.000Z",
      "start the learned-route proof run",
      ["learned-route proof run"],
      {
        runtimeHints: ["proof", "run"],
        feedback: [
          feedback(
            "2026-03-29T00:03:45.000Z",
            "Next steps: commit the rollout evaluator scaffold and rerun the rollout evaluator.",
          ),
        ],
      },
    ),
    turn(
      "turn-2",
      "2026-03-29T00:05:00.000Z",
      "2026-03-29T00:05:30.000Z",
      "what is the real-trace learned-route proof run?",
      ["real-trace learned-route proof run"],
      {
        runtimeHints: ["real-trace", "proof", "run"],
      },
    ),
  ],
});

const normalizedSeedCarryForwardEvalDedup = buildTrace({
  traceId: "trace-seed-carry-forward-eval-dedup",
  recordedAt: "2026-03-29T00:00:00.000Z",
  bundleBuiltAt: "2026-03-29T00:10:00.000Z",
  agentId: "agent",
  sessionId: "session-seed-carry-forward-eval",
  channel: "telegram",
  sourceStream: "telegram/direct/live-proof-story",
  privacyNotes: [
    "normalized dynamic test fixture from packages/cli/dist/test/learned-route-seed-carry-forward.test.js",
  ],
  workspace: workspaceFor("trace-seed-carry-forward-eval-dedup", "2026-03-29T00:00:00.000Z", {
    branch: "main",
    labels: ["real"],
  }),
  evalTurnCount: 1,
  seedBuiltAt: "2026-03-29T00:01:00.000Z",
  seedActivatedAt: "2026-03-29T00:02:00.000Z",
  seedCues: [
    cue(
      "cue-empty-memory-scaffold",
      "2026-03-29T00:00:30.000Z",
      "OpenClawBrain starts as a correctly attached but mostly empty memory scaffold.",
    ),
  ],
  turns: [
    turn(
      "story-turn-1",
      "2026-03-29T00:03:00.000Z",
      "2026-03-29T00:03:30.000Z",
      "Use our current brain install to make it concrete.",
      ["memory scaffold"],
      {
        runtimeHints: ["install", "story", "proof"],
        feedback: [
          feedback("2026-03-29T00:03:45.000Z", "Use the current shared ~/.openclaw install as the canonical example."),
        ],
      },
    ),
    turn(
      "story-turn-2",
      "2026-03-29T00:05:00.000Z",
      "2026-03-29T00:05:30.000Z",
      "This is not good enough. Please make a deep detailed plan.",
      ["memory scaffold"],
      {
        runtimeHints: ["rewrite", "proof-signals", "detail"],
        feedback: [
          feedback(
            "2026-03-29T00:05:45.000Z",
            "Ground the story in real host proof signals: BRAIN LOADED, loadProof=status_probe_ready, serveState=serving_active_pack, routeFn available=yes.",
          ),
        ],
      },
    ),
    turn(
      "story-turn-3",
      "2026-03-29T00:07:00.000Z",
      "2026-03-29T00:07:30.000Z",
      "Make it good and push it live now!",
      ["BRAIN LOADED", "routeFn available=yes"],
      {
        runtimeHints: ["publish", "live", "proof"],
      },
    ),
  ],
});

const exactTernRecordedSessionProof = buildTrace({
  traceId: "tern-recorded-session-proof",
  recordedAt: "2026-03-25T00:00:00.000Z",
  bundleBuiltAt: "2026-03-25T00:30:00.000Z",
  sessionId: "session-tern-proof",
  channel: "cli",
  sourceStream: "openclaw/runtime/cli",
  privacyNotes: ["synthetic fixture for deterministic proof bundle tests"],
  workspace: {
    workspaceId: "workspace-tern",
    snapshotId: "snapshot-tern",
    capturedAt: "2026-03-24T23:59:00.000Z",
    rootDir: "/workspace/tern",
    branch: "task/T-20260325-031-wave3-proof-bundle-writer",
    revision: "109e9b3",
    labels: ["proof", "recorded-session"],
  },
  seedBuiltAt: "2026-03-25T00:05:00.000Z",
  seedActivatedAt: "2026-03-25T00:06:00.000Z",
  seedCues: [
    cue(
      "cue-deploy-journal",
      "2026-03-25T00:01:00.000Z",
      "The operator lane restart checklist is archived in docs/evidence and incidents are tagged with postmortem IDs.",
    ),
    cue(
      "cue-restart-order",
      "2026-03-25T00:02:00.000Z",
      "Keep the operator lane restart order explicit when proving a recorded session replay.",
    ),
  ],
  turns: [
    turn(
      "turn-alpha",
      "2026-03-25T00:10:00.000Z",
      "2026-03-25T00:10:30.000Z",
      "Where is the restart checklist archived and how are incidents tagged?",
      ["docs/evidence", "postmortem IDs"],
      {
        feedback: [
          feedback("2026-03-25T00:11:00.000Z", "Answer with docs/evidence and postmortem IDs.", "correction"),
        ],
        minimumPhraseHits: 1,
      },
    ),
    turn(
      "turn-beta",
      "2026-03-25T00:20:00.000Z",
      "2026-03-25T00:20:20.000Z",
      "Summarize the operator lane restart order.",
      ["operator lane", "restart order"],
      {
        runtimeHints: ["proof", "replay"],
        feedback: [feedback("2026-03-25T00:21:00.000Z", "Keep the operator lane restart order explicit.")],
        minimumPhraseHits: 1,
      },
    ),
  ],
});

const definitions = [
  {
    slotId: "direct-answer-01",
    title: "README-before-editing rule recall",
    category: "direct_answer",
    sourceKind: "repo_published_fixture",
    sourcePaths: [
      "docs/evidence/2026-03-26/0ca08242290103617b5bcaa2f80522d0124fc53d/recorded-session-replay/trace-train-freeze-eval/trace.json",
    ],
    tags: ["readme", "rule-recall"],
    notes: [
      "Published proof-bundle trace checked into docs/evidence. The source itself is a synthetic replay fixture, not a verified production session.",
    ],
    trace: exactTrainFreezeEval,
  },
  {
    slotId: "direct-answer-02",
    title: "Frozen identity rule recall",
    category: "direct_answer",
    sourceKind: "repo_test_fixture_static",
    sourcePaths: ["packages/openclaw/dist/test/replay-freeze-identity.test.js"],
    tags: ["rule-recall", "frozen-identity"],
    notes: ["Exactly transcribed from the checked-in OpenClaw package test fixture."],
    trace: exactOpenclawReplayFreezeIdentity,
  },
  {
    slotId: "direct-answer-03",
    title: "Proof bundle layout answer",
    category: "direct_answer",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["docs/internal/recorded-session-replay.md"],
    tags: ["proof-layout", "single-answer"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent derived from the fixed proof-bundle layout doc."],
    trace: buildTrace({
      traceId: "trace-direct-answer-proof-bundle-layout",
      recordedAt: "2026-04-01T01:00:00.000Z",
      bundleBuiltAt: "2026-04-01T01:10:00.000Z",
      sessionId: "session-direct-answer-proof-bundle-layout",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from docs/internal/recorded-session-replay.md"],
      workspace: workspaceFor("trace-direct-answer-proof-bundle-layout", "2026-04-01T00:59:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T01:01:00.000Z",
      seedActivatedAt: "2026-04-01T01:02:00.000Z",
      seedCues: [
        cue(
          "cue-layout",
          "2026-04-01T00:58:30.000Z",
          "A recorded-session replay proof bundle uses a fixed layout including manifest.json, trace.json, fixture.json, bundle.json, environment.json, summary.md, summary-tables.json, coverage-snapshot.json, hardening-snapshot.json, hashes.json, and per-mode JSON files.",
        ),
      ],
      turns: [
        turn(
          "bundle-layout-turn-1",
          "2026-04-01T01:03:00.000Z",
          "2026-04-01T01:03:30.000Z",
          "Which files are always in the recorded-session replay proof bundle?",
          ["manifest.json", "bundle.json"],
          {
            runtimeHints: ["proof", "layout"],
            feedback: [
              feedback(
                "2026-04-01T01:03:45.000Z",
                "Keep the answer concrete: mention manifest.json, bundle.json, and a per-mode file.",
              ),
            ],
          },
        ),
        turn(
          "bundle-layout-turn-2",
          "2026-04-01T01:05:00.000Z",
          "2026-04-01T01:05:20.000Z",
          "Name the replay proof bundle files again.",
          ["manifest.json", "bundle.json", "modes/learned_route.json"],
          {
            runtimeHints: ["proof", "layout", "again"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "direct-answer-04",
    title: "Proof rerun command answer",
    category: "direct_answer",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["docs/reproduce-eval.md"],
    tags: ["command", "proof-rerun"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the checked-in reproduction command."],
    trace: buildTrace({
      traceId: "trace-direct-answer-reproduce-eval-command",
      recordedAt: "2026-04-01T01:20:00.000Z",
      bundleBuiltAt: "2026-04-01T01:30:00.000Z",
      sessionId: "session-direct-answer-reproduce-eval-command",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from docs/reproduce-eval.md"],
      workspace: workspaceFor("trace-direct-answer-reproduce-eval-command", "2026-04-01T01:19:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T01:21:00.000Z",
      seedActivatedAt: "2026-04-01T01:22:00.000Z",
      seedCues: [
        cue(
          "cue-command",
          "2026-04-01T01:18:30.000Z",
          "Run tsx scripts/validate-recorded-session-replay.ts --trace path/to/recorded-trace.json to regenerate a recorded-session replay proof bundle.",
        ),
      ],
      turns: [
        turn(
          "reproduce-command-turn-1",
          "2026-04-01T01:23:00.000Z",
          "2026-04-01T01:23:20.000Z",
          "What command reruns a sanitized trace proof?",
          ["tsx scripts/validate-recorded-session-replay.ts", "--trace"],
          {
            feedback: [
              feedback(
                "2026-04-01T01:23:35.000Z",
                "Use tsx scripts/validate-recorded-session-replay.ts --trace path/to/recorded-trace.json.",
              ),
            ],
            minimumPhraseHits: 1,
          },
        ),
        turn(
          "reproduce-command-turn-2",
          "2026-04-01T01:24:30.000Z",
          "2026-04-01T01:24:50.000Z",
          "What is the proof rerun command again?",
          ["tsx scripts/validate-recorded-session-replay.ts", "--trace"],
          {
            runtimeHints: ["command", "again"],
            minimumPhraseHits: 1,
          },
        ),
      ],
    }),
  },
  {
    slotId: "direct-answer-05",
    title: "Root release verification answer",
    category: "direct_answer",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["package.json"],
    tags: ["release", "command"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the root release verification scripts."],
    trace: buildTrace({
      traceId: "trace-direct-answer-release-verify",
      recordedAt: "2026-04-01T01:40:00.000Z",
      bundleBuiltAt: "2026-04-01T01:50:00.000Z",
      sessionId: "session-direct-answer-release-verify",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from package.json release scripts"],
      workspace: workspaceFor("trace-direct-answer-release-verify", "2026-04-01T01:39:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T01:41:00.000Z",
      seedActivatedAt: "2026-04-01T01:42:00.000Z",
      seedCues: [
        cue(
          "cue-release-verify",
          "2026-04-01T01:38:30.000Z",
          "The root release verification entrypoint is npm run release:verify. It starts with npm test before proof and package verification.",
        ),
      ],
      turns: [
        turn(
          "release-verify-turn-1",
          "2026-04-01T01:43:00.000Z",
          "2026-04-01T01:43:20.000Z",
          "What root command runs release verification?",
          ["npm run release:verify", "npm test"],
          {
            feedback: [feedback("2026-04-01T01:43:35.000Z", "Use npm run release:verify; it begins with npm test.")],
            minimumPhraseHits: 1,
          },
        ),
        turn(
          "release-verify-turn-2",
          "2026-04-01T01:44:30.000Z",
          "2026-04-01T01:44:50.000Z",
          "Which command is it again?",
          ["npm run release:verify", "npm test"],
          {
            runtimeHints: ["release", "command", "again"],
            minimumPhraseHits: 1,
          },
        ),
      ],
    }),
  },
  {
    slotId: "plan-execution-01",
    title: "Master proof plan and execution follow-through",
    category: "plan_execution",
    sourceKind: "repo_test_fixture_normalized",
    sourcePaths: ["packages/cli/dist/test/replay-score-resolution.test.js"],
    tags: ["plan", "execution", "master-plan"],
    notes: ["Normalized from a dynamic test fixture by freezing workspace.rootDir to a stable path."],
    trace: normalizedScoreResolution,
  },
  {
    slotId: "plan-execution-02",
    title: "Learned-route proof run kickoff",
    category: "plan_execution",
    sourceKind: "repo_test_fixture_normalized",
    sourcePaths: ["packages/cli/dist/test/learned-route-seed-carry-forward.test.js"],
    tags: ["plan", "proof-run", "kickoff"],
    notes: ["Normalized from a dynamic test fixture by freezing workspace.rootDir to a stable path."],
    trace: normalizedSeedCarryForward,
  },
  {
    slotId: "plan-execution-03",
    title: "Replay regression workflow plan",
    category: "plan_execution",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["docs/reproduce-eval.md", "docs/internal/recorded-session-replay.md"],
    tags: ["regression", "workflow", "hash-drift"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the checked-in replay regression workflow."],
    trace: buildTrace({
      traceId: "trace-plan-regression-workflow",
      recordedAt: "2026-04-01T02:00:00.000Z",
      bundleBuiltAt: "2026-04-01T02:10:00.000Z",
      sessionId: "session-plan-regression-workflow",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from docs/reproduce-eval.md"],
      workspace: workspaceFor("trace-plan-regression-workflow", "2026-04-01T01:59:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T02:01:00.000Z",
      seedActivatedAt: "2026-04-01T02:02:00.000Z",
      seedCues: [
        cue(
          "cue-regression",
          "2026-04-01T01:58:30.000Z",
          "If a replay rerun changes semantic hashes, compare hashes.json, summary-tables.json, coverage-snapshot.json, and hardening-snapshot.json, then inspect bundle.json and the per-mode outputs before claiming contract drift.",
        ),
      ],
      turns: [
        turn(
          "regression-workflow-turn-1",
          "2026-04-01T02:03:00.000Z",
          "2026-04-01T02:03:20.000Z",
          "Semantic hashes changed on rerun. What do we do next?",
          ["hashes.json", "bundle.json"],
          {
            runtimeHints: ["regression", "hashes"],
            feedback: [
              feedback(
                "2026-04-01T02:03:35.000Z",
                "Compare hashes.json, summary-tables.json, coverage-snapshot.json, and hardening-snapshot.json before you inspect bundle.json and modes/learned_route.json.",
              ),
            ],
            minimumPhraseHits: 1,
          },
        ),
        turn(
          "regression-workflow-turn-2",
          "2026-04-01T02:05:00.000Z",
          "2026-04-01T02:05:20.000Z",
          "Turn that into an end-to-end regression workflow.",
          ["hashes.json", "summary-tables.json", "modes/learned_route.json"],
          {
            runtimeHints: ["workflow", "end-to-end"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "plan-execution-04",
    title: "Proof artifact triage checklist",
    category: "plan_execution",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["docs/reproduce-eval.md"],
    tags: ["triage", "artifacts", "checklist"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the checked-in artifact inspection order."],
    trace: buildTrace({
      traceId: "trace-plan-proof-artifact-triage",
      recordedAt: "2026-04-01T02:20:00.000Z",
      bundleBuiltAt: "2026-04-01T02:30:00.000Z",
      sessionId: "session-plan-proof-artifact-triage",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from docs/reproduce-eval.md"],
      workspace: workspaceFor("trace-plan-proof-artifact-triage", "2026-04-01T02:19:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T02:21:00.000Z",
      seedActivatedAt: "2026-04-01T02:22:00.000Z",
      seedCues: [
        cue(
          "cue-artifact-order",
          "2026-04-01T02:18:30.000Z",
          "Start replay proof triage with summary.md, validation-report.json, hashes.json, coverage-snapshot.json, and hardening-snapshot.json before diving into bundle.json or the per-mode files.",
        ),
      ],
      turns: [
        turn(
          "artifact-triage-turn-1",
          "2026-04-01T02:23:00.000Z",
          "2026-04-01T02:23:20.000Z",
          "I reran a proof bundle. What should I inspect first?",
          ["summary.md", "validation-report.json", "hashes.json"],
          {
            runtimeHints: ["triage", "artifacts"],
            feedback: [
              feedback("2026-04-01T02:23:35.000Z", "Start with summary.md, validation-report.json, and hashes.json."),
            ],
            minimumPhraseHits: 1,
          },
        ),
        turn(
          "artifact-triage-turn-2",
          "2026-04-01T02:24:40.000Z",
          "2026-04-01T02:25:00.000Z",
          "Convert that into an execution checklist.",
          ["summary.md", "validation-report.json", "hashes.json"],
          {
            runtimeHints: ["checklist", "execution"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "plan-execution-05",
    title: "Lane handoff checklist",
    category: "plan_execution",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["task-artifacts/T-20260331-077/lane-b-serving.md", "task-status/T-20260331-077/lane-b-serving.json"],
    tags: ["handoff", "reporting", "execution"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the repo's checked-in handoff artifact pattern."],
    trace: buildTrace({
      traceId: "trace-plan-lane-handoff",
      recordedAt: "2026-04-01T02:40:00.000Z",
      bundleBuiltAt: "2026-04-01T02:50:00.000Z",
      sessionId: "session-plan-lane-handoff",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from checked-in task-artifacts and task-status examples"],
      workspace: workspaceFor("trace-plan-lane-handoff", "2026-04-01T02:39:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T02:41:00.000Z",
      seedActivatedAt: "2026-04-01T02:42:00.000Z",
      seedCues: [
        cue(
          "cue-handoff",
          "2026-04-01T02:38:30.000Z",
          "A lane handoff in this repo writes a concise task-artifacts markdown report, a machine-readable task-status JSON, records the checks that were run, and commits the branch state.",
        ),
      ],
      turns: [
        turn(
          "lane-handoff-turn-1",
          "2026-04-01T02:43:00.000Z",
          "2026-04-01T02:43:20.000Z",
          "What are the handoff steps for this lane?",
          ["task-artifacts", "task-status"],
          {
            feedback: [
              feedback(
                "2026-04-01T02:43:35.000Z",
                "Commit the branch, write the markdown report, write the machine-readable status, and record the checks.",
              ),
            ],
            minimumPhraseHits: 1,
          },
        ),
        turn(
          "lane-handoff-turn-2",
          "2026-04-01T02:44:30.000Z",
          "2026-04-01T02:44:50.000Z",
          "Give me the execution checklist again.",
          ["task-artifacts", "task-status", "commit"],
          {
            runtimeHints: ["handoff", "checklist"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "retrieval-memory-01",
    title: "Routing guide retrieval replay",
    category: "retrieval_memory_heavy",
    sourceKind: "repo_published_fixture",
    sourcePaths: [
      "docs/evidence/2026-03-26/0ca08242290103617b5bcaa2f80522d0124fc53d/recorded-session-replay/trace-comparative-replay/trace.json",
    ],
    tags: ["retrieval", "routing-guide", "follow-up"],
    notes: [
      "Published proof-bundle trace checked into docs/evidence. The source itself is a synthetic replay fixture, not a verified production session.",
    ],
    trace: exactComparativeReplay,
  },
  {
    slotId: "retrieval-memory-02",
    title: "Host-proof signal recall under follow-up pressure",
    category: "retrieval_memory_heavy",
    sourceKind: "repo_test_fixture_normalized",
    sourcePaths: ["packages/cli/dist/test/learned-route-seed-carry-forward.test.js"],
    tags: ["retrieval", "host-proof", "follow-up"],
    notes: ["Normalized from a dynamic test fixture by freezing workspace.rootDir to a stable path."],
    trace: normalizedSeedCarryForwardEvalDedup,
  },
  {
    slotId: "retrieval-memory-03",
    title: "Replay semantic hash recall",
    category: "retrieval_memory_heavy",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["docs/internal/recorded-session-replay.md"],
    tags: ["retrieval", "hashes", "memory"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the semantic hash contract."],
    trace: buildTrace({
      traceId: "trace-retrieval-proof-hashes",
      recordedAt: "2026-04-01T03:00:00.000Z",
      bundleBuiltAt: "2026-04-01T03:10:00.000Z",
      sessionId: "session-retrieval-proof-hashes",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from docs/internal/recorded-session-replay.md"],
      workspace: workspaceFor("trace-retrieval-proof-hashes", "2026-04-01T02:59:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T03:01:00.000Z",
      seedActivatedAt: "2026-04-01T03:02:00.000Z",
      seedCues: [
        cue(
          "cue-semantic-hashes",
          "2026-04-01T02:58:30.000Z",
          "Recorded session replay computes semantic hashes named traceHash, fixtureHash, scoreHash, and bundleHash.",
        ),
      ],
      turns: [
        turn(
          "proof-hashes-turn-1",
          "2026-04-01T03:03:00.000Z",
          "2026-04-01T03:03:20.000Z",
          "Which semantic hashes does replay compute?",
          ["traceHash", "bundleHash"],
          {
            runtimeHints: ["hashes", "semantic"],
            feedback: [
              feedback(
                "2026-04-01T03:03:35.000Z",
                "Name the semantic hashes directly: traceHash, fixtureHash, scoreHash, and bundleHash.",
              ),
            ],
            minimumPhraseHits: 1,
          },
        ),
        turn(
          "proof-hashes-turn-2",
          "2026-04-01T03:04:30.000Z",
          "2026-04-01T03:04:50.000Z",
          "Name those replay hashes again.",
          ["traceHash", "fixtureHash", "bundleHash"],
          {
            runtimeHints: ["hashes", "again"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "retrieval-memory-04",
    title: "Routing prior doc follow-up lookup",
    category: "retrieval_memory_heavy",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["docs/architecture/routing-prior.md", "packages/cli/dist/test/comparative-replay-modes.test.js"],
    tags: ["retrieval", "routing", "follow-up"],
    notes: [
      "Real trace source missing. Frozen as a replayable equivalent combining the routing-guide fixture with the checked-in routing-prior doc path.",
    ],
    trace: buildTrace({
      traceId: "trace-retrieval-routing-prior-doc",
      recordedAt: "2026-04-01T03:20:00.000Z",
      bundleBuiltAt: "2026-04-01T03:30:00.000Z",
      sessionId: "session-retrieval-routing-prior-doc",
      channel: "cli",
      sourceStream: "recorded/session",
      privacyNotes: [
        "synthetic replayable equivalent derived from the comparative replay routing fixture and docs/architecture/routing-prior.md",
      ],
      workspace: workspaceFor("trace-retrieval-routing-prior-doc", "2026-04-01T03:19:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T03:21:00.000Z",
      seedActivatedAt: "2026-04-01T03:22:00.000Z",
      seedCues: [
        cue(
          "cue-routing-docs",
          "2026-04-01T03:18:30.000Z",
          "The routing guide lives here, and the routing prior architecture note is checked in at docs/architecture/routing-prior.md.",
        ),
      ],
      turns: [
        turn(
          "routing-docs-turn-1",
          "2026-04-01T03:23:00.000Z",
          "2026-04-01T03:23:20.000Z",
          "show the routing guide",
          ["routing guide"],
          {
            runtimeHints: ["routing", "guide"],
            feedback: [
              feedback(
                "2026-04-01T03:23:35.000Z",
                "Keep the routing guide easy to find and mention the routing-prior doc when asked for the architecture note.",
              ),
            ],
          },
        ),
        turn(
          "routing-docs-turn-2",
          "2026-04-01T03:24:30.000Z",
          "2026-04-01T03:24:50.000Z",
          "show the routing guide and the routing prior doc again",
          ["routing guide", "routing-prior.md"],
          {
            runtimeHints: ["routing", "doc", "again"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "retrieval-memory-05",
    title: "Restart checklist archive lookup",
    category: "retrieval_memory_heavy",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["packages/cli/dist/test/recorded-session-replay-proof-bundle.test.js"],
    tags: ["retrieval", "restart-checklist", "follow-up"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the checked-in restart-checklist fixture language."],
    trace: buildTrace({
      traceId: "trace-retrieval-restart-checklist-lookup",
      recordedAt: "2026-04-01T03:40:00.000Z",
      bundleBuiltAt: "2026-04-01T03:50:00.000Z",
      sessionId: "session-retrieval-restart-checklist-lookup",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from packages/cli/dist/test/recorded-session-replay-proof-bundle.test.js"],
      workspace: workspaceFor("trace-retrieval-restart-checklist-lookup", "2026-04-01T03:39:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T03:41:00.000Z",
      seedActivatedAt: "2026-04-01T03:42:00.000Z",
      seedCues: [
        cue(
          "cue-restart-checklist",
          "2026-04-01T03:38:30.000Z",
          "The operator lane restart checklist is archived in docs/evidence and incidents are tagged with postmortem IDs.",
        ),
      ],
      turns: [
        turn(
          "restart-checklist-turn-1",
          "2026-04-01T03:43:00.000Z",
          "2026-04-01T03:43:20.000Z",
          "Where is the restart checklist archived?",
          ["docs/evidence"],
          {
            feedback: [feedback("2026-04-01T03:43:35.000Z", "Answer with docs/evidence and postmortem IDs.")],
          },
        ),
        turn(
          "restart-checklist-turn-2",
          "2026-04-01T03:44:30.000Z",
          "2026-04-01T03:44:50.000Z",
          "Where is it archived and how are incidents tagged again?",
          ["docs/evidence", "postmortem IDs"],
          {
            runtimeHints: ["restart", "archive", "again"],
            minimumPhraseHits: 1,
          },
        ),
      ],
    }),
  },
  {
    slotId: "correction-follow-up-01",
    title: "Restart order correction replay",
    category: "correction_follow_up_heavy",
    sourceKind: "repo_test_fixture_static",
    sourcePaths: ["packages/cli/dist/test/recorded-session-replay-proof-bundle.test.js"],
    tags: ["correction", "restart-order", "follow-up"],
    notes: ["Exactly transcribed from the checked-in proof-bundle writer test fixture."],
    trace: exactTernRecordedSessionProof,
  },
  {
    slotId: "correction-follow-up-02",
    title: "Explicit archive-path correction",
    category: "correction_follow_up_heavy",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["packages/cli/dist/test/recorded-session-replay-proof-bundle.test.js"],
    tags: ["correction", "explicit-paths"],
    notes: [
      "Real trace source missing. Frozen as a replayable equivalent emphasizing the correction-heavy path/label requirement.",
    ],
    trace: buildTrace({
      traceId: "trace-correction-answer-paths-explicit",
      recordedAt: "2026-04-01T04:00:00.000Z",
      bundleBuiltAt: "2026-04-01T04:10:00.000Z",
      sessionId: "session-correction-answer-paths-explicit",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from the recorded-session proof bundle fixture"],
      workspace: workspaceFor("trace-correction-answer-paths-explicit", "2026-04-01T03:59:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T04:01:00.000Z",
      seedActivatedAt: "2026-04-01T04:02:00.000Z",
      seedCues: [
        cue(
          "cue-explicit-archive",
          "2026-04-01T03:58:30.000Z",
          "When answering archive and incident-tag questions, keep the concrete path docs/evidence and the label postmortem IDs explicit.",
        ),
      ],
      turns: [
        turn(
          "explicit-paths-turn-1",
          "2026-04-01T04:03:00.000Z",
          "2026-04-01T04:03:20.000Z",
          "Where is the archive again?",
          ["docs/evidence"],
          {
            feedback: [
              feedback(
                "2026-04-01T04:03:35.000Z",
                "That is still too vague. Answer with docs/evidence and postmortem IDs.",
                "correction",
              ),
            ],
          },
        ),
        turn(
          "explicit-paths-turn-2",
          "2026-04-01T04:04:30.000Z",
          "2026-04-01T04:04:50.000Z",
          "Say it again without dropping the concrete path or incident tag.",
          ["docs/evidence", "postmortem IDs"],
          {
            runtimeHints: ["correction", "explicit"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "correction-follow-up-03",
    title: "Rollout verdict correction",
    category: "correction_follow_up_heavy",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["packages/cli/dist/test/replay-score-resolution.test.js"],
    tags: ["correction", "verdict", "plan"],
    notes: [
      "Real trace source missing. Frozen as a replayable equivalent around the rollout-verdict correction in the score-resolution fixture.",
    ],
    trace: buildTrace({
      traceId: "trace-correction-rollout-verdict",
      recordedAt: "2026-04-01T04:20:00.000Z",
      bundleBuiltAt: "2026-04-01T04:30:00.000Z",
      sessionId: "session-correction-rollout-verdict",
      channel: "telegram",
      sourceStream: "telegram/direct/proof-plan",
      privacyNotes: ["synthetic replayable equivalent derived from packages/cli/dist/test/replay-score-resolution.test.js"],
      workspace: workspaceFor("trace-correction-rollout-verdict", "2026-04-01T04:19:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T04:21:00.000Z",
      seedActivatedAt: "2026-04-01T04:22:00.000Z",
      seedCues: [
        cue("cue-rollout-verdict", "2026-04-01T04:18:30.000Z", "A master rollout plan ends with a verdict: ready, limited, or blocked."),
      ],
      turns: [
        turn(
          "rollout-verdict-turn-1",
          "2026-04-01T04:23:00.000Z",
          "2026-04-01T04:23:20.000Z",
          "Please make an end-to-end master plan for me.",
          ["master plan"],
          {
            feedback: [
              feedback(
                "2026-04-01T04:23:35.000Z",
                "The plan is incomplete unless it ends with a rollout verdict: ready, limited, or blocked.",
                "correction",
              ),
            ],
          },
        ),
        turn(
          "rollout-verdict-turn-2",
          "2026-04-01T04:24:40.000Z",
          "2026-04-01T04:25:00.000Z",
          "Rewrite the plan correctly.",
          ["ready, limited, or blocked"],
          {
            runtimeHints: ["rewrite", "verdict"],
          },
        ),
      ],
    }),
  },
  {
    slotId: "correction-follow-up-04",
    title: "Deeper proof story rewrite",
    category: "correction_follow_up_heavy",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["packages/cli/dist/test/learned-route-seed-carry-forward.test.js"],
    tags: ["correction", "rewrite", "host-proof"],
    notes: [
      "Real trace source missing. Frozen as a replayable equivalent around the deep-plan and host-proof-signal correction pattern.",
    ],
    trace: buildTrace({
      traceId: "trace-correction-deeper-proof-story",
      recordedAt: "2026-04-01T04:40:00.000Z",
      bundleBuiltAt: "2026-04-01T04:50:00.000Z",
      sessionId: "session-correction-deeper-proof-story",
      channel: "telegram",
      sourceStream: "telegram/direct/live-proof-story",
      privacyNotes: [
        "synthetic replayable equivalent derived from packages/cli/dist/test/learned-route-seed-carry-forward.test.js",
      ],
      workspace: workspaceFor("trace-correction-deeper-proof-story", "2026-04-01T04:39:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T04:41:00.000Z",
      seedActivatedAt: "2026-04-01T04:42:00.000Z",
      seedCues: [
        cue(
          "cue-proof-story",
          "2026-04-01T04:38:30.000Z",
          "OpenClawBrain starts as a correctly attached but mostly empty memory scaffold until host proof signals confirm the live state.",
        ),
      ],
      turns: [
        turn(
          "deeper-story-turn-1",
          "2026-04-01T04:43:00.000Z",
          "2026-04-01T04:43:20.000Z",
          "Use our current brain install to make it concrete.",
          ["memory scaffold"],
          {
            feedback: [feedback("2026-04-01T04:43:35.000Z", "Use the current shared ~/.openclaw install as the canonical example.")],
          },
        ),
        turn(
          "deeper-story-turn-2",
          "2026-04-01T04:44:30.000Z",
          "2026-04-01T04:44:50.000Z",
          "This is not good enough. Please make a deep detailed plan.",
          ["memory scaffold"],
          {
            feedback: [
              feedback(
                "2026-04-01T04:45:05.000Z",
                "Ground the rewrite in real host proof signals: BRAIN LOADED, loadProof=status_probe_ready, serveState=serving_active_pack, routeFn available=yes.",
                "correction",
              ),
            ],
          },
        ),
        turn(
          "deeper-story-turn-3",
          "2026-04-01T04:46:30.000Z",
          "2026-04-01T04:46:50.000Z",
          "Rewrite it correctly now.",
          ["BRAIN LOADED", "routeFn available=yes"],
          {
            runtimeHints: ["rewrite", "host-proof"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
  {
    slotId: "correction-follow-up-05",
    title: "Per-mode path correction",
    category: "correction_follow_up_heavy",
    sourceKind: "derived_replayable_equivalent",
    sourcePaths: ["docs/internal/recorded-session-replay.md"],
    tags: ["correction", "paths", "mode-files"],
    notes: ["Real trace source missing. Frozen as a replayable equivalent around the per-mode output path requirement."],
    trace: buildTrace({
      traceId: "trace-correction-mode-paths-explicit",
      recordedAt: "2026-04-01T05:00:00.000Z",
      bundleBuiltAt: "2026-04-01T05:10:00.000Z",
      sessionId: "session-correction-mode-paths-explicit",
      channel: "cli",
      sourceStream: "openclaw/runtime/cli",
      privacyNotes: ["synthetic replayable equivalent derived from docs/internal/recorded-session-replay.md"],
      workspace: workspaceFor("trace-correction-mode-paths-explicit", "2026-04-01T04:59:00.000Z", {
        branch: "main",
        labels: ["canonical-frozen-20"],
      }),
      evalTurnCount: 1,
      seedBuiltAt: "2026-04-01T05:01:00.000Z",
      seedActivatedAt: "2026-04-01T05:02:00.000Z",
      seedCues: [
        cue(
          "cue-mode-paths",
          "2026-04-01T04:58:30.000Z",
          "The proof bundle writes per-mode outputs at modes/no_brain.json, modes/vector_only.json, modes/graph_prior_only.json, and modes/learned_route.json.",
        ),
      ],
      turns: [
        turn(
          "mode-paths-turn-1",
          "2026-04-01T05:03:00.000Z",
          "2026-04-01T05:03:20.000Z",
          "Summarize the per-mode outputs.",
          ["modes/no_brain.json"],
          {
            feedback: [
              feedback(
                "2026-04-01T05:03:35.000Z",
                "Keep the file paths explicit: modes/no_brain.json and modes/learned_route.json.",
                "correction",
              ),
            ],
          },
        ),
        turn(
          "mode-paths-turn-2",
          "2026-04-01T05:04:30.000Z",
          "2026-04-01T05:04:50.000Z",
          "Say it again without dropping the concrete file names.",
          ["modes/no_brain.json", "modes/learned_route.json"],
          {
            runtimeHints: ["paths", "explicit"],
            minimumPhraseHits: 2,
          },
        ),
      ],
    }),
  },
];

function toManifestEntry(entry) {
  const traceDir = path.join(outputRoot, "traces", CATEGORY_DIRS[entry.category], entry.slotId);
  mkdirSync(traceDir, { recursive: true });
  const tracePath = path.join(traceDir, "trace.json");
  writeFileSync(tracePath, `${JSON.stringify(entry.trace, null, 2)}\n`, "utf8");

  return {
    slotId: entry.slotId,
    title: entry.title,
    category: entry.category,
    sourceKind: entry.sourceKind,
    sourcePaths: entry.sourcePaths,
    tags: entry.tags,
    notes: entry.notes,
    path: path.relative(outputRoot, tracePath).replace(/\\/g, "/"),
    status: "frozen_replayable_equivalent",
    realTraceSourceAvailable: false,
    sanitization: {
      classification:
        entry.sourceKind === "derived_replayable_equivalent"
          ? "synthetic_replayable_equivalent"
          : "synthetic_or_sanitized_repo_fixture",
      redactionRequired: false,
      notes:
        entry.sourceKind === "derived_replayable_equivalent"
          ? ["Authored from checked-in docs/tests because no stronger real-trace source is present in-repo."]
          : ["Checked-in source fixture already carries sanitized or synthetic content; no additional redaction was needed for freeze."],
    },
    shape: {
      turnCount: entry.trace.turns.length,
      evalTurnCount: entry.trace.evalTurnCount ?? 1,
      feedbackKinds: [
        ...new Set(entry.trace.turns.flatMap((item) => (item.feedback ?? []).map((feedbackItem) => feedbackItem.kind ?? "teaching"))),
      ].sort(),
      followUpTurnCount: Math.max(0, entry.trace.turns.length - 1),
      runtimeHintTurnCount: entry.trace.turns.filter((item) => Array.isArray(item.runtimeHints) && item.runtimeHints.length > 0).length,
    },
  };
}

function buildManifest(entries) {
  const manifestEntries = entries.map(toManifestEntry);
  const categoryCounts = CATEGORY_ORDER.reduce((acc, category) => {
    acc[category] = manifestEntries.filter((entry) => entry.category === category).length;
    return acc;
  }, {});
  const sourceSummary = manifestEntries.reduce((acc, entry) => {
    acc[entry.sourceKind] = (acc[entry.sourceKind] ?? 0) + 1;
    return acc;
  }, {});

  return {
    contract: "canonical_recorded_session_trace_set_manifest.v1",
    setId: "canonical-frozen-20",
    frozenAt: "2026-04-01T00:00:00.000Z",
    traceContract: "recorded_session_trace.v1",
    root: "evals/recorded-session-replay/canonical-frozen-20",
    traceCount: manifestEntries.length,
    categoryOrder: CATEGORY_ORDER,
    categoryCounts,
    sourceSummary,
    realTraceCoverage: {
      availableCount: 0,
      missingCount: manifestEntries.length,
      summary:
        "No checked-in recorded_session_trace.v1 input in this repo carries provenance strong enough to call it a verified first-party real production trace. This freeze therefore uses replayable equivalents only: 7 sourced directly or normalized from existing repo fixtures and 13 newly frozen equivalents derived from checked-in docs/tests.",
    },
    redactionPolicy: {
      additionalRedactionRequired: false,
      summary:
        "All 20 inputs are synthetic or sanitized replayable equivalents. No extra redaction was needed during freeze; provenance and synthetic status are recorded in this manifest.",
    },
    bundlePathTemplate: "docs/evidence/<YYYY-MM-DD>/<git-sha>/recorded-session-replay/<trace-id>/",
    selectionPrinciples: [
      "Prefer checked-in replayable recorded_session_trace.v1 sources before authoring any new equivalent.",
      "If no suitable source exists for a slot, freeze a replayable equivalent and mark the real-trace gap explicitly in provenance.",
      "Keep every slot replayable through the existing recorded-session proof-bundle writer.",
      "Preserve fixed category counts and stable on-disk paths so downstream lanes can build without re-deciding layout.",
    ],
    entries: manifestEntries,
  };
}

function buildSchema() {
  return {
    $schema: "https://json-schema.org/draft/2020-12/schema",
    title: "Canonical Recorded Session Trace Set Manifest",
    type: "object",
    additionalProperties: false,
    required: [
      "contract",
      "setId",
      "frozenAt",
      "traceContract",
      "root",
      "traceCount",
      "categoryOrder",
      "categoryCounts",
      "sourceSummary",
      "realTraceCoverage",
      "redactionPolicy",
      "bundlePathTemplate",
      "selectionPrinciples",
      "entries",
    ],
    properties: {
      contract: { const: "canonical_recorded_session_trace_set_manifest.v1" },
      setId: { type: "string", minLength: 1 },
      frozenAt: { type: "string", minLength: 1 },
      traceContract: { const: "recorded_session_trace.v1" },
      root: { type: "string", minLength: 1 },
      traceCount: { type: "integer", minimum: 1 },
      categoryOrder: {
        type: "array",
        minItems: 4,
        maxItems: 4,
        items: { enum: CATEGORY_ORDER },
      },
      categoryCounts: {
        type: "object",
        additionalProperties: false,
        required: CATEGORY_ORDER,
        properties: Object.fromEntries(CATEGORY_ORDER.map((category) => [category, { type: "integer", minimum: 0 }])),
      },
      sourceSummary: {
        type: "object",
        additionalProperties: false,
        required: [
          "repo_published_fixture",
          "repo_test_fixture_static",
          "repo_test_fixture_normalized",
          "derived_replayable_equivalent",
        ],
        properties: {
          repo_published_fixture: { type: "integer", minimum: 0 },
          repo_test_fixture_static: { type: "integer", minimum: 0 },
          repo_test_fixture_normalized: { type: "integer", minimum: 0 },
          derived_replayable_equivalent: { type: "integer", minimum: 0 },
        },
      },
      realTraceCoverage: {
        type: "object",
        additionalProperties: false,
        required: ["availableCount", "missingCount", "summary"],
        properties: {
          availableCount: { type: "integer", minimum: 0 },
          missingCount: { type: "integer", minimum: 0 },
          summary: { type: "string", minLength: 1 },
        },
      },
      redactionPolicy: {
        type: "object",
        additionalProperties: false,
        required: ["additionalRedactionRequired", "summary"],
        properties: {
          additionalRedactionRequired: { type: "boolean" },
          summary: { type: "string", minLength: 1 },
        },
      },
      bundlePathTemplate: { type: "string", minLength: 1 },
      selectionPrinciples: {
        type: "array",
        minItems: 1,
        items: { type: "string", minLength: 1 },
      },
      entries: {
        type: "array",
        minItems: 20,
        items: {
          type: "object",
          additionalProperties: false,
          required: [
            "slotId",
            "title",
            "category",
            "sourceKind",
            "sourcePaths",
            "tags",
            "notes",
            "path",
            "status",
            "realTraceSourceAvailable",
            "sanitization",
            "shape",
          ],
          properties: {
            slotId: { type: "string", minLength: 1 },
            title: { type: "string", minLength: 1 },
            category: { enum: CATEGORY_ORDER },
            sourceKind: {
              enum: [
                "repo_published_fixture",
                "repo_test_fixture_static",
                "repo_test_fixture_normalized",
                "derived_replayable_equivalent",
              ],
            },
            sourcePaths: {
              type: "array",
              minItems: 1,
              items: { type: "string", minLength: 1 },
            },
            tags: {
              type: "array",
              minItems: 1,
              items: { type: "string", minLength: 1 },
            },
            notes: {
              type: "array",
              minItems: 1,
              items: { type: "string", minLength: 1 },
            },
            path: { type: "string", minLength: 1 },
            status: { const: "frozen_replayable_equivalent" },
            realTraceSourceAvailable: { const: false },
            sanitization: {
              type: "object",
              additionalProperties: false,
              required: ["classification", "redactionRequired", "notes"],
              properties: {
                classification: {
                  enum: ["synthetic_replayable_equivalent", "synthetic_or_sanitized_repo_fixture"],
                },
                redactionRequired: { type: "boolean" },
                notes: {
                  type: "array",
                  minItems: 1,
                  items: { type: "string", minLength: 1 },
                },
              },
            },
            shape: {
              type: "object",
              additionalProperties: false,
              required: ["turnCount", "evalTurnCount", "feedbackKinds", "followUpTurnCount", "runtimeHintTurnCount"],
              properties: {
                turnCount: { type: "integer", minimum: 1 },
                evalTurnCount: { type: "integer", minimum: 1 },
                feedbackKinds: {
                  type: "array",
                  items: { type: "string", minLength: 1 },
                },
                followUpTurnCount: { type: "integer", minimum: 0 },
                runtimeHintTurnCount: { type: "integer", minimum: 0 },
              },
            },
          },
        },
      },
    },
  };
}

function buildReadme() {
  return [
    "# Canonical Frozen 20 Recorded-Session Inputs",
    "",
    "This directory freezes the canonical 20-slot recorded-session replay input surface for downstream eval work.",
    "",
    "## What is here",
    "",
    "- `manifest.json`: canonical slot manifest with category, provenance, sanitization notes, and stable paths",
    "- `manifest.schema.json`: machine-readable schema for the manifest",
    "- `traces/<category>/<slot-id>/trace.json`: the actual replayable `recorded_session_trace.v1` input for each slot",
    "",
    "## Truthfulness boundary",
    "",
    "As of 2026-04-01, this repo does not contain any checked-in `recorded_session_trace.v1` input with provenance strong enough to call it a verified first-party real production trace.",
    "",
    "This freeze therefore uses replayable equivalents only:",
    "",
    "- 2 published proof-bundle fixtures already checked into `docs/evidence`",
    "- 5 checked-in test fixtures lifted directly or normalized from dynamic temp-workspace tests",
    "- 13 newly frozen replayable equivalents derived from checked-in docs/tests where no stronger source existed",
    "",
    "Every slot records that gap explicitly in `manifest.json` via `realTraceSourceAvailable: false` and its `sourceKind` / `notes` fields.",
    "",
    "## Category contract",
    "",
    "The set keeps exactly:",
    "",
    "- 5 direct-answer traces",
    "- 5 plan/execution traces",
    "- 5 retrieval/memory-heavy traces",
    "- 5 correction/follow-up-heavy traces",
    "",
    "## Path contract",
    "",
    "The canonical path shape is fixed:",
    "",
    "`traces/<category-dir>/<slot-id>/trace.json`",
    "",
    "Where `<category-dir>` is one of:",
    "",
    "- `direct-answer`",
    "- `plan-execution`",
    "- `retrieval-memory-heavy`",
    "- `correction-follow-up-heavy`",
    "",
    "Downstream lanes should treat the slot ids and these paths as stable.",
    "",
    "## Verification",
    "",
    "The focused regression is:",
    "",
    "`npx vitest run test/canonical-frozen-trace-set.test.ts`",
    "",
    "That test checks the manifest shape, category counts, provenance summary, and that every frozen trace can round-trip through the recorded-session proof-bundle writer.",
    "",
    "For a single trace, you can also write a proof bundle with:",
    "",
    "`tsx scripts/validate-recorded-session-replay.ts --trace <path-to-trace.json>`",
    "",
  ].join("\n");
}

function main() {
  rmSync(outputRoot, { recursive: true, force: true });
  mkdirSync(outputRoot, { recursive: true });

  const manifest = buildManifest(definitions);
  const schema = buildSchema();
  const readme = buildReadme();

  writeFileSync(path.join(outputRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputRoot, "manifest.schema.json"), `${JSON.stringify(schema, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputRoot, "README.md"), `${readme}\n`, "utf8");
}

main();
