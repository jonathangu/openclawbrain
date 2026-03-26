import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT, validateRecordedSessionReplayProofBundle, writeRecordedSessionReplayProofBundle } from "../src/index.js";

function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-recorded-session-proof-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}

function createRecordedTrace() {
    return {
        contract: "recorded_session_trace.v1",
        traceId: "tern-recorded-session-proof",
        source: "sanitized_recorded_session",
        recordedAt: "2026-03-25T00:00:00.000Z",
        bundleBuiltAt: "2026-03-25T00:30:00.000Z",
        sessionId: "session-tern-proof",
        channel: "cli",
        sourceStream: "openclaw/runtime/cli",
        privacy: {
            sanitized: true,
            notes: ["synthetic fixture for deterministic proof bundle tests"]
        },
        workspace: {
            workspaceId: "workspace-tern",
            snapshotId: "snapshot-tern",
            capturedAt: "2026-03-24T23:59:00.000Z",
            rootDir: "/workspace/tern",
            branch: "task/T-20260325-031-wave3-proof-bundle-writer",
            revision: "109e9b3",
            labels: ["proof", "recorded-session"]
        },
        seedBuiltAt: "2026-03-25T00:05:00.000Z",
        seedActivatedAt: "2026-03-25T00:06:00.000Z",
        seedCues: [
            {
                cueId: "cue-deploy-journal",
                createdAt: "2026-03-25T00:01:00.000Z",
                content: "The operator lane restart checklist is archived in docs/evidence and incidents are tagged with postmortem IDs.",
                kind: "teaching"
            },
            {
                cueId: "cue-restart-order",
                createdAt: "2026-03-25T00:02:00.000Z",
                content: "Keep the operator lane restart order explicit when proving a recorded session replay.",
                kind: "teaching"
            }
        ],
        turns: [
            {
                turnId: "turn-alpha",
                createdAt: "2026-03-25T00:10:00.000Z",
                deliveredAt: "2026-03-25T00:10:30.000Z",
                userMessage: "Where is the restart checklist archived and how are incidents tagged?",
                feedback: [
                    {
                        createdAt: "2026-03-25T00:11:00.000Z",
                        content: "Answer with docs/evidence and postmortem IDs.",
                        kind: "correction"
                    }
                ],
                expectedContextPhrases: ["docs/evidence", "postmortem IDs"],
                minimumPhraseHits: 1
            },
            {
                turnId: "turn-beta",
                createdAt: "2026-03-25T00:20:00.000Z",
                deliveredAt: "2026-03-25T00:20:20.000Z",
                userMessage: "Summarize the operator lane restart order.",
                runtimeHints: ["proof", "replay"],
                feedback: [
                    {
                        createdAt: "2026-03-25T00:21:00.000Z",
                        content: "Keep the operator lane restart order explicit.",
                        kind: "teaching"
                    }
                ],
                expectedContextPhrases: ["operator lane", "restart order"],
                minimumPhraseHits: 1
            }
        ]
    };
}

function curatedRelativePaths() {
    return [
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.manifest,
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.trace,
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.fixture,
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.bundle,
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.environment,
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.summary,
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.summaryTables,
        RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.hashes,
        `${RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.modeDir}/no_brain.json`,
        `${RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.modeDir}/vector_only.json`,
        `${RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.modeDir}/graph_prior_only.json`,
        `${RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.modeDir}/learned_route.json`
    ];
}

test("recorded session replay proof bundle writes the curated deterministic artifact set", (t) => {
    const tempRoot = createTempRoot(t);
    const firstRoot = path.join(tempRoot, "first");
    const secondRoot = path.join(tempRoot, "second");
    const trace = createRecordedTrace();
    const first = writeRecordedSessionReplayProofBundle({
        rootDir: firstRoot,
        trace,
        scratchRootDir: tempRoot
    });
    const second = writeRecordedSessionReplayProofBundle({
        rootDir: secondRoot,
        trace,
        scratchRootDir: tempRoot
    });
    assert.deepEqual(first.manifest.modeOrder, ["no_brain", "vector_only", "graph_prior_only", "learned_route"]);
    assert.equal(first.modeOutputs.length, 4);
    assert.equal(first.hashes.semantic.traceHash, first.fixture.traceHash);
    assert.equal(first.hashes.semantic.fixtureHash, first.fixture.fixtureHash);
    assert.equal(first.hashes.semantic.scoreHash, first.bundle.scoreHash);
    assert.equal(first.hashes.semantic.bundleHash, first.bundle.bundleHash);
    assert.equal(first.hashes.files.length, 11);
    for (const relativePath of curatedRelativePaths()) {
        assert.equal(existsSync(path.join(firstRoot, relativePath)), true, `${relativePath} should exist`);
        assert.equal(readFileSync(path.join(firstRoot, relativePath), "utf8"), readFileSync(path.join(secondRoot, relativePath), "utf8"), `${relativePath} should be reproducible across output roots`);
    }
    const validation = validateRecordedSessionReplayProofBundle(firstRoot);
    assert.equal(validation.ok, true);
    assert.equal(validation.fileHashesMatch, true);
    assert.equal(validation.bundleHashMatches, true);
    assert.equal(validation.scoreHashMatches, true);
    assert.equal(validation.verifiedFileCount, validation.expectedFileCount);
});

test("recorded session replay proof validator catches per-mode drift and file-hash drift", (t) => {
    const tempRoot = createTempRoot(t);
    const rootDir = path.join(tempRoot, "bundle");
    writeRecordedSessionReplayProofBundle({
        rootDir,
        trace: createRecordedTrace(),
        scratchRootDir: tempRoot
    });
    const learnedReplayPath = path.join(rootDir, RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT.modeDir, "learned_route.json");
    const learnedReplay = JSON.parse(readFileSync(learnedReplayPath, "utf8"));
    learnedReplay.summary.qualityScore = -1;
    writeFileSync(learnedReplayPath, JSON.stringify(learnedReplay), "utf8");
    const validation = validateRecordedSessionReplayProofBundle(rootDir);
    assert.equal(validation.ok, false);
    assert.equal(validation.fileHashesMatch, false);
    assert.match(validation.errors.join("\n"), /mode output drift detected for learned_route/);
    assert.match(validation.errors.join("\n"), /hashes\.json file digests do not match the written artifacts/);
});
