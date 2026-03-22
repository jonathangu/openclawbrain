import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildNormalizedEventExport } from "@openclawbrain/event-export";
import { buildCandidatePackFromNormalizedEventExport } from "../src/local-learner.js";
import { CONTRACT_IDS, ROUTER_PG_PROFILE_V2 } from "@openclawbrain/contracts";

function createWorkspace(t) {
    const rootDir = mkdtempSync(path.join(os.tmpdir(), "ocb-native-pg-v2-"));
    writeFileSync(path.join(rootDir, "README.md"), "# Policy Workspace\nRelevant routing context lives here.\n", "utf8");
    t.after(() => {
        rmSync(rootDir, { recursive: true, force: true });
    });
    return {
        workspaceId: "ws-native-pg-v2",
        snapshotId: "snapshot-native-pg-v2",
        capturedAt: "2026-03-22T10:00:00.000Z",
        rootDir,
        branch: "main",
        revision: "abc123",
        dirty: false,
        labels: ["test"],
        files: ["README.md"]
    };
}

function buildCrossSurfaceFixture(t) {
    const workspace = createWorkspace(t);
    const interactionEvents = [
        {
            contract: CONTRACT_IDS.interactionEvents,
            eventId: "evt-i1",
            agentId: "agent",
            sessionId: "sess-1",
            channel: "cli",
            sequence: 1,
            kind: "memory_compiled",
            createdAt: "2026-03-22T10:00:00.000Z",
            source: { runtimeOwner: "openclaw", stream: "session.tail" },
            messageId: "msg-1"
        },
        {
            contract: CONTRACT_IDS.interactionEvents,
            eventId: "evt-i2",
            agentId: "agent",
            sessionId: "sess-1",
            channel: "cli",
            sequence: 2,
            kind: "memory_compiled",
            createdAt: "2026-03-22T10:01:00.000Z",
            source: { runtimeOwner: "openclaw", stream: "session.tail" },
            messageId: "msg-2"
        }
    ];
    const feedbackEvents = [
        {
            contract: CONTRACT_IDS.feedbackEvents,
            eventId: "evt-f1",
            agentId: "agent",
            sessionId: "sess-1",
            channel: "cli",
            sequence: 3,
            kind: "approval",
            createdAt: "2026-03-22T10:02:00.000Z",
            source: { runtimeOwner: "openclaw", stream: "session.tail" },
            content: "yes that context helped",
            relatedInteractionId: "evt-i2"
        }
    ];
    const normalizedEventExport = buildNormalizedEventExport({
        interactionEvents,
        feedbackEvents
    });
    const structuralOps = { connect: 3, split: 1, merge: 1, prune: 1 };
    const servedPack = buildCandidatePackFromNormalizedEventExport({
        packLabel: "native-pg-v2",
        workspace,
        normalizedEventExport,
        learnedRouting: false,
        builtAt: "2026-03-22T10:05:00.000Z",
        structuralOps
    });
    const packId = servedPack.summary.packId;
    const chosenContextIds = [`${packId}:event:evt-i2`, `${packId}:merge:1`];
    const serveTimeDecision = {
        recordType: "serve_time_route_decision",
        recordId: "decision-1",
        recordedAt: "2026-03-22T10:01:01.000Z",
        activationRoot: "/tmp/activation-root",
        breadcrumbs: {
            entrypoint: "compileRuntimeContext",
            invocationSurface: "direct_compile_call",
            hostEvent: null,
            installedEntryPath: null,
            syntheticTurn: false
        },
        sessionId: "sess-1",
        channel: "cli",
        userMessage: "show me the relevant policy context",
        turnSequenceStart: 2,
        turnCompileEventId: "evt-i2",
        turnCreatedAt: "2026-03-22T10:01:00.000Z",
        activePackId: packId,
        activePackBuiltAt: "2026-03-22T10:05:00.000Z",
        activePackEventExportDigest: servedPack.summary.eventExportDigest,
        activePackRouterChecksum: null,
        activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
        routerIdentity: null,
        usedLearnedRouteFn: true,
        servedArtifact: null,
        selectionDigest: "selection-1",
        requestedBudget: {
            modeRequested: "learned_required",
            maxContextBlocks: 2,
            maxContextChars: null
        },
        actualBudget: {
            modeEffective: "learned_required",
            selectedCount: 2,
            selectedCharCount: 120,
            selectedTokenCount: 24
        },
        candidateSetIds: [`${packId}:event:evt-i1`, `${packId}:event:evt-i2`, `${packId}:merge:1`],
        chosenContextIds,
        candidateScores: [
            {
                blockId: `${packId}:event:evt-i1`,
                source: "event:i1",
                selected: false,
                compactedFrom: [],
                matchedTokens: ["policy"],
                routingChannels: ["graph"],
                channelScores: { graph: 0.2 },
                routeFnScore: 0.2,
                actionScore: 0.2,
                actionProbability: 0.2,
                actionLogProbability: Math.log(0.2),
                traversalScore: 0.2,
                priority: 1
            },
            {
                blockId: `${packId}:event:evt-i2`,
                source: "event:i2",
                selected: true,
                compactedFrom: [],
                matchedTokens: ["policy", "context"],
                routingChannels: ["graph"],
                channelScores: { graph: 0.9 },
                routeFnScore: 0.9,
                actionScore: 0.9,
                actionProbability: 0.5,
                actionLogProbability: Math.log(0.5),
                traversalScore: 0.9,
                priority: 2
            },
            {
                blockId: `${packId}:merge:1`,
                source: "merge",
                selected: true,
                compactedFrom: [`${packId}:event:evt-i1`, `${packId}:event:evt-i2`],
                matchedTokens: ["policy", "related"],
                routingChannels: ["graph"],
                channelScores: { graph: 0.8 },
                routeFnScore: 0.8,
                actionScore: 0.8,
                actionProbability: 0.3,
                actionLogProbability: Math.log(0.3),
                traversalScore: 0.8,
                priority: 2
            }
        ],
        structuralSignals: null,
        fallbackReason: null,
        hotPathTiming: { totalMs: 1 },
        kernelContextCount: 2,
        brainContextCount: 0,
        selectedKernelContextIds: chosenContextIds,
        selectedBrainContextIds: [],
        promotionLink: null
    };
    return { workspace, normalizedEventExport, structuralOps, serveTimeDecision };
}

test("native V2 route updates survive serve-pack to candidate-pack block-id remap", (t) => {
    const { workspace, normalizedEventExport, structuralOps, serveTimeDecision } = buildCrossSurfaceFixture(t);
    const result = buildCandidatePackFromNormalizedEventExport({
        packLabel: "native-pg-v2",
        workspace,
        normalizedEventExport,
        learnedRouting: true,
        builtAt: "2026-03-22T10:06:00.000Z",
        structuralOps,
        pgVersion: "v2",
        serveTimeDecisions: [serveTimeDecision],
        baselineState: {
            movingAverage: 0,
            count: 0,
            alpha: 0.1,
            lastUpdatedAt: "2026-03-22T10:00:00.000Z"
        }
    });
    const router = result.payloads.router;
    assert.ok(router, "expected learned router artifact");
    assert.equal(result.routingBuild.learnedRoutingPath, "policy_gradient_v2");
    assert.equal(result.routingBuild.pgVersionUsed, "v2");
    assert.deepEqual(router.training.objective.profile, ROUTER_PG_PROFILE_V2);
    assert.equal(router.training.noOpReason, null);
    assert.equal(router.training.routeTraceCount, 0);
    assert.equal(router.training.supervisionCount, 0);
    assert.equal(router.traces.length, 0, "native V2 should not fall back to trace-based V1 updates");
    assert.ok(router.policyUpdates.length > 0, "expected native V2 to emit policy updates");
    assert.ok(router.policyUpdates.some((update) => update.delta !== 0), "expected at least one nonzero native V2 delta");
    assert.ok(router.policyUpdates.every((update) => update.blockId.startsWith(`${result.summary.packId}:`)), "expected updates against candidate-pack block ids");
});


test("native V2 remap tolerates missing optional block-id arrays during full replay", (t) => {
    const { workspace, normalizedEventExport, structuralOps, serveTimeDecision } = buildCrossSurfaceFixture(t);
    const replayDecision = {
        ...serveTimeDecision,
        candidateSetIds: undefined,
        selectedKernelContextIds: undefined,
        selectedBrainContextIds: undefined,
        candidateScores: serveTimeDecision.candidateScores.map((score, index) => index === 0
            ? { ...score, compactedFrom: undefined }
            : score)
    };
    const result = buildCandidatePackFromNormalizedEventExport({
        packLabel: "native-pg-v2-replay",
        workspace,
        normalizedEventExport,
        learnedRouting: true,
        builtAt: "2026-03-22T10:07:00.000Z",
        structuralOps,
        pgVersion: "v2",
        serveTimeDecisions: [replayDecision],
        baselineState: {
            movingAverage: 0,
            count: 0,
            alpha: 0.1,
            lastUpdatedAt: "2026-03-22T10:00:00.000Z"
        }
    });
    const router = result.payloads.router;
    assert.ok(router, "expected learned router artifact");
    assert.equal(router.training.noOpReason, null);
    assert.ok(router.policyUpdates.some((update) => update.delta !== 0), "expected nonzero native V2 delta even when optional arrays are absent");
});
