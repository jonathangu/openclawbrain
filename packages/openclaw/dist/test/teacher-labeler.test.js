import test from "node:test";
import assert from "node:assert/strict";
import { buildNormalizedEventExport, createInteractionEvent } from "@openclawbrain/contracts";
import { createOllamaTeacherLabeler } from "../src/teacher-labeler.js";

function makeMessageDeliveredInteraction() {
    return createInteractionEvent({
        eventId: "evt-message-delivered-1",
        agentId: "main",
        sessionId: "session-1",
        channel: "cli",
        sequence: 1,
        kind: "message_delivered",
        createdAt: "2026-04-03T12:00:00.000Z",
        source: {
            runtimeOwner: "openclaw",
            stream: "openclaw/session-store/current_profile/main/cli"
        },
        messageId: "msg-1"
    });
}

function makeServeTimeDecision() {
    return {
        recordId: "decision-1",
        selectionDigest: null,
        activePackGraphChecksum: null,
        turnCompileEventId: null,
        sessionId: "session-1",
        channel: "cli",
        turnCreatedAt: "2026-04-03T12:00:00.000Z",
        recordedAt: "2026-04-03T12:00:01.000Z",
        userMessage: "How do I deploy this service?",
        usedLearnedRouteFn: true,
        fallbackReason: null,
        actualBudget: {
            selectedCount: 2
        },
        kernelContextCount: 1,
        brainContextCount: 1,
        chosenContextIds: ["ctx-kernel-1", "ctx-brain-1"],
        selectedBrainContextIds: ["ctx-brain-1"],
        selectedKernelContextIds: ["ctx-kernel-1"]
    };
}

test("openclaw teacher labeler can materialize an artifact from a session-tail message_delivered interaction", async () => {
    const interaction = makeMessageDeliveredInteraction();
    const normalizedEventExport = buildNormalizedEventExport({
        interactionEvents: [interaction],
        feedbackEvents: []
    });
    const labeler = createOllamaTeacherLabeler({
        provider: "ollama",
        maxArtifactsPerExport: 1,
        maxInteractionsPerExport: 1,
        client: {
            async generate() {
                return {
                    response: JSON.stringify({
                        labels: [
                            {
                                interactionEventId: interaction.eventId,
                                kind: "teaching",
                                content: "Deploy by running the release step after tests pass."
                            }
                        ]
                    })
                };
            }
        }
    });

    const result = await labeler.label({
        normalizedEventExport,
        observedAt: "2026-04-03T12:01:00.000Z",
        staleAfterMs: 60_000,
        serveTimeDecisions: [makeServeTimeDecision()],
        existingArtifacts: []
    });

    assert.equal(result.status, "ok");
    assert.equal(result.detail, "candidates=1;labels=1");
    assert.equal(result.artifacts.length, 1);
    assert.equal(result.artifacts[0].relatedInteractionId, interaction.eventId);
    assert.equal(result.artifacts[0].kind, "teaching");
    assert.equal(result.artifacts[0].content, "Deploy by running the release step after tests pass.");
});
