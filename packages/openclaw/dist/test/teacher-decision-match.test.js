import test from "node:test";
import assert from "node:assert/strict";
import { createServeTimeDecisionMatcher } from "../src/teacher-decision-match.js";

function makeDecision(overrides = {}) {
    return {
        turnCompileEventId: null,
        sessionId: "sess-1",
        channel: "cli",
        turnCreatedAt: "2026-03-20T18:00:00.000Z",
        recordedAt: "2026-03-20T18:00:01.000Z",
        userMessage: "How do I deploy this?",
        ...overrides,
    };
}

function makeInteraction(overrides = {}) {
    return {
        eventId: "evt-interaction",
        sessionId: "sess-1",
        channel: "cli",
        createdAt: "2026-03-20T18:00:00.000Z",
        ...overrides,
    };
}

test("serve-time decision matcher prefers exact compile event ids", () => {
    const exact = makeDecision({
        turnCompileEventId: "evt-exact",
        turnCreatedAt: "2026-03-20T18:05:00.000Z",
    });
    const nearby = makeDecision({
        recordedAt: "2026-03-20T18:00:00.500Z",
    });
    const matcher = createServeTimeDecisionMatcher([nearby, exact]);
    assert.equal(matcher(makeInteraction({
        eventId: "evt-exact",
        createdAt: "2026-03-20T18:00:00.250Z",
    })), exact);
});

test("serve-time decision matcher falls back to exact timestamp keys when ids drift", () => {
    const decision = makeDecision();
    const matcher = createServeTimeDecisionMatcher([decision]);
    assert.equal(matcher(makeInteraction({
        eventId: "evt-other",
        createdAt: "2026-03-20T18:00:00.000Z",
    })), decision);
});

test("serve-time decision matcher accepts the nearest same-session decision inside the grace window", () => {
    const decision = makeDecision({
        turnCompileEventId: null,
        turnCreatedAt: "2026-03-20T18:00:04.000Z",
        recordedAt: "2026-03-20T18:00:05.000Z",
    });
    const matcher = createServeTimeDecisionMatcher([decision], { maxTimeDeltaMs: 10_000 });
    assert.equal(matcher(makeInteraction({
        createdAt: "2026-03-20T18:00:06.200Z",
    })), decision);
});

test("serve-time decision matcher refuses ambiguous nearest-time matches", () => {
    const earlier = makeDecision({
        turnCompileEventId: null,
        turnCreatedAt: "2026-03-20T18:00:04.000Z",
        recordedAt: "2026-03-20T18:00:04.000Z",
    });
    const later = makeDecision({
        turnCompileEventId: null,
        turnCreatedAt: "2026-03-20T18:00:08.000Z",
        recordedAt: "2026-03-20T18:00:08.000Z",
    });
    const matcher = createServeTimeDecisionMatcher([earlier, later], { maxTimeDeltaMs: 10_000 });
    assert.equal(matcher(makeInteraction({
        eventId: "evt-ambiguous",
        createdAt: "2026-03-20T18:00:06.000Z",
    })), null);
});

