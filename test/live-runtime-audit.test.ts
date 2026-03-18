import { describe, expect, it } from "vitest";

import { summarizeTeacherLoopTruth } from "../src/live-runtime-audit.js";

describe("summarizeTeacherLoopTruth", () => {
  it("treats a fresh watch heartbeat with no teacher artifacts as healthy and not stale", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "none",
        lastNoOpReason: "no_teacher_artifacts",
        latestFreshness: "stale",
        queueDepth: 0,
        running: false,
        watchState: "watching",
      }),
    ).toEqual({
      healthy: true,
      idle: true,
      stale: false,
    });
  });

  it("keeps stale snapshots stale even without an explicit failure", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "none",
        lastNoOpReason: "none",
        latestFreshness: "fresh",
        queueDepth: 0,
        running: false,
        watchState: "stale_snapshot",
      }),
    ).toEqual({
      healthy: false,
      idle: true,
      stale: true,
    });
  });

  it("keeps explicit teacher failures unhealthy", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "provider_error",
        lastNoOpReason: "none",
        latestFreshness: "fresh",
        queueDepth: 1,
        running: false,
        watchState: "watching",
      }),
    ).toEqual({
      healthy: false,
      idle: false,
      stale: false,
    });
  });
});
