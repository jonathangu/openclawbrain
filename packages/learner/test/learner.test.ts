import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { FIXTURE_FEEDBACK_EVENTS, FIXTURE_INTERACTION_EVENTS, FIXTURE_NORMALIZED_EVENT_EXPORT } from "@openclawbrain/contracts";
import {
  buildCandidatePack,
  buildCandidatePackFromNormalizedEventExport,
  materializeCandidatePack,
  materializeCandidatePackFromNormalizedEventExport
} from "@openclawbrain/learner";

test("learner emits deterministic immutable pack manifests", (t) => {
  const input = {
    packLabel: "demo",
    workspace: {
      workspaceId: "workspace-1",
      snapshotId: "workspace-1@snapshot-1",
      capturedAt: "2026-03-06T00:00:00.000Z",
      rootDir: "/workspace/demo",
      branch: "main",
      revision: "demo-rev-1",
      labels: ["demo", "typescript"]
    },
    eventRange: {
      start: 101,
      end: 104
    },
    eventExports: {
      interactionEvents: FIXTURE_INTERACTION_EVENTS,
      feedbackEvents: FIXTURE_FEEDBACK_EVENTS
    },
    learnedRouting: true,
    structuralOps: {
      connect: 2,
      prune: 1
    }
  } as const;

  const first = buildCandidatePack(input);
  const second = buildCandidatePack(input);
  const rootDir = path.join(tmpdir(), `openclawbrain-ts-learn-${Date.now()}`);
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const descriptor = materializeCandidatePack(rootDir, input);

  assert.equal(first.summary.packId, second.summary.packId);
  assert.equal(first.manifest.immutable, true);
  assert.equal(first.manifest.routePolicy, "requires_learned_routing");
  assert.equal(descriptor.manifest.packId, first.summary.packId);
  assert.equal(descriptor.router?.routerIdentity, `${first.summary.packId}:route_fn`);
  assert.equal(descriptor.graph.blocks.length, 9);
  assert.equal(first.summary.eventRange.start, 101);
  assert.equal(first.summary.eventRange.end, 104);
  assert.equal(first.summary.eventRange.count, 4);
  assert.equal(first.summary.workspaceSnapshot, "workspace-1@snapshot-1");
  assert.equal(first.manifest.provenance.workspace.snapshotId, "workspace-1@snapshot-1");
  assert.equal(first.summary.eventExportDigest, first.manifest.provenance.eventExports?.exportDigest ?? null);
  assert.equal(first.manifest.provenance.eventExports?.interactionCount, FIXTURE_INTERACTION_EVENTS.length);
  assert.equal(first.manifest.provenance.eventExports?.feedbackCount, FIXTURE_FEEDBACK_EVENTS.length);
  assert.match(descriptor.graph.blocks.map((block) => block.text).join("\n"), /Workspace snapshot workspace-1@snapshot-1/);
  assert.match(descriptor.graph.blocks.map((block) => block.text).join("\n"), /Use the unified feedback scanner before enabling default loop scans/);
  assert.equal(descriptor.vectors.entries.some((entry) => entry.blockId.endsWith("evt-feedback-fixture-1")), true);
});

test("learner rejects mismatched explicit event ranges when event exports are supplied", () => {
  assert.throws(
    () =>
      buildCandidatePack({
        packLabel: "bad-range",
        workspace: {
          workspaceId: "workspace-1",
          snapshotId: "workspace-1@snapshot-bad-range",
          capturedAt: "2026-03-06T00:00:00.000Z",
          rootDir: "/workspace/demo"
        },
        eventRange: {
          start: 100,
          end: 104
        },
        eventExports: {
          interactionEvents: FIXTURE_INTERACTION_EVENTS,
          feedbackEvents: FIXTURE_FEEDBACK_EVENTS
        },
        learnedRouting: false
      }),
    /does not match requested range/
  );
});

test("learner can materialize a candidate pack directly from a normalized event export", (t) => {
  const rootDir = path.join(tmpdir(), `openclawbrain-ts-export-${Date.now()}`);
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "from-export",
    workspace: {
      workspaceId: "workspace-export",
      snapshotId: "workspace-export@snapshot-1",
      capturedAt: "2026-03-06T01:00:00.000Z",
      rootDir: "/workspace/export",
      revision: "workspace-export-rev"
    },
    normalizedEventExport: FIXTURE_NORMALIZED_EVENT_EXPORT,
    learnedRouting: true,
    structuralOps: {
      split: 1,
      connect: 1
    }
  });
  const descriptor = materializeCandidatePackFromNormalizedEventExport(rootDir, {
    packLabel: "from-export",
    workspace: {
      workspaceId: "workspace-export",
      snapshotId: "workspace-export@snapshot-1",
      capturedAt: "2026-03-06T01:00:00.000Z",
      rootDir: "/workspace/export",
      revision: "workspace-export-rev"
    },
    normalizedEventExport: FIXTURE_NORMALIZED_EVENT_EXPORT,
    learnedRouting: true,
    structuralOps: {
      split: 1,
      connect: 1
    }
  });

  assert.equal(result.summary.eventExportDigest, FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest);
  assert.equal(result.summary.workspaceSnapshot, "workspace-export@snapshot-1");
  assert.equal(descriptor.manifest.provenance.eventRange.firstEventId, "evt-interaction-fixture-1");
  assert.equal(descriptor.manifest.provenance.eventRange.lastEventId, "evt-feedback-fixture-2");
  assert.equal(descriptor.graph.blocks.some((block) => block.id.endsWith("evt-feedback-fixture-1")), true);
  assert.match(descriptor.graph.blocks.map((block) => block.source).join("\n"), /openclaw\/runtime\/whatsapp:teaching/);
});
