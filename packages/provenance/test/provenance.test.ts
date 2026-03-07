import assert from "node:assert/strict";
import test from "node:test";

import { FIXTURE_NORMALIZED_EVENT_EXPORT } from "@openclawbrain/contracts";
import { buildArtifactProvenance, validateArtifactProvenance } from "@openclawbrain/provenance";

test("provenance package builds validated pack provenance from workspace, event exports, and learning surfaces", () => {
  const provenance = buildArtifactProvenance({
    workspace: {
      workspaceId: "workspace-prov",
      snapshotId: "workspace-prov@snapshot-1",
      capturedAt: "2026-03-06T04:00:00.000Z",
      rootDir: "/workspace/prov",
      revision: "workspace-prov-rev"
    },
    eventRange: FIXTURE_NORMALIZED_EVENT_EXPORT.range,
    eventExports: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance,
    builtAt: "2026-03-06T04:30:00.000Z",
    offlineArtifacts: ["feedback_events.v1", "route_labels.v1"]
  });

  assert.deepEqual(validateArtifactProvenance(provenance), []);
  assert.equal(provenance.workspaceSnapshot, provenance.workspace.snapshotId);
  assert.equal(provenance.eventExports?.exportDigest, FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest);
  assert.deepEqual(provenance.learningSurface, FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.learningSurface);
});

test("provenance defaults to fast-boot passive learning surfaces when no event export is attached", () => {
  const provenance = buildArtifactProvenance({
    workspace: {
      workspaceId: "workspace-default",
      snapshotId: "workspace-default@snapshot-1",
      capturedAt: "2026-03-06T05:00:00.000Z",
      rootDir: "/workspace/default"
    },
    eventRange: {
      start: 0,
      end: -1,
      count: 0,
      firstEventId: null,
      lastEventId: null,
      firstCreatedAt: null,
      lastCreatedAt: null
    },
    builtAt: "2026-03-06T05:30:00.000Z",
    offlineArtifacts: ["feedback_events.v1"]
  });

  assert.equal(provenance.learningSurface.bootProfile, "fast_boot_defaults");
  assert.equal(provenance.learningSurface.learningCadence, "passive_background");
  assert.match(provenance.learningSurface.scanSurfaces.join("\n"), /workspace:workspace-default/);
});
