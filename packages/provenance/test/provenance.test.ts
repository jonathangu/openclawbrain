import assert from "node:assert/strict";
import test from "node:test";

import { FIXTURE_NORMALIZED_EVENT_EXPORT } from "@openclawbrain/contracts";
import { buildArtifactProvenance, validateArtifactProvenance } from "@openclawbrain/provenance";

test("provenance package builds validated pack provenance from workspace and event exports", () => {
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
});
