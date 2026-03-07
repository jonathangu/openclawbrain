import assert from "node:assert/strict";
import test from "node:test";

import {
  createWorkspaceMetadata,
  validateWorkspaceMetadata,
  workspaceManifestDigest,
  workspaceSnapshotId
} from "@openclawbrain/workspace-metadata";

test("workspace metadata package normalizes snapshot inputs deterministically", () => {
  const metadata = createWorkspaceMetadata({
    workspaceId: "workspace-ts",
    snapshotId: "workspace-ts@snapshot-1",
    capturedAt: "2026-03-06T03:00:00.000Z",
    rootDir: "/workspace/ts",
    branch: "main",
    revision: "workspace-rev-1",
    labels: ["typescript", "typescript", "public"],
    files: ["README.md", "packages/contracts/src/index.ts", "README.md"]
  });

  assert.deepEqual(validateWorkspaceMetadata(metadata), []);
  assert.deepEqual(metadata.labels, ["typescript", "public"]);
  assert.deepEqual(metadata.files, ["README.md", "packages/contracts/src/index.ts"]);
  assert.equal(workspaceSnapshotId(metadata), "workspace-ts@snapshot-1");
  assert.equal(workspaceManifestDigest(metadata), metadata.manifestDigest);
  assert.match(metadata.manifestDigest ?? "", /^sha256-/);
});
