import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { exportGraphifySourceBundle } from "../src/import-export.js";
import { buildOpenClawSessionCorpusSnapshot } from "../src/session-tail.js";

function makeTempDir() {
  const dir = path.join(tmpdir(), `ocb-source-bundle-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

function hashJsonText(text) {
  return `sha256:${createHash("sha256").update(text.replace(/\n$/u, "")).digest("hex")}`;
}

test("buildOpenClawSessionCorpusSnapshot normalizes a session-store corpus into canonical events", () => {
  const root = makeTempDir();
  const openclawHome = path.join(root, ".openclaw-smoke");
  const sessionsDir = path.join(openclawHome, "agents", "main", "sessions");
  mkdirSync(sessionsDir, { recursive: true });

  const sessionFile = path.join(sessionsDir, "smoke-session.jsonl");
  try {
    writeFileSync(
      path.join(openclawHome, "openclaw.json"),
      JSON.stringify({ profile: "smoke" }, null, 2),
      "utf8",
    );
    writeFileSync(
      sessionFile,
      [
        JSON.stringify({
          type: "session",
          version: 1,
          id: "session-1",
          timestamp: "2026-04-01T00:00:00.000Z",
          cwd: "/tmp",
        }),
        JSON.stringify({
          type: "message",
          id: "msg-1",
          parentId: null,
          timestamp: "2026-04-01T00:00:01.000Z",
          message: {
            role: "assistant",
            content: "Canonical export ready",
            timestamp: 1711929601000,
          },
        }),
        JSON.stringify({
          type: "message",
          id: "msg-2",
          parentId: "msg-1",
          timestamp: "2026-04-01T00:00:02.000Z",
          message: {
            role: "user",
            content: "Please export the corpus",
            timestamp: 1711929602000,
          },
        }),
      ].join("\n") + "\n",
      "utf8",
    );
    writeFileSync(
      path.join(sessionsDir, "sessions.json"),
      JSON.stringify(
        {
          smoke: {
            sessionId: "session-1",
            sessionFile,
            updatedAt: 1,
            chatType: "telegram",
            origin: "test",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    const snapshot = buildOpenClawSessionCorpusSnapshot({
      homeDir: root,
      profileRoots: [openclawHome],
      observedAt: "2026-04-01T00:00:03.000Z",
    });

    assert.equal(snapshot.lane, "local_session_tail");
    assert.equal(snapshot.sourceSummaries.length, 1);
    assert.equal(snapshot.sourceSummaries[0].profileRoot, openclawHome);
    assert.equal(snapshot.normalizedEventExport === null, false);
    assert.ok(snapshot.normalizedEventExport?.provenance.exportDigest.startsWith("sha256-"));
    assert.ok(snapshot.corpusId.startsWith("graphify-source-corpus:"));
    assert.ok(snapshot.sourceSummaries[0].eventCounts.total > 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("exportGraphifySourceBundle writes a canonical machine bundle with stable digests", () => {
  const root = makeTempDir();
  const openclawHome = path.join(root, ".openclaw-smoke");
  const activationRoot = path.join(root, ".openclawbrain", "activation");
  const sessionsDir = path.join(openclawHome, "agents", "main", "sessions");
  const outputDir = path.join(root, "artifacts", "graphify-source-bundles", "run-smoke");
  mkdirSync(sessionsDir, { recursive: true });
  mkdirSync(path.join(activationRoot, "attachment-truth"), { recursive: true });

  const sessionFile = path.join(sessionsDir, "smoke-session.jsonl");
  try {
    writeFileSync(
      path.join(openclawHome, "openclaw.json"),
      JSON.stringify({ profile: "smoke" }, null, 2),
      "utf8",
    );
    writeFileSync(
      sessionFile,
      [
        JSON.stringify({
          type: "session",
          version: 1,
          id: "session-1",
          timestamp: "2026-04-01T00:00:00.000Z",
          cwd: "/tmp",
        }),
        JSON.stringify({
          type: "message",
          id: "msg-1",
          parentId: null,
          timestamp: "2026-04-01T00:00:01.000Z",
          message: {
            role: "assistant",
            content: "Canonical export ready",
            timestamp: 1711929601000,
          },
        }),
      ].join("\n") + "\n",
      "utf8",
    );
    writeFileSync(
      path.join(sessionsDir, "sessions.json"),
      JSON.stringify(
        {
          smoke: {
            sessionId: "session-1",
            sessionFile,
            updatedAt: 1,
            chatType: "telegram",
            origin: "test",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    const result = exportGraphifySourceBundle({
      openclawHome,
      activationRoot,
      outputDir,
      observedAt: "2026-04-01T00:00:03.000Z",
    });

    assert.equal(result.ok, true);
    assert.equal(result.bundleDir, path.resolve(outputDir));
    assert.equal(result.bundleId, result.corpusId);
    assert.equal(result.corpusManifest.contract, "graphify_source_bundle_manifest.v1");
    assert.equal(result.runtimeStatus.contract, "graphify_source_runtime_status.v1");
    assert.equal(result.workspaceMetadata.contract, "graphify_source_workspace_metadata.v1");
    assert.ok(result.corpusDigest.startsWith("sha256:"));
    assert.equal(result.corpusManifest.corpusDigest, result.corpusDigest);
    assert.equal(result.corpusManifest.fileDigests.runtimeStatus, hashJsonText(readFileSync(result.outputPaths.runtimeStatus, "utf8")));
    assert.equal(result.corpusManifest.fileDigests.workspaceMetadata, hashJsonText(readFileSync(result.outputPaths.workspaceMetadata, "utf8")));
    assert.equal(result.corpusManifest.fileDigests.normalizedEventExport, hashJsonText(readFileSync(result.outputPaths.normalizedEventExport, "utf8")));
    assert.ok(result.outputPaths.proofFiles["session-tail.json"]);
    assert.ok(result.outputPaths.proofFiles["runtime-load-proofs.json"]);
    assert.ok(readFileSync(result.outputPaths.corpusManifest, "utf8").includes("graphify_source_bundle_manifest.v1"));
    assert.equal(result.runtimeStatus.sessionTail.emittedEventCount > 0, true);
    assert.equal(result.workspaceMetadata.sourceBundleRunId, "run-smoke");
    assert.equal(result.workspaceMetadata.corpusDigest, result.corpusDigest);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
