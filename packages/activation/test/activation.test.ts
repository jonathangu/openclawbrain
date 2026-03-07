import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  activatePack,
  describeActivationTarget,
  inspectActivationState,
  loadActivationPointers,
  loadPackFromActivation,
  promoteCandidatePack,
  stageCandidatePack
} from "@openclawbrain/activation";
import {
  FIXTURE_ARTIFACT_MANIFEST,
  FIXTURE_PACK_GRAPH,
  FIXTURE_PACK_VECTORS,
  FIXTURE_ROUTER_ARTIFACT,
  type ArtifactManifestV1,
  type PackGraphPayloadV1,
  type PackVectorsPayloadV1,
  type RouterArtifactV1
} from "@openclawbrain/contracts";
import { PACK_LAYOUT, computePayloadChecksum, writePackFile } from "@openclawbrain/pack-format";

function materializePack(rootDir: string, options: { packId: string; learnedRouting: boolean; eventStart: number; snapshotId: string; revision: string | null }): void {
  const graph: PackGraphPayloadV1 = {
    packId: options.packId,
    blocks: FIXTURE_PACK_GRAPH.blocks.map((block, index) => ({
      ...block,
      id: `${options.packId}:block:${index}`
    }))
  };
  const vectors: PackVectorsPayloadV1 = {
    packId: options.packId,
    entries: graph.blocks.map((block, index) => {
      const template = FIXTURE_PACK_VECTORS.entries[index] ?? FIXTURE_PACK_VECTORS.entries[0];
      if (template === undefined) {
        throw new Error("fixture vectors are required for activation surface test");
      }

      return {
        ...template,
        blockId: block.id
      };
    })
  };
  const router: RouterArtifactV1 | null = options.learnedRouting
    ? {
        ...FIXTURE_ROUTER_ARTIFACT,
        routerIdentity: `${options.packId}:route_fn`
      }
    : null;
  const manifest: ArtifactManifestV1 = {
    ...FIXTURE_ARTIFACT_MANIFEST,
    packId: options.packId,
    routePolicy: options.learnedRouting ? "requires_learned_routing" : "heuristic_allowed",
    runtimeAssets: {
      graphPath: PACK_LAYOUT.graph,
      vectorPath: PACK_LAYOUT.vectors,
      router: options.learnedRouting
        ? {
            kind: "artifact",
            identity: router?.routerIdentity ?? null,
            artifactPath: PACK_LAYOUT.router
          }
        : {
            kind: "none",
            identity: null,
            artifactPath: null
          }
    },
    payloadChecksums: {
      graph: computePayloadChecksum(graph),
      vector: computePayloadChecksum(vectors),
      router: router === null ? null : computePayloadChecksum(router)
    },
    modelFingerprints: options.learnedRouting
      ? ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", router?.routerIdentity ?? "router:missing"]
      : ["BAAI/bge-large-en-v1.5"],
    provenance: {
      ...FIXTURE_ARTIFACT_MANIFEST.provenance,
      workspace: {
        ...FIXTURE_ARTIFACT_MANIFEST.provenance.workspace,
        snapshotId: options.snapshotId,
        revision: options.revision
      },
      workspaceSnapshot: options.snapshotId,
      eventRange: {
        start: options.eventStart,
        end: options.eventStart + 1,
        count: 2,
        firstEventId: null,
        lastEventId: null,
        firstCreatedAt: null,
        lastCreatedAt: null
      },
      eventExports: null
    }
  };

  writePackFile(rootDir, PACK_LAYOUT.graph, graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, vectors);
  if (router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, manifest);
}

test("activation package exposes staging and promotion with workspace provenance", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-surface-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-surface-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-surface-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  materializePack(activeRoot, {
    packId: "pack-active-surface",
    learnedRouting: false,
    eventStart: 10,
    snapshotId: "workspace-active@snapshot-1",
    revision: "workspace-active-rev"
  });
  materializePack(candidateRoot, {
    packId: "pack-candidate-surface",
    learnedRouting: true,
    eventStart: 20,
    snapshotId: "workspace-candidate@snapshot-2",
    revision: "workspace-candidate-rev"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T05:00:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T05:05:00.000Z");

  const inspection = inspectActivationState(activationRoot, "2026-03-06T05:06:00.000Z");
  assert.equal(inspection.active?.workspaceSnapshot, "workspace-active@snapshot-1");
  assert.equal(inspection.candidate?.workspaceSnapshot, "workspace-candidate@snapshot-2");
  assert.equal(inspection.promotion.allowed, true);

  promoteCandidatePack(activationRoot, "2026-03-06T05:10:00.000Z");

  const pointers = loadActivationPointers(activationRoot).pointers;
  const activeTarget = describeActivationTarget(activationRoot, "active", { requireActivationReady: true });
  const activePack = loadPackFromActivation(activationRoot, "active", { requireActivationReady: true });
  assert.equal(pointers.active?.workspaceSnapshot, "workspace-candidate@snapshot-2");
  assert.equal(pointers.previous?.workspaceSnapshot, "workspace-active@snapshot-1");
  assert.equal(activeTarget?.workspaceSnapshot, "workspace-candidate@snapshot-2");
  assert.equal(activePack?.manifest.packId, pointers.active?.packId);
});
