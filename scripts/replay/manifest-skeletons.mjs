import { createHash } from "node:crypto";
import path from "node:path";

export const REPLAY_MANIFEST_SKELETON_CONTRACT = "openclawbrain_replay_manifest_skeleton.v1";
export const REPLAY_MANIFEST_SKELETON_SET_CONTRACT = "openclawbrain_replay_manifest_skeleton_set.v1";
export const PROOF_MANIFEST_SKELETON_CONTRACT = "openclawbrain_proof_manifest_skeleton.v1";
export const PROOF_MANIFEST_SMOKE_CONTRACT = "openclawbrain_proof_manifest_smoke.v1";
export const PROOF_CRON_MANIFEST_LAYOUT = {
  manifest: "manifest.json",
  replayManifests: "replay-manifests.json",
  smoke: "manifest-smoke.json",
};

export function renderJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

export function sha256Text(value) {
  return `sha256-${createHash("sha256").update(String(value ?? ""), "utf8").digest("hex")}`;
}

function toPosixPath(value) {
  return String(value ?? "").split(path.sep).join("/");
}

function normalizeOptionalString(value) {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function firstString(...values) {
  for (const value of values) {
    const normalized = normalizeOptionalString(value);
    if (normalized !== null) {
      return normalized;
    }
  }
  return null;
}

function allStringsMatch(values) {
  const filtered = values
    .map((value) => normalizeOptionalString(value))
    .filter((value) => value !== null);
  return filtered.length > 0 && new Set(filtered).size === 1;
}

function joinRelativePath(root, child) {
  const normalizedChild = toPosixPath(child ?? "");
  if (typeof root !== "string" || root.length === 0) {
    return normalizedChild;
  }
  return path.posix.join(toPosixPath(root), normalizedChild);
}

function buildReleaseCloseoutLink() {
  return {
    relation: "release-closeout",
    linkFormat: "workspace_relative_path+sha256",
    state: "unlinked",
    relativePath: null,
    contract: null,
    digest: null,
    note: "reserved linkage slot only; this lane does not write a release-closeout manifest",
  };
}

function buildDigestIndex(entries) {
  const index = new Map();
  for (const entry of Array.isArray(entries) ? entries : []) {
    const relativePath = normalizeOptionalString(entry?.path);
    const digest = normalizeOptionalString(entry?.digest);
    if (relativePath !== null && digest !== null) {
      index.set(relativePath, digest);
    }
  }
  return index;
}

function buildReplayLightweightReference(item) {
  return {
    traceId: item.replay.traceId,
    proofBundleRelativePath: item.proofBundle.relativePath,
    fixtureHash: item.replay.fixture.fixtureHash,
    bundleHash: item.replay.bundle.bundleHash,
    scoreHash: item.replay.bundle.scoreHash,
    proofManifestDigest: item.proofBundle.files.manifest.digest,
  };
}

export function buildReplayManifestSkeleton(bundle) {
  const manifest = bundle?.manifest ?? null;
  const fixture = bundle?.fixture ?? null;
  const hashLedger = bundle?.hashes ?? null;
  const digestIndex = buildDigestIndex(hashLedger?.files);

  const tracePath = firstString(manifest?.files?.trace, "trace.json");
  const fixturePath = firstString(manifest?.files?.fixture, "fixture.json");
  const bundlePath = firstString(manifest?.files?.bundle, "bundle.json");
  const hashesPath = firstString(manifest?.files?.hashes, "hashes.json");
  const manifestPath = "manifest.json";
  const summaryPath = "summary.md";
  const validationPath = "validation-report.json";

  const traceHash = firstString(hashLedger?.semantic?.traceHash, hashLedger?.traceHash, manifest?.hashes?.traceHash, bundle?.metrics?.traceHash);
  const fixtureHash = firstString(fixture?.fixtureHash, manifest?.hashes?.fixtureHash, hashLedger?.semantic?.fixtureHash, hashLedger?.fixtureHash, bundle?.metrics?.fixtureHash);
  const replayFixtureHash = firstString(manifest?.hashes?.fixtureHash, hashLedger?.semantic?.fixtureHash, hashLedger?.fixtureHash, bundle?.metrics?.fixtureHash);
  const bundleHash = firstString(manifest?.hashes?.bundleHash, hashLedger?.semantic?.bundleHash, hashLedger?.bundleHash, bundle?.metrics?.bundleHash);
  const replayBundleHash = firstString(bundle?.metrics?.bundleHash, hashLedger?.semantic?.bundleHash, hashLedger?.bundleHash);
  const scoreHash = firstString(manifest?.hashes?.scoreHash, hashLedger?.semantic?.scoreHash, hashLedger?.scoreHash, bundle?.metrics?.scoreHash);
  const replayScoreHash = firstString(bundle?.metrics?.scoreHash, hashLedger?.semantic?.scoreHash, hashLedger?.scoreHash);
  const manifestDigest = firstString(digestIndex.get(manifestPath), manifest ? sha256Text(renderJson(manifest)) : null);
  const hashLedgerManifestDigest = firstString(digestIndex.get(manifestPath));
  const hashAlgorithm = firstString(manifest?.hashAlgorithm, hashLedger?.algorithm, "sha256");

  return {
    contract: REPLAY_MANIFEST_SKELETON_CONTRACT,
    kind: "recorded-session-replay",
    hashAlgorithm,
    canonicalAt: bundle?.canonicalAt ?? null,
    contracts: {
      replayManifest: manifest?.contract ?? null,
      trace: manifest?.contracts?.trace ?? null,
      fixture: manifest?.contracts?.fixture ?? fixture?.contract ?? null,
      replayBundle: manifest?.contracts?.bundle ?? null,
      hashLedger: hashLedger?.contract ?? manifest?.contracts?.hashes ?? null,
    },
    replay: {
      traceId: firstString(bundle?.metrics?.traceId, manifest?.traceId) ?? bundle?.bundleId ?? null,
      source: manifest?.source ?? null,
      recordedAt: manifest?.recordedAt ?? null,
      generatedAt: manifest?.generatedAt ?? null,
      winnerMode: bundle?.metrics?.winnerMode ?? null,
      modeOrder: Array.isArray(manifest?.modeOrder) ? manifest.modeOrder : [],
      trace: {
        relativePath: joinRelativePath(bundle?.relativePath ?? "", tracePath),
        traceHash,
        fileDigest: firstString(digestIndex.get(tracePath)),
      },
      fixture: {
        relativePath: joinRelativePath(bundle?.relativePath ?? "", fixturePath),
        fixtureHash,
        fileDigest: firstString(digestIndex.get(fixturePath)),
      },
      bundle: {
        relativePath: joinRelativePath(bundle?.relativePath ?? "", bundlePath),
        bundleHash,
        scoreHash,
        fileDigest: firstString(digestIndex.get(bundlePath)),
      },
    },
    proofBundle: {
      relativePath: bundle?.relativePath ?? null,
      validationOk: bundle?.validationOk ?? null,
      files: {
        manifest: {
          relativePath: joinRelativePath(bundle?.relativePath ?? "", manifestPath),
          digest: manifestDigest,
        },
        hashLedger: {
          relativePath: joinRelativePath(bundle?.relativePath ?? "", hashesPath),
          contract: hashLedger?.contract ?? null,
          algorithm: hashLedger?.algorithm ?? null,
        },
        summary: {
          relativePath: joinRelativePath(bundle?.relativePath ?? "", summaryPath),
        },
        validation: {
          relativePath: joinRelativePath(bundle?.relativePath ?? "", validationPath),
        },
      },
    },
    linkage: {
      traceToReplay: {
        linked: allStringsMatch([traceHash, manifest?.hashes?.traceHash, hashLedger?.semantic?.traceHash, bundle?.metrics?.traceHash]),
        traceHash,
        proofManifestTraceHash: firstString(manifest?.hashes?.traceHash),
        hashLedgerTraceHash: firstString(hashLedger?.semantic?.traceHash),
      },
      fixtureToReplay: {
        linked: allStringsMatch([fixtureHash, replayFixtureHash, hashLedger?.semantic?.fixtureHash, bundle?.metrics?.fixtureHash]),
        fixtureHash,
        replayFixtureHash,
        hashLedgerFixtureHash: firstString(hashLedger?.semantic?.fixtureHash),
      },
      replayToProofManifest: {
        linked: allStringsMatch([bundleHash, replayBundleHash, manifest?.hashes?.bundleHash, hashLedger?.semantic?.bundleHash])
          && allStringsMatch([scoreHash, replayScoreHash, manifest?.hashes?.scoreHash, hashLedger?.semantic?.scoreHash]),
        bundleHashLinked: allStringsMatch([bundleHash, replayBundleHash, manifest?.hashes?.bundleHash, hashLedger?.semantic?.bundleHash]),
        scoreHashLinked: allStringsMatch([scoreHash, replayScoreHash, manifest?.hashes?.scoreHash, hashLedger?.semantic?.scoreHash]),
        replayBundleHash,
        proofManifestBundleHash: firstString(manifest?.hashes?.bundleHash),
        hashLedgerBundleHash: firstString(hashLedger?.semantic?.bundleHash),
        replayScoreHash,
        proofManifestScoreHash: firstString(manifest?.hashes?.scoreHash),
        hashLedgerScoreHash: firstString(hashLedger?.semantic?.scoreHash),
      },
      manifestToHashLedger: {
        linked: allStringsMatch([manifestDigest, hashLedgerManifestDigest]),
        manifestDigest,
        hashLedgerManifestDigest,
      },
    },
    releaseCloseout: buildReleaseCloseoutLink(),
  };
}

export function buildReplayManifestLinkageSummary(items) {
  return {
    traceToReplayLinkedCount: items.filter((item) => item.linkage.traceToReplay.linked).length,
    fixtureToReplayLinkedCount: items.filter((item) => item.linkage.fixtureToReplay.linked).length,
    replayToProofManifestLinkedCount: items.filter((item) => item.linkage.replayToProofManifest.linked).length,
    manifestToHashLedgerLinkedCount: items.filter((item) => item.linkage.manifestToHashLedger.linked).length,
    releaseCloseoutLinkedCount: items.filter((item) => item.releaseCloseout.state === "linked").length,
    releaseCloseoutPendingCount: items.filter((item) => item.releaseCloseout.state !== "linked").length,
  };
}

export function buildReplayManifestSkeletonSet(bundles) {
  const items = (Array.isArray(bundles) ? bundles : [])
    .filter((bundle) => bundle?.kind === "recorded-session-replay")
    .map((bundle) => buildReplayManifestSkeleton(bundle))
    .sort((left, right) => {
      const leftMs = left.canonicalAt ? Date.parse(left.canonicalAt) : 0;
      const rightMs = right.canonicalAt ? Date.parse(right.canonicalAt) : 0;
      if (rightMs !== leftMs) {
        return rightMs - leftMs;
      }
      return String(left.proofBundle.relativePath ?? "").localeCompare(String(right.proofBundle.relativePath ?? ""));
    });

  return {
    contract: REPLAY_MANIFEST_SKELETON_SET_CONTRACT,
    hashAlgorithm: "sha256",
    count: items.length,
    traceIds: items.map((item) => item.replay.traceId),
    linkageSummary: buildReplayManifestLinkageSummary(items),
    items,
  };
}

export function buildProofManifestSkeleton(input) {
  const replayManifestSet = input?.replayManifestSet ?? buildReplayManifestSkeletonSet([]);
  return {
    contract: PROOF_MANIFEST_SKELETON_CONTRACT,
    runKind: input?.runKind ?? null,
    generatedAt: input?.generatedAt ?? null,
    hashAlgorithm: "sha256",
    generator: {
      entrypoint: "scripts/proof-cron.mjs",
      sourceContract: input?.sourceContract ?? null,
    },
    output: {
      relativePath: toPosixPath(input?.outputRelativePath ?? ""),
      primary: {
        role: input?.primary?.role ?? null,
        path: input?.primary?.path ?? null,
        digest: input?.primary?.digest ?? null,
        contract: input?.primary?.contract ?? null,
      },
      supporting: Array.isArray(input?.supporting)
        ? input.supporting.map((entry) => ({
            role: entry?.role ?? null,
            path: entry?.path ?? null,
            digest: entry?.digest ?? null,
            contract: entry?.contract ?? null,
          }))
        : [],
      replayManifests: {
        path: PROOF_CRON_MANIFEST_LAYOUT.replayManifests,
        contract: replayManifestSet.contract,
        digest: input?.replayManifestsDigest ?? null,
      },
      smoke: {
        path: PROOF_CRON_MANIFEST_LAYOUT.smoke,
        contract: PROOF_MANIFEST_SMOKE_CONTRACT,
      },
    },
    bundleInventory: {
      totalBundles: input?.bundleInventory?.totalBundles ?? 0,
      operatorProofCount: input?.bundleInventory?.operatorProofCount ?? 0,
      replayProofCount: input?.bundleInventory?.replayProofCount ?? 0,
      hostEvidenceCount: input?.bundleInventory?.hostEvidenceCount ?? 0,
      genericProofCount: input?.bundleInventory?.genericProofCount ?? 0,
      validationOkCount: input?.bundleInventory?.validationOkCount ?? 0,
      validationFailCount: input?.bundleInventory?.validationFailCount ?? 0,
      validationUnknownCount: input?.bundleInventory?.validationUnknownCount ?? 0,
    },
    replayInputs: {
      count: replayManifestSet.count,
      traceIds: replayManifestSet.traceIds,
      linkageSummary: replayManifestSet.linkageSummary,
      items: replayManifestSet.items.map((item) => buildReplayLightweightReference(item)),
    },
    releaseCloseout: buildReleaseCloseoutLink(),
  };
}

export function buildProofManifestSmoke(input) {
  return {
    contract: PROOF_MANIFEST_SMOKE_CONTRACT,
    runKind: input?.runKind ?? null,
    generatedAt: input?.generatedAt ?? null,
    hashAlgorithm: "sha256",
    output: {
      manifestPath: PROOF_CRON_MANIFEST_LAYOUT.manifest,
      manifestDigest: input?.manifestDigest ?? null,
      replayManifestsPath: PROOF_CRON_MANIFEST_LAYOUT.replayManifests,
      replayManifestsDigest: input?.replayManifestsDigest ?? null,
      primary: {
        role: input?.primary?.role ?? null,
        path: input?.primary?.path ?? null,
        digest: input?.primary?.digest ?? null,
      },
      supporting: Array.isArray(input?.supporting)
        ? input.supporting.map((entry) => ({
            role: entry?.role ?? null,
            path: entry?.path ?? null,
            digest: entry?.digest ?? null,
          }))
        : [],
    },
    bundleInventory: {
      totalBundles: input?.bundleInventory?.totalBundles ?? 0,
      operatorProofCount: input?.bundleInventory?.operatorProofCount ?? 0,
      replayProofCount: input?.bundleInventory?.replayProofCount ?? 0,
      hostEvidenceCount: input?.bundleInventory?.hostEvidenceCount ?? 0,
      genericProofCount: input?.bundleInventory?.genericProofCount ?? 0,
    },
    replayInputs: {
      count: input?.replayManifestSet?.count ?? 0,
      traceIds: Array.isArray(input?.replayManifestSet?.traceIds) ? input.replayManifestSet.traceIds : [],
      linkageSummary: input?.replayManifestSet?.linkageSummary ?? buildReplayManifestLinkageSummary([]),
      allReplayHashesLinked:
        (input?.replayManifestSet?.linkageSummary?.fixtureToReplayLinkedCount ?? 0) === (input?.replayManifestSet?.count ?? 0)
        && (input?.replayManifestSet?.linkageSummary?.replayToProofManifestLinkedCount ?? 0) === (input?.replayManifestSet?.count ?? 0)
        && (input?.replayManifestSet?.linkageSummary?.manifestToHashLedgerLinkedCount ?? 0) === (input?.replayManifestSet?.count ?? 0),
    },
    releaseCloseout: {
      linkedCount: input?.replayManifestSet?.linkageSummary?.releaseCloseoutLinkedCount ?? 0,
      pendingCount: input?.replayManifestSet?.linkageSummary?.releaseCloseoutPendingCount ?? 0,
    },
  };
}
