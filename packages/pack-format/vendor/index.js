import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { buildServedArtifactProof, CONTRACT_IDS, canonicalJson, checksumJsonPayload, validateActivationPointers, validateArtifactManifest, validatePackGraphPayload, validatePackVectorsPayload, validateRouterArtifact } from "@openclawbrain/contracts";
import { appendLearningSpineLogEntry, buildLearningSpineLogId } from "./learning-spine-logs.js";
export const PACK_LAYOUT = {
    graph: "graph.json",
    manifest: "manifest.json",
    router: "router/model.json",
    vectors: "vectors.json"
};
export const ACTIVATION_LAYOUT = {
    pointers: "activation-pointers.json"
};
function compareIsoDates(left, right) {
    return Date.parse(left) - Date.parse(right);
}
function isStrictlyFresherTarget(candidate, active) {
    return (compareIsoDates(candidate.builtAt, active.builtAt) > 0 ||
        candidate.eventRange.end > active.eventRange.end ||
        candidate.eventRange.count > active.eventRange.count ||
        candidate.workspaceSnapshot !== active.workspaceSnapshot ||
        (candidate.workspaceRevision ?? null) !== (active.workspaceRevision ?? null) ||
        (candidate.eventExportDigest ?? null) !== (active.eventExportDigest ?? null));
}
function promotionFreshnessDelta(active, candidate) {
    return {
        builtAt: compareIsoDates(candidate.builtAt, active.builtAt) > 0,
        eventRangeEnd: candidate.eventRange.end > active.eventRange.end,
        eventRangeCount: candidate.eventRange.count > active.eventRange.count,
        workspaceSnapshot: candidate.workspaceSnapshot !== active.workspaceSnapshot,
        workspaceRevision: (candidate.workspaceRevision ?? null) !== (active.workspaceRevision ?? null),
        eventExportDigest: (candidate.eventExportDigest ?? null) !== (active.eventExportDigest ?? null)
    };
}
function emptyLearnedRouteFnFreshness() {
    return {
        packId: null,
        required: false,
        available: false,
        routerAssetKind: null,
        routerIdentity: null,
        routeFnVersion: null,
        trainingMethod: null,
        routerChecksum: null,
        routerTrainedAt: null,
        packBuiltAt: null,
        workspaceSnapshot: null,
        eventExportDigest: null,
        updateMechanism: null,
        updateVersion: null,
        objective: null,
        pgProfile: null,
        objectiveChecksum: null,
        collectedLabels: null,
        supervisionCount: null,
        updateCount: null,
        weightsChecksum: null,
        freshnessChecksum: null
    };
}
function emptyRouteArtifactDiff() {
    return {
        activePackId: null,
        candidatePackId: null,
        comparable: false,
        routerChanged: false,
        objectiveChanged: false,
        weightsChanged: false,
        freshnessChanged: false,
        labelCountsChanged: false,
        updateCountChanged: false,
        activeRouterChecksum: null,
        candidateRouterChecksum: null,
        activeObjectiveChecksum: null,
        candidateObjectiveChecksum: null,
        activeWeightsChecksum: null,
        candidateWeightsChecksum: null,
        activeFreshnessChecksum: null,
        candidateFreshnessChecksum: null,
        activeCollectedLabels: null,
        candidateCollectedLabels: null,
        activeUpdateCount: null,
        candidateUpdateCount: null,
        activeTopUpdatedBlocks: [],
        candidateTopUpdatedBlocks: []
    };
}
function emptyGraphDynamicsFreshness() {
    return {
        packId: null,
        graphChecksum: null,
        builtAt: null,
        workspaceSnapshot: null,
        eventRange: null,
        eventExportDigest: null,
        runtimePlasticitySource: null,
        bootstrapping: null,
        hebbian: null,
        decay: null,
        structuralOps: null
    };
}
function emptyInitHandoff() {
    return {
        packId: null,
        initMode: null,
        seedStateVisible: false,
        seedBlockCount: 0,
        seedSources: [],
        seedRoles: [],
        handoffState: "missing",
        pgRouteAuthoritative: false,
        learnedRouteUpdateCount: null
    };
}
function isSeedStateBlock(block) {
    if (block.id.includes(":event:") || block.id.includes(":teacher:")) {
        return false;
    }
    if (block.source.startsWith("split:") || block.source.startsWith("merge:")) {
        return false;
    }
    return true;
}
function sortedUnique(values) {
    return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}
export function describePackInitHandoff(packOrRootDir) {
    let pack;
    if (typeof packOrRootDir === "string") {
        try {
            pack = loadPack(packOrRootDir);
        }
        catch {
            return emptyInitHandoff();
        }
    }
    else {
        pack = packOrRootDir;
    }
    const seedBlocks = pack.graph.blocks.filter(isSeedStateBlock);
    const learnedRouteUpdateCount = pack.router?.training.updateCount ?? null;
    const pgRouteAuthoritative = pack.router !== null &&
        pack.manifest.graphDynamics.runtimePlasticitySource === "live_loop" &&
        pack.router.training.objective.updateMechanism === "policy_gradient";
    return {
        packId: pack.manifest.packId,
        initMode: pack.manifest.provenance.learningSurface.bootProfile,
        seedStateVisible: seedBlocks.length > 0,
        seedBlockCount: seedBlocks.length,
        seedSources: sortedUnique(seedBlocks.map((block) => block.source)),
        seedRoles: sortedUnique(seedBlocks.map((block) => block.learning.role)),
        handoffState: pgRouteAuthoritative ? "pg_promoted_pack_authoritative" : "seed_state_authoritative",
        pgRouteAuthoritative,
        learnedRouteUpdateCount
    };
}
function describeLearnedRouteFnFreshness(pack) {
    if (pack === null) {
        return emptyLearnedRouteFnFreshness();
    }
    return {
        packId: pack.manifest.packId,
        required: pack.manifest.routePolicy === "requires_learned_routing",
        available: pack.router !== null,
        routerAssetKind: pack.manifest.runtimeAssets.router.kind,
        routerIdentity: pack.manifest.runtimeAssets.router.identity,
        routeFnVersion: pack.router?.strategy ?? null,
        trainingMethod: pack.router?.training.method ?? null,
        routerChecksum: pack.manifest.payloadChecksums.router,
        routerTrainedAt: pack.router?.trainedAt ?? null,
        packBuiltAt: pack.manifest.provenance.builtAt,
        workspaceSnapshot: pack.manifest.provenance.workspaceSnapshot,
        eventExportDigest: pack.manifest.provenance.eventExports?.exportDigest ?? null,
        updateMechanism: pack.router?.training.objective.updateMechanism ?? null,
        updateVersion: pack.router?.training.objective.updateVersion ?? null,
        objective: pack.router?.training.objective.objective ?? null,
        pgProfile: pack.router?.training.objective.profile ?? null,
        objectiveChecksum: pack.router?.training.objective.objectiveChecksum ?? null,
        collectedLabels: pack.router === null ? null : { ...pack.router.training.collectedLabels },
        supervisionCount: pack.router?.training.supervisionCount ?? null,
        updateCount: pack.router?.training.updateCount ?? null,
        weightsChecksum: pack.router?.training.weightsChecksum ?? null,
        freshnessChecksum: pack.router?.training.freshnessChecksum ?? null
    };
}
function describeRouteArtifactDiff(activePack, candidatePack) {
    if (activePack === null && candidatePack === null) {
        return emptyRouteArtifactDiff();
    }
    return {
        activePackId: activePack?.manifest.packId ?? null,
        candidatePackId: candidatePack?.manifest.packId ?? null,
        comparable: activePack?.router !== null && candidatePack?.router !== null,
        routerChanged: (activePack?.manifest.payloadChecksums.router ?? null) !== (candidatePack?.manifest.payloadChecksums.router ?? null),
        objectiveChanged: (activePack?.router?.training.objective.objectiveChecksum ?? null) !==
            (candidatePack?.router?.training.objective.objectiveChecksum ?? null),
        weightsChanged: (activePack?.router?.training.weightsChecksum ?? null) !== (candidatePack?.router?.training.weightsChecksum ?? null),
        freshnessChanged: (activePack?.router?.training.freshnessChecksum ?? null) !== (candidatePack?.router?.training.freshnessChecksum ?? null),
        labelCountsChanged: JSON.stringify(activePack?.router?.training.collectedLabels ?? null) !==
            JSON.stringify(candidatePack?.router?.training.collectedLabels ?? null),
        updateCountChanged: (activePack?.router?.training.updateCount ?? null) !== (candidatePack?.router?.training.updateCount ?? null),
        activeRouterChecksum: activePack?.manifest.payloadChecksums.router ?? null,
        candidateRouterChecksum: candidatePack?.manifest.payloadChecksums.router ?? null,
        activeObjectiveChecksum: activePack?.router?.training.objective.objectiveChecksum ?? null,
        candidateObjectiveChecksum: candidatePack?.router?.training.objective.objectiveChecksum ?? null,
        activeWeightsChecksum: activePack?.router?.training.weightsChecksum ?? null,
        candidateWeightsChecksum: candidatePack?.router?.training.weightsChecksum ?? null,
        activeFreshnessChecksum: activePack?.router?.training.freshnessChecksum ?? null,
        candidateFreshnessChecksum: candidatePack?.router?.training.freshnessChecksum ?? null,
        activeCollectedLabels: activePack?.router === null || activePack?.router === undefined ? null : { ...activePack.router.training.collectedLabels },
        candidateCollectedLabels: candidatePack?.router === null || candidatePack?.router === undefined ? null : { ...candidatePack.router.training.collectedLabels },
        activeUpdateCount: activePack?.router?.training.updateCount ?? null,
        candidateUpdateCount: candidatePack?.router?.training.updateCount ?? null,
        activeTopUpdatedBlocks: activePack?.router?.policyUpdates.slice(0, 3).map((update) => update.blockId) ?? [],
        candidateTopUpdatedBlocks: candidatePack?.router?.policyUpdates.slice(0, 3).map((update) => update.blockId) ?? []
    };
}
function describeGraphDynamicsFreshness(pack) {
    if (pack === null) {
        return emptyGraphDynamicsFreshness();
    }
    return {
        packId: pack.manifest.packId,
        graphChecksum: pack.manifest.payloadChecksums.graph,
        builtAt: pack.manifest.provenance.builtAt,
        workspaceSnapshot: pack.manifest.provenance.workspaceSnapshot,
        eventRange: {
            start: pack.manifest.provenance.eventRange.start,
            end: pack.manifest.provenance.eventRange.end,
            count: pack.manifest.provenance.eventRange.count
        },
        eventExportDigest: pack.manifest.provenance.eventExports?.exportDigest ?? null,
        runtimePlasticitySource: pack.manifest.graphDynamics.runtimePlasticitySource,
        bootstrapping: { ...pack.manifest.graphDynamics.bootstrapping },
        hebbian: { ...pack.manifest.graphDynamics.hebbian },
        decay: { ...pack.manifest.graphDynamics.decay },
        structuralOps: { ...pack.manifest.graphDynamics.structuralOps }
    };
}
export function summarizeGraphSplitBlocks(pack) {
    const splitBlocks = pack.graph.blocks.filter((block) => block.source.startsWith("split:"));
    const strongestBlockId = pack.graph.evolution?.strongestBlockId ?? null;
    return {
        count: splitBlocks.length,
        blockIds: splitBlocks.map((block) => block.id),
        sources: [...new Set(splitBlocks.map((block) => block.source))],
        strongestBlockId: strongestBlockId !== null && splitBlocks.some((block) => block.id === strongestBlockId) ? strongestBlockId : null
    };
}
function buildPlainGraphEvolutionSummary(input) {
    if (!input.changed) {
        return `materialization kept the block graph stable: ${input.blockCount} live blocks, strongest ${input.strongestBlockId ?? "none"}, no split, merge, prune, or connect ops ran`;
    }
    const liveBlockDetail = input.prePruneBlockCount === input.blockCount
        ? `${input.blockCount} live blocks`
        : `${input.blockCount} live blocks from ${input.prePruneBlockCount} before prune`;
    const connectClause = input.connectDiagnostics === null ||
        (input.connectDiagnostics.appliedPairCount === 0 && input.connectDiagnostics.createdEdgeCount === 0)
        ? ""
        : `; connect joined ${input.connectDiagnostics.appliedPairCount} block pairs and created ${input.connectDiagnostics.createdEdgeCount} edges`;
    return `materialization changed the block graph: ${liveBlockDetail}, strongest ${input.strongestBlockId ?? "none"}, ops split ${input.structuralOps.split} merge ${input.structuralOps.merge} prune ${input.structuralOps.prune} connect ${input.structuralOps.connect}${connectClause}`;
}
export function summarizeStructuralGraphEvolution(input) {
    const operationsApplied = ["split", "merge", "prune", "connect"].filter((key) => input.structuralOps[key] > 0);
    const prePruneBlockCount = input.blockCount + input.prunedBlockCount;
    const changed = operationsApplied.length > 0 || input.prunedBlockCount > 0;
    return {
        changed,
        operationsApplied: [...operationsApplied],
        liveBlockCount: input.blockCount,
        prunedBlockCount: input.prunedBlockCount,
        prePruneBlockCount,
        operatorSummary: buildPlainGraphEvolutionSummary({
            changed,
            blockCount: input.blockCount,
            prePruneBlockCount,
            strongestBlockId: input.strongestBlockId,
            structuralOps: input.structuralOps,
            connectDiagnostics: input.connectDiagnostics
        })
    };
}
function summarizeStructuralEvolution(pack) {
    return summarizeStructuralGraphEvolution({
        blockCount: pack.graph.blocks.length,
        strongestBlockId: pack.graph.evolution?.strongestBlockId ?? null,
        structuralOps: pack.manifest.graphDynamics.structuralOps,
        prunedBlockCount: pack.graph.evolution?.prunedBlockIds.length ?? 0,
        connectDiagnostics: pack.graph.evolution?.connectDiagnostics ?? null
    });
}
export function describeGraphEvolutionLog(pack) {
    const evolution = pack.graph.evolution;
    const manifest = pack.manifest;
    return {
        packId: manifest.packId,
        provenance: manifest.graphDynamics.runtimePlasticitySource,
        builtAt: manifest.provenance.builtAt,
        graphChecksum: manifest.payloadChecksums.graph,
        blockCount: pack.graph.blocks.length,
        structuralOps: { ...manifest.graphDynamics.structuralOps },
        connectDiagnostics: evolution?.connectDiagnostics ?? null,
        structuralEvolutionSummary: summarizeStructuralEvolution(pack),
        prunedBlockIds: evolution?.prunedBlockIds ?? [],
        hebbianSummary: {
            applied: evolution?.hebbianApplied ?? manifest.graphDynamics.hebbian.enabled,
            learningRate: manifest.graphDynamics.hebbian.learningRate
        },
        decaySummary: {
            applied: evolution?.decayApplied ?? manifest.graphDynamics.decay.enabled,
            halfLifeDays: manifest.graphDynamics.decay.halfLifeDays
        },
        strongestBlockId: evolution?.strongestBlockId ?? null,
        eventExportDigest: manifest.provenance.eventExports?.exportDigest ?? null
    };
}
function sha256File(filePath) {
    return `sha256-${createHash("sha256").update(readFileSync(filePath)).digest("hex")}`;
}
function readJsonFile(filePath) {
    return JSON.parse(readFileSync(filePath, "utf8"));
}
function pushFileError(errors, filePath, label) {
    if (!existsSync(filePath)) {
        errors.push(`${label} not found: ${filePath}`);
    }
}
function validatePackAssetPath(assetPath, label) {
    const errors = [];
    if (path.isAbsolute(assetPath)) {
        errors.push(`${label} must be relative to the pack root`);
    }
    const segments = assetPath.split(/[\\/]+/u);
    if (segments.includes("..")) {
        errors.push(`${label} must not escape the pack root`);
    }
    return errors;
}
function resolvePackAssetPath(rootDir, assetPath, label) {
    const resolvedRootDir = path.resolve(rootDir);
    const resolvedAssetPath = path.resolve(resolvedRootDir, assetPath);
    const relativeAssetPath = path.relative(resolvedRootDir, resolvedAssetPath);
    if (relativeAssetPath.startsWith("..") || path.isAbsolute(relativeAssetPath)) {
        throw new Error(`Invalid pack descriptor: ${label} escapes pack root: ${assetPath}`);
    }
    return resolvedAssetPath;
}
export function validatePackDescriptor(manifest) {
    const errors = validateArtifactManifest(manifest);
    errors.push(...validatePackAssetPath(manifest.runtimeAssets.graphPath, "graphPath"));
    errors.push(...validatePackAssetPath(manifest.runtimeAssets.vectorPath, "vectorPath"));
    if (manifest.runtimeAssets.router.artifactPath !== null) {
        errors.push(...validatePackAssetPath(manifest.runtimeAssets.router.artifactPath, "router artifactPath"));
    }
    if (!manifest.runtimeAssets.graphPath.endsWith(".json")) {
        errors.push("graph payload must be json-addressable in the initial layout");
    }
    if (!manifest.runtimeAssets.vectorPath.endsWith(".json")) {
        errors.push("vector payload must be json-addressable in the initial layout");
    }
    if (manifest.runtimeAssets.router.artifactPath !== null && !manifest.runtimeAssets.router.artifactPath.endsWith(".json")) {
        errors.push("router payload must be json-addressable in the initial layout");
    }
    if (manifest.routePolicy === "requires_learned_routing" && manifest.runtimeAssets.router.artifactPath === null) {
        errors.push("learned-routing packs require a router artifact path");
    }
    return errors;
}
export function validatePackActivationReadiness(packOrRootDir) {
    let pack;
    if (typeof packOrRootDir === "string") {
        try {
            pack = loadPack(packOrRootDir);
        }
        catch (error) {
            return [error instanceof Error ? error.message : String(error)];
        }
    }
    else {
        pack = packOrRootDir;
    }
    const errors = [];
    const { manifest, router } = pack;
    if (manifest.routePolicy !== "requires_learned_routing") {
        return errors;
    }
    if (manifest.runtimeAssets.router.kind !== "artifact") {
        errors.push("learned-routing packs require runtimeAssets.router.kind=artifact for activation");
    }
    if (manifest.runtimeAssets.router.identity === null) {
        errors.push("learned-routing packs require runtimeAssets.router.identity for activation");
    }
    if (manifest.runtimeAssets.router.artifactPath === null) {
        errors.push("learned-routing packs require runtimeAssets.router.artifactPath for activation");
    }
    if (manifest.payloadChecksums.router === null) {
        errors.push("learned-routing packs require router checksum metadata for activation");
    }
    if (router === null) {
        errors.push("learned-routing packs require a router artifact for activation");
        return errors;
    }
    if (router.requiresLearnedRouting !== true) {
        errors.push("learned-routing packs require router.requiresLearnedRouting=true for activation");
    }
    if (manifest.runtimeAssets.router.identity !== router.routerIdentity) {
        errors.push(`learned-routing packs require router identity ${manifest.runtimeAssets.router.identity ?? "null"} but found ${router.routerIdentity}`);
    }
    if (manifest.provenance.eventExports !== null &&
        router.training.eventExportDigest !== manifest.provenance.eventExports.exportDigest) {
        errors.push(`learned-routing packs require router event export digest ${manifest.provenance.eventExports.exportDigest} but found ${router.training.eventExportDigest ?? "null"}`);
    }
    return errors;
}
export function computePayloadChecksum(value) {
    return checksumJsonPayload(value);
}
export function writePackFile(rootDir, relativePath, payload) {
    const filePath = path.join(rootDir, relativePath);
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, canonicalJson(payload), "utf8");
    return filePath;
}
function emptyActivationPointers() {
    return {
        contract: CONTRACT_IDS.activationPointers,
        active: null,
        candidate: null,
        previous: null
    };
}
function buildActivationPointerRecord(slot, pack, updatedAt) {
    return {
        slot,
        packId: pack.manifest.packId,
        packRootDir: path.resolve(pack.rootDir),
        manifestPath: path.resolve(pack.manifestPath),
        manifestDigest: sha256File(pack.manifestPath),
        routePolicy: pack.manifest.routePolicy,
        routerIdentity: pack.manifest.runtimeAssets.router.identity,
        workspaceSnapshot: pack.manifest.provenance.workspaceSnapshot,
        workspaceRevision: pack.manifest.provenance.workspace.revision,
        eventRange: {
            start: pack.manifest.provenance.eventRange.start,
            end: pack.manifest.provenance.eventRange.end,
            count: pack.manifest.provenance.eventRange.count
        },
        eventExportDigest: pack.manifest.provenance.eventExports?.exportDigest ?? null,
        builtAt: pack.manifest.provenance.builtAt,
        updatedAt
    };
}
function buildCompileTargetFromPack(pack) {
    return {
        packId: pack.manifest.packId,
        routePolicy: pack.manifest.routePolicy,
        routerIdentity: pack.manifest.runtimeAssets.router.identity,
        workspaceSnapshot: pack.manifest.provenance.workspaceSnapshot,
        workspaceRevision: pack.manifest.provenance.workspace.revision,
        eventRange: {
            start: pack.manifest.provenance.eventRange.start,
            end: pack.manifest.provenance.eventRange.end,
            count: pack.manifest.provenance.eventRange.count
        },
        eventExportDigest: pack.manifest.provenance.eventExports?.exportDigest ?? null,
        builtAt: pack.manifest.provenance.builtAt
    };
}
function buildCompileTargetFromInspection(inspection) {
    return {
        packId: inspection.packId,
        routePolicy: inspection.routePolicy,
        routerIdentity: inspection.routerIdentity,
        workspaceSnapshot: inspection.workspaceSnapshot,
        workspaceRevision: inspection.workspaceRevision,
        eventRange: {
            start: inspection.eventRange.start,
            end: inspection.eventRange.end,
            count: inspection.eventRange.count
        },
        eventExportDigest: inspection.eventExportDigest,
        builtAt: inspection.builtAt
    };
}
function pointerPackIdentityFindings(slot, record, pack) {
    const expected = buildActivationPointerRecord(slot, pack, record.updatedAt);
    const errors = [];
    if (path.resolve(record.packRootDir) !== path.resolve(expected.packRootDir)) {
        errors.push(`pointer packRootDir ${record.packRootDir} does not match pack root ${expected.packRootDir}`);
    }
    if (path.resolve(record.manifestPath) !== path.resolve(expected.manifestPath)) {
        errors.push(`pointer manifestPath ${record.manifestPath} does not match pack manifest ${expected.manifestPath}`);
    }
    if (record.manifestDigest !== expected.manifestDigest) {
        errors.push(`pointer manifestDigest ${record.manifestDigest} does not match pack manifest digest ${expected.manifestDigest}`);
    }
    if (record.routePolicy !== expected.routePolicy) {
        errors.push(`pointer routePolicy ${record.routePolicy} does not match pack routePolicy ${expected.routePolicy}`);
    }
    if (record.routerIdentity !== expected.routerIdentity) {
        errors.push(`pointer routerIdentity ${record.routerIdentity ?? "null"} does not match pack router identity ${expected.routerIdentity ?? "null"}`);
    }
    if (record.workspaceSnapshot !== expected.workspaceSnapshot) {
        errors.push(`pointer workspaceSnapshot ${record.workspaceSnapshot} does not match pack workspaceSnapshot ${expected.workspaceSnapshot}`);
    }
    if ((record.workspaceRevision ?? null) !== (expected.workspaceRevision ?? null)) {
        errors.push(`pointer workspaceRevision ${record.workspaceRevision ?? "null"} does not match pack workspace revision ${expected.workspaceRevision ?? "null"}`);
    }
    if (record.eventRange.start !== expected.eventRange.start) {
        errors.push(`pointer eventRange.start ${record.eventRange.start} does not match pack eventRange.start ${expected.eventRange.start}`);
    }
    if (record.eventRange.end !== expected.eventRange.end) {
        errors.push(`pointer eventRange.end ${record.eventRange.end} does not match pack eventRange.end ${expected.eventRange.end}`);
    }
    if (record.eventRange.count !== expected.eventRange.count) {
        errors.push(`pointer eventRange.count ${record.eventRange.count} does not match pack eventRange.count ${expected.eventRange.count}`);
    }
    if ((record.eventExportDigest ?? null) !== (expected.eventExportDigest ?? null)) {
        errors.push(`pointer eventExportDigest ${record.eventExportDigest ?? "null"} does not match pack event export digest ${expected.eventExportDigest ?? "null"}`);
    }
    if (record.builtAt !== expected.builtAt) {
        errors.push(`pointer builtAt ${record.builtAt} does not match pack builtAt ${expected.builtAt}`);
    }
    return errors;
}
function assertPointerPinnedToPack(slot, record, pack) {
    if (record === null || record.packId !== pack.manifest.packId) {
        return;
    }
    const errors = pointerPackIdentityFindings(slot, record, pack);
    if (errors.length > 0) {
        throw new Error(`${slot} pointer for packId ${record.packId} is already pinned to a different manifest: ${errors.join("; ")}`);
    }
}
function assertRetainedPointerMatchesManifest(slot, record, options = {}) {
    if (record === null) {
        return;
    }
    try {
        ensurePackRecordMatchesManifest(record, {
            requireActivationReady: options.requireActivationReady === true
        });
    }
    catch (error) {
        throw new Error(`${slot} pointer cannot be retained: ${error instanceof Error ? error.message : String(error)}`);
    }
}
function ensurePackRecordMatchesManifest(record, options = {}) {
    const pack = loadPack(path.resolve(record.packRootDir));
    const errors = [];
    if (path.resolve(pack.manifestPath) !== path.resolve(record.manifestPath)) {
        errors.push(`pointer manifestPath ${record.manifestPath} does not match pack manifest ${pack.manifestPath}`);
    }
    const manifestDigest = sha256File(pack.manifestPath);
    if (manifestDigest !== record.manifestDigest) {
        errors.push(`pointer manifestDigest ${record.manifestDigest} does not match pack manifest digest ${manifestDigest}`);
    }
    if (pack.manifest.packId !== record.packId) {
        errors.push(`pointer packId ${record.packId} does not match manifest packId ${pack.manifest.packId}`);
    }
    if (pack.manifest.routePolicy !== record.routePolicy) {
        errors.push(`pointer routePolicy ${record.routePolicy} does not match manifest routePolicy ${pack.manifest.routePolicy}`);
    }
    if (pack.manifest.runtimeAssets.router.identity !== record.routerIdentity) {
        errors.push(`pointer routerIdentity ${record.routerIdentity ?? "null"} does not match manifest router identity ${pack.manifest.runtimeAssets.router.identity ?? "null"}`);
    }
    if (pack.manifest.provenance.workspaceSnapshot !== record.workspaceSnapshot) {
        errors.push(`pointer workspaceSnapshot ${record.workspaceSnapshot} does not match manifest workspaceSnapshot ${pack.manifest.provenance.workspaceSnapshot}`);
    }
    if ((pack.manifest.provenance.workspace.revision ?? null) !== record.workspaceRevision) {
        errors.push(`pointer workspaceRevision ${record.workspaceRevision ?? "null"} does not match manifest workspace revision ${pack.manifest.provenance.workspace.revision ?? "null"}`);
    }
    if (pack.manifest.provenance.eventRange.start !== record.eventRange.start) {
        errors.push(`pointer eventRange.start ${record.eventRange.start} does not match manifest eventRange.start ${pack.manifest.provenance.eventRange.start}`);
    }
    if (pack.manifest.provenance.eventRange.end !== record.eventRange.end) {
        errors.push(`pointer eventRange.end ${record.eventRange.end} does not match manifest eventRange.end ${pack.manifest.provenance.eventRange.end}`);
    }
    if (pack.manifest.provenance.eventRange.count !== record.eventRange.count) {
        errors.push(`pointer eventRange.count ${record.eventRange.count} does not match manifest eventRange.count ${pack.manifest.provenance.eventRange.count}`);
    }
    if ((pack.manifest.provenance.eventExports?.exportDigest ?? null) !== record.eventExportDigest) {
        errors.push(`pointer eventExportDigest ${record.eventExportDigest ?? "null"} does not match manifest event export digest ${pack.manifest.provenance.eventExports?.exportDigest ?? "null"}`);
    }
    if (pack.manifest.provenance.builtAt !== record.builtAt) {
        errors.push(`pointer builtAt ${record.builtAt} does not match manifest builtAt ${pack.manifest.provenance.builtAt}`);
    }
    if (options.requireActivationReady === true) {
        errors.push(...validatePackActivationReadiness(pack));
    }
    if (errors.length > 0) {
        throw new Error(`Invalid activation pointer: ${errors.join("; ")}`);
    }
    return pack;
}
function writeActivationPointers(rootDir, pointers) {
    const errors = validateActivationPointers(pointers);
    if (errors.length > 0) {
        throw new Error(`Invalid activation pointers: ${errors.join("; ")}`);
    }
    const resolvedRootDir = path.resolve(rootDir);
    const pointerPath = path.join(resolvedRootDir, ACTIVATION_LAYOUT.pointers);
    mkdirSync(path.dirname(pointerPath), { recursive: true });
    writeFileSync(pointerPath, canonicalJson(pointers), "utf8");
    return {
        rootDir: resolvedRootDir,
        pointerPath,
        pointers
    };
}
function normalizeActivationMutation(input, defaultUpdatedAt, defaultReason) {
    if (typeof input === "string") {
        return {
            updatedAt: input,
            reason: defaultReason
        };
    }
    const normalizedReason = input?.reason?.trim();
    return {
        updatedAt: input?.updatedAt ?? defaultUpdatedAt,
        reason: normalizedReason === undefined || normalizedReason.length === 0 ? defaultReason : normalizedReason
    };
}
function buildLearningSpineActivationPackSnapshot(record) {
    if (record === null) {
        return null;
    }
    let routerChecksum = null;
    let graphChecksum = null;
    let routerIdentity = record.routerIdentity;
    try {
        const pack = loadPack(path.resolve(record.packRootDir));
        routerChecksum = pack.manifest.payloadChecksums.router;
        graphChecksum = pack.manifest.payloadChecksums.graph;
        routerIdentity = pack.router?.routerIdentity ?? record.routerIdentity;
    }
    catch {
        routerChecksum = null;
        graphChecksum = null;
    }
    return {
        slot: record.slot,
        packId: record.packId,
        routePolicy: record.routePolicy,
        routerIdentity,
        routerChecksum,
        graphChecksum,
        eventExportDigest: record.eventExportDigest,
        builtAt: record.builtAt,
        updatedAt: record.updatedAt
    };
}
function buildLearningSpineActivationPointersSnapshot(pointers) {
    return {
        active: buildLearningSpineActivationPackSnapshot(pointers.active),
        candidate: buildLearningSpineActivationPackSnapshot(pointers.candidate),
        previous: buildLearningSpineActivationPackSnapshot(pointers.previous)
    };
}
function safeAppendPromotionActivationLog(input) {
    try {
        const entryBase = {
            recordType: "promotion_activation",
            operation: input.operation,
            occurredAt: input.occurredAt,
            activationRoot: path.resolve(input.rootDir),
            reason: input.reason,
            promotionReason: input.operation === "promote_candidate_pack" ? input.reason : null,
            rollbackTarget: input.rollbackTarget ?? null,
            before: buildLearningSpineActivationPointersSnapshot(input.before),
            after: buildLearningSpineActivationPointersSnapshot(input.after),
            laterServedTurn: null
        };
        const entry = {
            ...entryBase,
            recordId: buildLearningSpineLogId("promotion-activation", entryBase)
        };
        appendLearningSpineLogEntry(input.rootDir, "promotionActivationEvents", entry);
    }
    catch {
    }
}
function inspectPointerRecord(slot, record) {
    if (record === null) {
        return null;
    }
    const findings = [];
    try {
        const pack = ensurePackRecordMatchesManifest(record, { requireActivationReady: true });
        findings.push(...validatePackActivationReadiness(pack));
    }
    catch (error) {
        findings.push(error instanceof Error ? error.message : String(error));
    }
    return {
        slot,
        packId: record.packId,
        routePolicy: record.routePolicy,
        routerIdentity: record.routerIdentity,
        workspaceSnapshot: record.workspaceSnapshot,
        workspaceRevision: record.workspaceRevision,
        eventRange: record.eventRange,
        eventExportDigest: record.eventExportDigest,
        builtAt: record.builtAt,
        activationReady: findings.length === 0,
        findings
    };
}
function promotionCoherenceFindings(active, candidate) {
    const findings = [];
    if (compareIsoDates(candidate.builtAt, active.builtAt) < 0) {
        findings.push("candidate pack builtAt must not precede active pack builtAt during promotion");
    }
    // Note: eventRange.end is NOT checked here. Quality filters (noise stripping)
    // can legitimately reduce the event pool, producing a candidate with a lower
    // eventRange.end than the active pack. The builtAt check above is sufficient
    // to prevent accidental time-regression. Freshness detection in
    // isStrictlyFresherTarget uses OR-logic across multiple dimensions.
    return findings;
}
function rollbackCoherenceFindings(active, previous) {
    const findings = [];
    if (compareIsoDates(previous.builtAt, active.builtAt) > 0) {
        findings.push("previous pack builtAt must not follow active pack builtAt during rollback");
    }
    if (previous.eventRange.end > active.eventRange.end) {
        findings.push("previous eventRange.end must be <= active eventRange.end during rollback");
    }
    return findings;
}
function duplicatePackIdError(slot, record) {
    return `${slot} pointer cannot reuse packId ${record.packId}`;
}
function previewPromotionPointers(current, updatedAt) {
    const findings = [];
    if (current.candidate === null) {
        findings.push("candidate pointer is required for promotion");
    }
    let candidatePack = null;
    if (current.candidate !== null) {
        try {
            candidatePack = ensurePackRecordMatchesManifest(current.candidate, { requireActivationReady: true });
        }
        catch (error) {
            findings.push(error instanceof Error ? error.message : String(error));
        }
    }
    let activePack = null;
    if (current.active !== null) {
        try {
            activePack = ensurePackRecordMatchesManifest(current.active, { requireActivationReady: true });
        }
        catch (error) {
            findings.push(error instanceof Error ? error.message : String(error));
        }
    }
    if (findings.length > 0 || candidatePack === null) {
        return {
            allowed: false,
            findings,
            nextPointers: null
        };
    }
    if (activePack !== null) {
        findings.push(...promotionCoherenceFindings(buildCompileTargetFromPack(activePack), buildCompileTargetFromPack(candidatePack)));
    }
    if (findings.length > 0) {
        return {
            allowed: false,
            findings,
            nextPointers: null
        };
    }
    return {
        allowed: true,
        findings: [],
        nextPointers: {
            contract: CONTRACT_IDS.activationPointers,
            active: buildActivationPointerRecord("active", candidatePack, updatedAt),
            candidate: null,
            previous: activePack === null ? null : buildActivationPointerRecord("previous", activePack, updatedAt)
        }
    };
}
function previewRollbackPointers(current, updatedAt) {
    const findings = [];
    if (current.active === null) {
        findings.push("active pointer is required for rollback");
    }
    if (current.previous === null) {
        findings.push("previous pointer is required for rollback");
    }
    if (current.candidate !== null) {
        findings.push("rollback requires an empty candidate pointer");
    }
    let previousPack = null;
    if (current.previous !== null) {
        try {
            previousPack = ensurePackRecordMatchesManifest(current.previous, { requireActivationReady: true });
        }
        catch (error) {
            findings.push(error instanceof Error ? error.message : String(error));
        }
    }
    let activePack = null;
    if (current.active !== null) {
        try {
            activePack = ensurePackRecordMatchesManifest(current.active, { requireActivationReady: true });
        }
        catch (error) {
            findings.push(error instanceof Error ? error.message : String(error));
        }
    }
    if (findings.length > 0 || previousPack === null) {
        return {
            allowed: false,
            findings,
            nextPointers: null
        };
    }
    if (activePack !== null) {
        findings.push(...rollbackCoherenceFindings(buildCompileTargetFromPack(activePack), buildCompileTargetFromPack(previousPack)));
    }
    if (findings.length > 0) {
        return {
            allowed: false,
            findings,
            nextPointers: null
        };
    }
    return {
        allowed: true,
        findings: [],
        nextPointers: {
            contract: CONTRACT_IDS.activationPointers,
            active: buildActivationPointerRecord("active", previousPack, updatedAt),
            candidate: activePack === null ? null : buildActivationPointerRecord("candidate", activePack, updatedAt),
            previous: null
        }
    };
}
function assertPackIdAvailable(current, slot, packId) {
    const duplicates = ["active", "candidate", "previous"].filter((key) => {
        if (key === slot) {
            return false;
        }
        return current[key]?.packId === packId;
    });
    const duplicateSlot = duplicates[0];
    if (duplicateSlot !== undefined) {
        throw new Error(duplicatePackIdError(duplicateSlot, current[duplicateSlot]));
    }
}
export function describePackCompileTarget(packOrRootDir) {
    const pack = typeof packOrRootDir === "string" ? loadPack(packOrRootDir) : packOrRootDir;
    return buildCompileTargetFromPack(pack);
}
export function loadPackFromActivation(rootDir, slot = "active", options = {}) {
    const record = loadActivationPointers(rootDir).pointers[slot];
    if (record === null) {
        return null;
    }
    return ensurePackRecordMatchesManifest(record, {
        requireActivationReady: options.requireActivationReady === true
    });
}
export function describeActivationTarget(rootDir, slot = "active", options = {}) {
    const pack = loadPackFromActivation(rootDir, slot, options);
    return pack === null ? null : buildCompileTargetFromPack(pack);
}
export function loadActivationPointers(rootDir) {
    const resolvedRootDir = path.resolve(rootDir);
    const pointerPath = path.join(resolvedRootDir, ACTIVATION_LAYOUT.pointers);
    if (!existsSync(pointerPath)) {
        return {
            rootDir: resolvedRootDir,
            pointerPath,
            pointers: emptyActivationPointers()
        };
    }
    const pointers = readJsonFile(pointerPath);
    const errors = validateActivationPointers(pointers);
    if (errors.length > 0) {
        throw new Error(`Invalid activation pointers: ${errors.join("; ")}`);
    }
    return {
        rootDir: resolvedRootDir,
        pointerPath,
        pointers
    };
}
export function inspectActivationState(rootDir, updatedAt = "2026-03-06T00:00:00.000Z") {
    const state = loadActivationPointers(rootDir);
    return {
        ...state,
        active: inspectPointerRecord("active", state.pointers.active),
        candidate: inspectPointerRecord("candidate", state.pointers.candidate),
        previous: inspectPointerRecord("previous", state.pointers.previous),
        promotion: previewPromotionPointers(state.pointers, updatedAt),
        rollback: previewRollbackPointers(state.pointers, updatedAt)
    };
}
export function describeActivationObservability(rootDir, slot = "active", options = {}) {
    const inspection = inspectActivationState(rootDir, options.updatedAt);
    const selectedInspection = slot === "active" ? inspection.active : slot === "candidate" ? inspection.candidate : inspection.previous;
    const target = selectedInspection === null ? null : buildCompileTargetFromInspection(selectedInspection);
    let pack = null;
    let activePack = null;
    let candidatePack = null;
    try {
        pack = loadPackFromActivation(rootDir, slot, {
            requireActivationReady: options.requireActivationReady === true
        });
    }
    catch {
        pack = null;
    }
    try {
        activePack = loadPackFromActivation(rootDir, "active", {
            requireActivationReady: options.requireActivationReady === true
        });
    }
    catch {
        activePack = null;
    }
    try {
        candidatePack = loadPackFromActivation(rootDir, "candidate", {
            requireActivationReady: options.requireActivationReady === true
        });
    }
    catch {
        candidatePack = null;
    }
    const activeTarget = inspection.active === null ? null : buildCompileTargetFromInspection(inspection.active);
    const candidateTarget = inspection.candidate === null ? null : buildCompileTargetFromInspection(inspection.candidate);
    const candidateAheadBy = activeTarget !== null && candidateTarget !== null ? promotionFreshnessDelta(activeTarget, candidateTarget) : null;
    return {
        slot,
        target,
        servedArtifact: pack === null || target === null ? null : buildServedArtifactProof(target, pack.manifest.routeArtifact),
        learnedRouteFn: describeLearnedRouteFnFreshness(pack),
        routeArtifactDiff: describeRouteArtifactDiff(activePack, candidatePack),
        graphDynamics: describeGraphDynamicsFreshness(pack),
        graphEvolutionLog: pack === null ? null : describeGraphEvolutionLog(pack),
        initHandoff: pack === null ? emptyInitHandoff() : describePackInitHandoff(pack),
        promotionFreshness: {
            activePackId: inspection.active?.packId ?? null,
            candidatePackId: inspection.candidate?.packId ?? null,
            previousPackId: inspection.previous?.packId ?? null,
            activeUpdatedAt: inspection.pointers.active?.updatedAt ?? null,
            candidateUpdatedAt: inspection.pointers.candidate?.updatedAt ?? null,
            previousUpdatedAt: inspection.pointers.previous?.updatedAt ?? null,
            promotionAllowed: inspection.promotion.allowed,
            promotionFindings: [...inspection.promotion.findings],
            rollbackAllowed: inspection.rollback.allowed,
            rollbackFindings: [...inspection.rollback.findings],
            activeBehindPromotionReadyCandidate: activeTarget !== null &&
                candidateTarget !== null &&
                inspection.promotion.allowed &&
                isStrictlyFresherTarget(candidateTarget, activeTarget),
            candidateAheadBy
        }
    };
}
export function activatePack(rootDir, packRootDir, updatedAtOrOptions = "2026-03-06T00:00:00.000Z") {
    const mutation = normalizeActivationMutation(updatedAtOrOptions, "2026-03-06T00:00:00.000Z", "activate_pack");
    const current = loadActivationPointers(rootDir).pointers;
    const pack = loadPack(path.resolve(packRootDir));
    const activationErrors = validatePackActivationReadiness(pack);
    if (activationErrors.length > 0) {
        throw new Error(`Pack is not activation-ready: ${activationErrors.join("; ")}`);
    }
    assertPointerPinnedToPack("active", current.active, pack);
    if (current.active?.packId !== pack.manifest.packId) {
        assertPackIdAvailable(current, "active", pack.manifest.packId);
    }
    let previous = null;
    if (current.active !== null && current.active.packId !== pack.manifest.packId) {
        const activePack = ensurePackRecordMatchesManifest(current.active, { requireActivationReady: true });
        previous = buildActivationPointerRecord("previous", activePack, mutation.updatedAt);
    }
    const nextPointers = {
        contract: CONTRACT_IDS.activationPointers,
        active: buildActivationPointerRecord("active", pack, mutation.updatedAt),
        candidate: null,
        previous
    };
    const result = writeActivationPointers(rootDir, nextPointers);
    safeAppendPromotionActivationLog({
        rootDir,
        operation: "activate_pack",
        occurredAt: mutation.updatedAt,
        reason: mutation.reason,
        before: current,
        after: nextPointers
    });
    return result;
}
export function stageCandidatePack(rootDir, packRootDir, updatedAtOrOptions = "2026-03-06T00:00:00.000Z") {
    const mutation = normalizeActivationMutation(updatedAtOrOptions, "2026-03-06T00:00:00.000Z", "stage_candidate_pack");
    const current = loadActivationPointers(rootDir).pointers;
    const pack = loadPack(path.resolve(packRootDir));
    assertPointerPinnedToPack("candidate", current.candidate, pack);
    assertRetainedPointerMatchesManifest("active", current.active, { requireActivationReady: true });
    assertRetainedPointerMatchesManifest("previous", current.previous, { requireActivationReady: true });
    assertPackIdAvailable(current, "candidate", pack.manifest.packId);
    const nextPointers = {
        contract: CONTRACT_IDS.activationPointers,
        active: current.active,
        candidate: buildActivationPointerRecord("candidate", pack, mutation.updatedAt),
        previous: current.previous
    };
    const result = writeActivationPointers(rootDir, nextPointers);
    safeAppendPromotionActivationLog({
        rootDir,
        operation: "stage_candidate_pack",
        occurredAt: mutation.updatedAt,
        reason: mutation.reason,
        before: current,
        after: nextPointers
    });
    return result;
}
export function promoteCandidatePack(rootDir, updatedAtOrOptions = "2026-03-06T00:00:00.000Z") {
    const mutation = normalizeActivationMutation(updatedAtOrOptions, "2026-03-06T00:00:00.000Z", "promote_candidate_pack");
    const current = loadActivationPointers(rootDir).pointers;
    const preview = previewPromotionPointers(current, mutation.updatedAt);
    if (!preview.allowed || preview.nextPointers === null) {
        throw new Error(`Promotion blocked: ${preview.findings.join("; ")}`);
    }
    const result = writeActivationPointers(rootDir, preview.nextPointers);
    safeAppendPromotionActivationLog({
        rootDir,
        operation: "promote_candidate_pack",
        occurredAt: mutation.updatedAt,
        reason: mutation.reason,
        before: current,
        after: preview.nextPointers
    });
    return result;
}
export function rollbackActivePack(rootDir, updatedAtOrOptions = "2026-03-06T00:00:00.000Z") {
    const mutation = normalizeActivationMutation(updatedAtOrOptions, "2026-03-06T00:00:00.000Z", "rollback_active_pack");
    const current = loadActivationPointers(rootDir).pointers;
    const preview = previewRollbackPointers(current, mutation.updatedAt);
    if (!preview.allowed || preview.nextPointers === null) {
        throw new Error(`Rollback blocked: ${preview.findings.join("; ")}`);
    }
    const result = writeActivationPointers(rootDir, preview.nextPointers);
    safeAppendPromotionActivationLog({
        rootDir,
        operation: "rollback_active_pack",
        occurredAt: mutation.updatedAt,
        reason: mutation.reason,
        before: current,
        after: preview.nextPointers,
        rollbackTarget: preview.nextPointers.active?.packId ?? null
    });
    return result;
}
export { LEARNING_SPINE_LOG_LAYOUT, appendLearningSpineLogEntry, buildLearningSpineLogId, readLearningSpineLogEntries, resolveLearningSpineLogPath } from "./learning-spine-logs.js";
export function loadPack(rootDir) {
    const manifestPath = path.join(rootDir, PACK_LAYOUT.manifest);
    if (!existsSync(manifestPath)) {
        throw new Error(`pack manifest not found: ${manifestPath}`);
    }
    const manifest = readJsonFile(manifestPath);
    const manifestErrors = validatePackDescriptor(manifest);
    if (manifestErrors.length > 0) {
        throw new Error(`Invalid pack descriptor: ${manifestErrors.join("; ")}`);
    }
    const graphPath = resolvePackAssetPath(rootDir, manifest.runtimeAssets.graphPath, "graph payload");
    const vectorPath = resolvePackAssetPath(rootDir, manifest.runtimeAssets.vectorPath, "vector payload");
    const routerPath = manifest.runtimeAssets.router.artifactPath === null
        ? null
        : resolvePackAssetPath(rootDir, manifest.runtimeAssets.router.artifactPath, "router payload");
    const fileErrors = [];
    pushFileError(fileErrors, graphPath, "graph payload");
    pushFileError(fileErrors, vectorPath, "vector payload");
    if (routerPath !== null && manifest.runtimeAssets.router.kind !== "none") {
        pushFileError(fileErrors, routerPath, "router payload");
    }
    if (fileErrors.length > 0) {
        throw new Error(`Invalid pack descriptor: ${fileErrors.join("; ")}`);
    }
    const graph = readJsonFile(graphPath);
    const vectors = readJsonFile(vectorPath);
    const router = routerPath === null ? null : readJsonFile(routerPath);
    const payloadErrors = [
        ...validatePackGraphPayload(graph, manifest.packId),
        ...validatePackVectorsPayload(vectors, graph),
        ...(router === null ? [] : validateRouterArtifact(router, manifest))
    ];
    if (manifest.payloadChecksums.graph !== sha256File(graphPath)) {
        payloadErrors.push("graph checksum does not match manifest");
    }
    if (manifest.payloadChecksums.vector !== sha256File(vectorPath)) {
        payloadErrors.push("vector checksum does not match manifest");
    }
    if (routerPath === null) {
        if (manifest.payloadChecksums.router !== null) {
            payloadErrors.push("router checksum must be null when router artifact is absent");
        }
    }
    else {
        const routerChecksum = sha256File(routerPath);
        if (manifest.payloadChecksums.router !== routerChecksum) {
            payloadErrors.push("router checksum does not match manifest");
        }
    }
    if (payloadErrors.length > 0) {
        throw new Error(`Invalid pack descriptor: ${payloadErrors.join("; ")}`);
    }
    return {
        rootDir,
        manifestPath,
        graphPath,
        vectorPath,
        routerPath,
        manifest,
        graph,
        vectors,
        router
    };
}
//# sourceMappingURL=index.js.map