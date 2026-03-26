#!/usr/bin/env node
import { execFileSync, execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, readSync, openSync, closeSync, realpathSync, rmSync, statSync, writeFileSync, symlinkSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
import { DEFAULT_OLLAMA_EMBEDDING_MODEL, createOllamaEmbedder } from "@openclawbrain/compiler";
import { ensureManagedLearnerServiceForActivationRoot, inspectManagedLearnerService, removeManagedLearnerServiceForActivationRoot, parseDaemonArgs, runDaemonCommand } from "./daemon.js";
import { exportBrain, importBrain } from "./import-export.js";
import { buildNormalizedEventExport } from "@openclawbrain/contracts";
import { buildTeacherSupervisionArtifactsFromNormalizedEventExport, createAlwaysOnLearningRuntimeState, describeAlwaysOnLearningRuntimeState, drainAlwaysOnLearningRuntime, loadOrInitBaseline, reindexCandidatePackBuildResultWithEmbedder, materializeAlwaysOnLearningCandidatePack, persistBaseline } from "@openclawbrain/learner";
import { inspectActivationState, loadPackFromActivation, promoteCandidatePack, resolveLearningSpineLogPath, stageCandidatePack } from "@openclawbrain/pack-format";
import { resolveActivationRoot } from "./resolve-activation-root.js";
import { describeOpenClawHomeInspection, discoverOpenClawHomes, formatOpenClawHomeLayout, formatOpenClawHomeProfileSource, inspectOpenClawHome } from "./openclaw-home-layout.js";
import { inspectOpenClawBrainHookStatus, inspectOpenClawBrainPluginAllowlist } from "./openclaw-hook-truth.js";
import { describeOpenClawBrainInstallIdentity, describeOpenClawBrainInstallLayout, findInstalledOpenClawBrainPlugin, getOpenClawBrainKnownPluginIds, normalizeOpenClawBrainPluginsConfig, pinInstalledOpenClawBrainPluginActivationRoot, resolveOpenClawBrainInstallTarget } from "./openclaw-plugin-install.js";
import { loadAttachmentPolicyDeclaration, resolveEffectiveAttachmentPolicyTruth, writeAttachmentPolicyDeclaration } from "./attachment-policy-truth.js";
import { DEFAULT_WATCH_POLL_INTERVAL_SECONDS, buildNormalizedEventExportFromScannedEvents, bootstrapRuntimeAttach, buildOperatorSurfaceReport, clearOpenClawProfileRuntimeLoadProof, compileRuntimeContext, createAsyncTeacherLiveLoop, createOpenClawLocalSessionTail, createRuntimeEventExportScanner, describeCurrentProfileBrainStatus, formatOperatorRollbackReport, listOpenClawProfileRuntimeLoadProofs, loadRuntimeEventExportBundle, loadWatchTeacherSnapshotState, persistWatchTeacherSnapshot, rollbackRuntimeAttach, resolveAttachmentRuntimeLoadProofsPath, resolveOperatorTeacherSnapshotPath, resolveAsyncTeacherLiveLoopSnapshotPath, resolveWatchSessionTailCursorPath, resolveWatchStateRoot, resolveWatchTeacherSnapshotPath, scanLiveEventExport, scanRecordedSession, summarizeLearningPathFromMaterialization, summarizeNormalizedEventExportLabelFlow, writeScannedEventExportBundle } from "./index.js";
import { appendLearningUpdateLogs } from "./learning-spine.js";
import { readBoundedJsonlTail } from "./bounded-jsonl-reader.js";
import { buildPassiveLearningSessionExportFromOpenClawSessionStore } from "./local-session-passive-learning.js";
import { summarizePackVectorEmbeddingState } from "./embedding-status.js";
import { buildTracedLearningStatusSurface, loadBrainStoreTracedLearningBridge, mergeTracedLearningBridgePayload, persistBrainStoreTracedLearningBridge, writeTracedLearningBridge } from "./traced-learning-bridge.js";
import { discoverOpenClawSessionStores, loadOpenClawSessionIndex, readOpenClawSessionFile } from "./session-store.js";
import { readOpenClawBrainProviderDefaults, readOpenClawBrainProviderConfig, readOpenClawBrainProviderConfigFromSources, resolveOpenClawBrainProviderDefaultsPath } from "./provider-config.js";
import { formatOperatorLearningPathSummary } from "./status-learning-path.js";
const OPENCLAWBRAIN_EMBEDDER_BASE_URL_ENV = "OPENCLAWBRAIN_EMBEDDER_BASE_URL";
const OPENCLAWBRAIN_EMBEDDER_PROVIDER_ENV = "OPENCLAWBRAIN_EMBEDDER_PROVIDER";
const OPENCLAWBRAIN_EMBEDDER_MODEL_ENV = "OPENCLAWBRAIN_EMBEDDER_MODEL";
const OPENCLAWBRAIN_INSTALL_SKIP_EMBEDDER_PROVISION_ENV = "OPENCLAWBRAIN_INSTALL_SKIP_EMBEDDER_PROVISION";
const INSTALL_COMPATIBLE_LOCAL_TEACHER_MODEL_PREFIXES = [
    "qwen3.5:9b",
    "qwen3.5:8b",
    "qwen3:8b",
    "qwen2.5:7b"
];
function quoteShellArg(value) {
    return `'${value.replace(/'/g, `"'"'`)}'`;
}
function normalizeOptionalCliString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}
function canonicalizeExistingCliPath(filePath) {
    const resolvedPath = path.resolve(filePath);
    try {
        return realpathSync(resolvedPath);
    }
    catch {
        return resolvedPath;
    }
}
function readTruthyEnvFlag(name, env = process.env) {
    const value = normalizeOptionalCliString(env[name]);
    if (value === null) {
        return false;
    }
    return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}
function getCliHomeDir() {
    return process.env.HOME ?? process.env.USERPROFILE ?? "~";
}
function discoverInstallCandidateOpenClawHomes(homeDir = getCliHomeDir()) {
    return discoverOpenClawHomes(homeDir).map((inspection) => inspection.openclawHome);
}
function summarizeSharedActivationRootReferenceProof(reference) {
    switch (reference.installState) {
        case "installed":
            return reference.serveAttachmentState === "attached"
                ? "hook installed and loadable"
                : "hook files exist, but the current install is not loadable yet";
        case "blocked_by_allowlist":
            return "hook files exist but OpenClaw will not load them because plugins.allow blocks openclawbrain";
        case "not_installed":
        default:
            return "hook files are incomplete, so serve-path attachment is not proven";
    }
}
function partitionSharedActivationRootHookReferences(references) {
    return references.reduce((result, reference) => {
        if (reference.serveAttachmentState === "attached") {
            result.attached.push(reference);
        }
        else {
            result.halfAttached.push(reference);
        }
        return result;
    }, {
        attached: [],
        halfAttached: []
    });
}
function formatSharedActivationRootReferenceList(references, options = {}) {
    return references
        .map((reference) => {
        const parts = [path.resolve(reference.openclawHome)];
        if (options.includeInspection) {
            parts.push(describeOpenClawHomeInspection(reference.inspection));
        }
        if (options.includeProof) {
            parts.push(summarizeSharedActivationRootReferenceProof(reference));
        }
        return `  - ${parts.join(" | ")}`;
    })
        .join("\n");
}
function findInstalledHookReferencesForActivationRoot(input) {
    const resolvedActivationRoot = path.resolve(input.activationRoot);
    const resolvedExcludedHome = input.excludingOpenClawHome === undefined || input.excludingOpenClawHome === null
        ? null
        : path.resolve(input.excludingOpenClawHome);
    return discoverOpenClawHomes(input.homeDir ?? getCliHomeDir())
        .filter((inspection) => resolvedExcludedHome === null || path.resolve(inspection.openclawHome) !== resolvedExcludedHome)
        .flatMap((inspection) => {
        const installedActivationRoot = resolveActivationRoot({
            openclawHome: inspection.openclawHome,
            quiet: true
        });
        if (installedActivationRoot.trim().length === 0) {
            return [];
        }
        const installHook = summarizeStatusInstallHook(inspection.openclawHome);
        const reference = {
            openclawHome: inspection.openclawHome,
            inspection,
            installState: installHook.state === "installed" || installHook.state === "blocked_by_allowlist"
                ? installHook.state
                : "not_installed",
            serveAttachmentState: installHook.state === "installed" && installHook.loadability === "loadable"
                ? "attached"
                : "half_attached",
            hookDetail: installHook.detail
        };
        return path.resolve(installedActivationRoot) === resolvedActivationRoot
            ? [reference]
            : [];
    })
        .sort((left, right) => left.openclawHome.localeCompare(right.openclawHome));
}
function findOtherInstalledHookReferencesForActivationRoot(input) {
    return findInstalledHookReferencesForActivationRoot(input);
}
function resolveWatchProfileRootsForActivationRoot(activationRoot, homeDir = getCliHomeDir()) {
    const references = findInstalledHookReferencesForActivationRoot({
        activationRoot,
        homeDir
    });
    const partitioned = partitionSharedActivationRootHookReferences(references);
    const attachedProfileRoots = partitioned.attached.map((reference) => path.resolve(reference.openclawHome));
    return {
        attachedProfileRoots: attachedProfileRoots.length > 0 || references.length > 0
            ? attachedProfileRoots
            : undefined,
        halfAttachedReferences: partitioned.halfAttached
    };
}
function assertActivationRootPurgeIsNotShared(input) {
    const sharedReferences = findOtherInstalledHookReferencesForActivationRoot({
        activationRoot: input.activationRoot,
        excludingOpenClawHome: input.openclawHome
    });
    if (sharedReferences.length === 0) {
        return;
    }
    const partitioned = partitionSharedActivationRootHookReferences(sharedReferences);
    const attachedProfiles = formatSharedActivationRootReferenceList(partitioned.attached, {
        includeInspection: true,
        includeProof: true
    });
    const halfAttachedProfiles = formatSharedActivationRootReferenceList(partitioned.halfAttached, {
        includeInspection: true,
        includeProof: true
    });
    throw new Error(partitioned.attached.length > 0
        ? [
            `Refusing to purge activation root ${path.resolve(input.activationRoot)} because another installed OpenClaw profile still points at it.`,
            "Other attached profiles:",
            attachedProfiles,
            ...(partitioned.halfAttached.length === 0
                ? []
                : [
                    "Other half-attached profiles:",
                    halfAttachedProfiles
                ]),
            "Use uninstall --keep-data or detach on this profile first, then remove or repair the remaining profile hooks before purging shared brain data.",
            "For Eagle dogfood, prefer its own activation root so CormorantAI stays untouched."
        ].join("\n")
        : [
            `Refusing to purge activation root ${path.resolve(input.activationRoot)} because another OpenClaw profile still points at it, but its hook is not loadable yet.`,
            "Other half-attached profiles:",
            halfAttachedProfiles,
            "Repair or remove those half-attached profile hooks before purging shared brain data so status stays explicit instead of drifting into a missing-root surprise.",
            "For Eagle dogfood, prefer its own activation root so CormorantAI stays untouched."
        ].join("\n"));
}
function formatInstallOpenClawHomeSource(source) {
    switch (source) {
        case "explicit":
            return "--openclaw-home";
        case "env":
            return "OPENCLAW_HOME";
        case "discovered_single_home":
            return "single discovered install target";
        default:
            return source;
    }
}
function resolveInstallOpenClawHome(explicitOpenclawHome) {
    const normalizedExplicitHome = normalizeOptionalCliString(explicitOpenclawHome);
    if (normalizedExplicitHome !== null) {
        return {
            openclawHome: path.resolve(normalizedExplicitHome),
            openclawHomeSource: "explicit"
        };
    }
    const envOpenClawHome = normalizeOptionalCliString(process.env.OPENCLAW_HOME);
    if (envOpenClawHome !== null) {
        return {
            openclawHome: path.resolve(envOpenClawHome),
            openclawHomeSource: "env"
        };
    }
    const discoveredHomes = discoverInstallCandidateOpenClawHomes();
    if (discoveredHomes.length === 1) {
        return {
            openclawHome: path.resolve(discoveredHomes[0]),
            openclawHomeSource: "discovered_single_home"
        };
    }
    if (discoveredHomes.length > 1) {
        const installPrefix = detectConsumerSafeOperatorCliPrefix();
        const targetChoices = discoverOpenClawHomes()
            .map((inspection) => {
            const resolvedCandidate = path.resolve(inspection.openclawHome);
            return `  - ${resolvedCandidate} (${describeOpenClawHomeInspection(inspection)})\n    ${installPrefix} install --openclaw-home ${quoteShellArg(resolvedCandidate)}`;
        })
            .join("\n");
        throw new Error([
            "Refusing ambiguous OpenClaw install targets.",
            targetChoices,
            "Pass --openclaw-home <path> or set OPENCLAW_HOME to pin one OpenClaw home."
        ].join("\n"));
    }
    throw new Error("No OpenClaw home found. Pass --openclaw-home <path> or set OPENCLAW_HOME.");
}
function resolveInstallActivationRoot(openclawHome, explicitActivationRoot) {
    const normalizedExplicitActivationRoot = normalizeOptionalCliString(explicitActivationRoot);
    if (normalizedExplicitActivationRoot !== null) {
        return {
            activationRoot: path.resolve(normalizedExplicitActivationRoot),
            source: "explicit"
        };
    }
    return {
        activationRoot: path.resolve(path.dirname(openclawHome), ".openclawbrain", "activation"),
        source: "default_from_openclaw_home"
    };
}
function resolveInstallWorkspaceId(openclawHome, explicitWorkspaceId) {
    const normalizedExplicitWorkspaceId = normalizeOptionalCliString(explicitWorkspaceId);
    if (normalizedExplicitWorkspaceId !== null) {
        return {
            workspaceId: normalizedExplicitWorkspaceId,
            source: "explicit"
        };
    }
    const inspection = inspectOpenClawHome(openclawHome);
    if (inspection.profileId !== null) {
        return {
            workspaceId: inspection.profileId,
            source: inspection.profileSource === "directory_name"
                ? "openclaw_home_dir"
                : inspection.profileSource === "openclaw_json_profile"
                    ? "openclaw_json_profile"
                    : inspection.profileSource === "openclaw_json_single_profile_key"
                        ? "openclaw_json_single_profile_key"
                        : "fallback"
        };
    }
    if (inspection.layout === "shared_home_profiles_in_config" || inspection.layout === "single_openclaw_home") {
        return {
            workspaceId: "current_profile",
            source: "current_profile_boundary"
        };
    }
    const dirName = path.basename(openclawHome);
    if (dirName === ".openclaw") {
        return {
            workspaceId: "default",
            source: "openclaw_home_dir"
        };
    }
    const derivedWorkspaceId = dirName.startsWith(".openclaw-") ? dirName.slice(".openclaw-".length) : dirName;
    if (derivedWorkspaceId.trim().length > 0) {
        return {
            workspaceId: derivedWorkspaceId,
            source: "openclaw_home_dir"
        };
    }
    return {
        workspaceId: "workspace",
        source: "fallback"
    };
}
function resolveInstallEmbedderProvisionSkip(explicitSkip) {
    if (explicitSkip) {
        return {
            skipEmbedderProvision: true,
            skipEmbedderProvisionSource: "flag"
        };
    }
    if (readTruthyEnvFlag(OPENCLAWBRAIN_INSTALL_SKIP_EMBEDDER_PROVISION_ENV)) {
        return {
            skipEmbedderProvision: true,
            skipEmbedderProvisionSource: "env"
        };
    }
    return {
        skipEmbedderProvision: false,
        skipEmbedderProvisionSource: null
    };
}
function formatInstallActivationRootSource(source) {
    if (source === "explicit") {
        return "explicit --activation-root";
    }
    return "default beside --openclaw-home";
}
function formatInstallWorkspaceIdSource(source) {
    switch (source) {
        case "explicit":
            return "explicit --workspace-id";
        case "openclaw_json_profile":
            return "from openclaw.json profile";
        case "openclaw_json_single_profile_key":
            return "from the only openclaw.json profiles entry";
        case "current_profile_boundary":
            return "current_profile boundary for a shared OpenClaw home";
        case "openclaw_home_dir":
            return "from OpenClaw home dir";
        default:
            return "fallback default";
    }
}
function detectConsumerSafeOperatorCliPrefix() {
    const npmExecPath = (process.env.npm_execpath ?? "").toLowerCase();
    const userAgent = process.env.npm_config_user_agent ?? "";
    if (npmExecPath.includes("npm-cli.js")) {
        return "npm exec openclawbrain --";
    }
    if (npmExecPath.includes("pnpm")) {
        return "pnpm exec openclawbrain";
    }
    if (userAgent.startsWith("npm/")) {
        return "npm exec openclawbrain --";
    }
    if (userAgent.startsWith("pnpm/")) {
        return "pnpm exec openclawbrain";
    }
    return "npm exec openclawbrain --";
}
function buildStatusReplacementCommand(input, json) {
    if (typeof input.activationRoot !== "string" || input.activationRoot.trim().length === 0) {
        return null;
    }
    const parts = [
        detectConsumerSafeOperatorCliPrefix(),
        "status",
        "--activation-root",
        quoteShellArg(path.resolve(input.activationRoot))
    ];
    if (typeof input.eventExportPath === "string" && input.eventExportPath.trim().length > 0) {
        parts.push("--event-export", quoteShellArg(input.eventExportPath));
    }
    if (typeof input.teacherSnapshotPath === "string" && input.teacherSnapshotPath.trim().length > 0) {
        parts.push("--teacher-snapshot", quoteShellArg(input.teacherSnapshotPath));
    }
    if (typeof input.updatedAt === "string" && input.updatedAt.trim().length > 0) {
        parts.push("--updated-at", quoteShellArg(input.updatedAt));
    }
    if (input.brainAttachmentPolicy !== null && input.brainAttachmentPolicy !== undefined) {
        parts.push("--brain-attachment-policy", input.brainAttachmentPolicy);
    }
    if (json) {
        parts.push("--json");
    }
    return parts.join(" ");
}
function buildDoctorDeletedMessage(args) {
    let activationRoot = null;
    let eventExportPath = null;
    let teacherSnapshotPath = null;
    let updatedAt = null;
    let brainAttachmentPolicy = null;
    let json = false;
    for (let index = 0; index < args.length; index += 1) {
        const arg = args[index];
        if (arg === "--json") {
            json = true;
            continue;
        }
        const next = args[index + 1];
        if (next === undefined) {
            continue;
        }
        if (arg === "--activation-root") {
            activationRoot = next;
            index += 1;
            continue;
        }
        if (arg === "--event-export") {
            eventExportPath = next;
            index += 1;
            continue;
        }
        if (arg === "--teacher-snapshot") {
            teacherSnapshotPath = next;
            index += 1;
            continue;
        }
        if (arg === "--updated-at") {
            updatedAt = next;
            index += 1;
            continue;
        }
        if (arg === "--brain-attachment-policy") {
            if (next === "undeclared" || next === "dedicated" || next === "shared") {
                brainAttachmentPolicy = next;
            }
            index += 1;
        }
    }
    const replacementInput = {
        activationRoot,
        eventExportPath,
        teacherSnapshotPath,
        updatedAt,
        brainAttachmentPolicy
    };
    const humanCommand = buildStatusReplacementCommand(replacementInput, false);
    const jsonCommand = buildStatusReplacementCommand(replacementInput, true);
    const lines = [
        "`doctor` is no longer a separate operator surface.",
        'Use `openclawbrain status --activation-root <path>` as the human answer to "How\'s the brain?" and `status --json` for the canonical current-profile object.',
        "Use `describeAttachStatus()` or the proof helpers only when you need deeper activation diagnostics."
    ];
    if (json && jsonCommand !== null) {
        lines.push(`Replacement: ${jsonCommand}`);
    }
    else if (humanCommand !== null) {
        lines.push(`Replacement: ${humanCommand}`);
        if (jsonCommand !== null) {
            lines.push(`Canonical JSON: ${jsonCommand}`);
        }
    }
    return lines.join(" ");
}
function buildSetupDeletedMessage() {
    return [
        "`setup` has been removed.",
        "Use `openclawbrain install` instead.",
        "The install command still accepts the explicit targeting flags that setup used: `--openclaw-home`, `--activation-root`, `--workspace-id`, and `--shared`."
    ].join(" ");
}
function operatorCliHelp() {
    return [
        "Usage:",
        "  openclawbrain install [--openclaw-home <path>] [options]",
        "  openclawbrain attach --openclaw-home <path> [options]",
        "  openclawbrain <status|rollback> [--activation-root <path>|--openclaw-home <path>] [options]",
        "  openclawbrain watch --activation-root <path> [--scan-root <path>] [--interval <seconds>]",
        "  openclawbrain daemon <start|stop|status|logs> --activation-root <path> [--json]",
        "  openclawbrain detach --openclaw-home <path> [options]",
        "  openclawbrain uninstall --openclaw-home <path> [--keep-data|--purge-data] [options]",
        "  openclawbrain context \"message\" [--activation-root <path>|--openclaw-home <path>]",
        "  openclawbrain history [--activation-root <path>|--openclaw-home <path>] [--limit N] [--json]",
        "  openclawbrain scan --session <trace.json> --root <path> [options]",
        "  openclawbrain scan --live <event-export-path> --workspace <workspace.json> [options]",
        "  openclawbrain learn [--activation-root <path>|--openclaw-home <path>] [--json]",
        "  openclawbrain-ops <status|rollback> [--activation-root <path>|--openclaw-home <path>] [options]  # compatibility alias",
        "  openclawbrain-ops scan --session <trace.json> --root <path> [options]    # compatibility alias",
        "",
        "Options:",
        "  --openclaw-home <path>      OpenClaw home dir for install/attach/detach/uninstall (e.g. ~/.openclaw-Tern or ~/.openclaw). Also pins status/rollback/context/history/learn to that installed target when applicable.",
        "  --shared                    Set brain-attachment-policy to shared instead of dedicated (install/attach only).",
        `  --skip-embedder-provision  Skip the default Ollama ${DEFAULT_OLLAMA_EMBEDDING_MODEL} pull before install/attach bootstrap. Use only when intentionally deferring embedder setup. Also supports ${OPENCLAWBRAIN_INSTALL_SKIP_EMBEDDER_PROVISION_ENV}=1.`,
        "  --activation-root <path>    Explicit activation root for attach/watch/daemon and other stateful commands; install/attach default to sibling .openclawbrain/activation next to the selected OpenClaw home.",
        "  --keep-data                 Preserve activation data on uninstall; detach always behaves this way.",
        "  --purge-data                Remove activation data on uninstall; requires the installed profile hook or --activation-root.",
        "  --restart <never|safe|external>  Restart guidance mode for detach/uninstall. 'safe' is conservative; 'never' leaves restart entirely to the operator.",
        "  --workspace-id <id>         Workspace identifier for install/attach provenance; defaults to the detected profile target from openclaw.json when possible, otherwise the profile name or current_profile boundary.",
        "  --event-export <path>       Event-export bundle root or normalized export JSON payload.",
        "  --teacher-snapshot <path>   Canonical watch teacher snapshot JSON or raw async teacher snapshot JSON; keeps live-first, principal-priority, and passive-backfill learner truth explicit.",
        "  --updated-at <iso>          Observation time to use for freshness checks.",
        "  --brain-attachment-policy <undeclared|dedicated|shared>  Override attachment policy semantics for status inspection.",
        "  --detailed                   Show verbose diagnostic output for status (default is human-friendly summary).",
        "  --dry-run                   Preview rollback pointer movement without writing activation state.",
        "  --session <path>            Sanitized recorded-session trace JSON to replay.",
        "  --live <path>               Runtime event-export bundle root or normalized export JSON to scan once.",
        "  --root <path>               Output root for scan --session replay artifacts.",
        "  --workspace <path>          Workspace metadata JSON for scan --live candidate materialization.",
        "  --pack-label <label>        Candidate-pack label for scan --live. Defaults to scanner-live-cli.",
        "  --observed-at <iso>         Observation time for scan --live freshness checks.",
        "  --snapshot-out <path>       Write the one-shot scan --live snapshot JSON.",
        "  --limit <N>                 Maximum number of history entries to show (default: 20, history only).",
        "  --scan-root <path>          Event-export scan root for watch mode (defaults to <activation-root>/event-exports).",
        "  --interval <seconds>        Polling interval for watch mode (default: 30).",
        "  --json                      Emit machine-readable JSON instead of text.",
        "  --help                      Show this help.",
        "",
        "Lifecycle flow:",
        "  1. install            openclawbrain install — safe first-time default; writes the generated shadow hook or pins an already-installed native package plugin for one OpenClaw home",
        "  2. attach             openclawbrain attach --openclaw-home <path> [--activation-root <path>] — explicit reattach/manual hook path for known brain data; use install first",
        "  3. status             openclawbrain status --activation-root <path> — answer \"How's the brain?\" for that boundary",
        "  4. status --detailed  openclawbrain status --activation-root <path> --detailed — explain serve path, freshness, backlog, and failure mode",
        "  5. watch              openclawbrain watch --activation-root <path> — run the foreground learning/watch loop",
        "  6. daemon start       openclawbrain daemon start --activation-root <path> — keep watch running in the background on macOS",
        "  7. daemon status      openclawbrain daemon status --activation-root <path> — inspect the background watch state",
        "  8. detach             openclawbrain detach --openclaw-home <path> — remove the profile hookup only and keep brain data",
        "  9. uninstall          openclawbrain uninstall --openclaw-home <path> --keep-data|--purge-data — remove the hookup and choose the data outcome explicitly",
        "",
        "Advanced/operator surfaces:",
        "  context      preview the brain context that would be injected for a message",
        "  rollback     preview or apply active <- previous, active -> candidate pointer movement",
        "  scan         inspect one recorded session or live event export without claiming a daemon is running",
        "  learn        one-shot local-session learning pass against the resolved activation root",
        "  status --teacher-snapshot keeps the current live-first / principal-priority / passive-backfill learner order visible when that snapshot exists",
        "  native package installs still need the openclawbrain CLI available because install/attach pin the activation root for that package copy",
        "  watch/daemon persist their operator snapshot at <activation-root>/watch/teacher-snapshot.json; --teacher-snapshot overrides the default path",
        "  watch teacher defaults come from install-written provider-defaults.json under the activation root; OPENCLAWBRAIN_TEACHER_* and OPENCLAWBRAIN_EMBEDDER_* are host-shell overrides only, not live gateway wiring",
        "",
        "Exit codes:",
        "  install: 0 on successful profile hookup/bootstrap, 1 on input/read failure.",
        "  status: 0 on successful inspection, 1 on input/read failure.",
        "  rollback: 0 when ready/applied, 1 when blocked or on input/read failure.",
        "  attach: 0 on successful profile hookup/bootstrap, 1 on input/read failure.",
        "  detach: 0 on successful unhook, 1 on input/read failure.",
        "  uninstall: 0 on successful unhook/cleanup, 1 on input/read failure.",
        "  scan: 0 on successful replay/scan, 1 on input/read failure."
    ].join("\n");
}
function yesNo(value) {
    if (value === null) {
        return "unknown";
    }
    return value ? "yes" : "no";
}
function formatPrincipalLatest(report) {
    const latest = report.principal.latestFeedback;
    return latest === null ? "none" : `${latest.teacherIdentity}/${latest.kind}`;
}
function formatPrincipalCheckpointFrontier(report) {
    const checkpoint = report.learning.leadingPrincipalCheckpoint;
    if (checkpoint === null) {
        return "none";
    }
    const learnedThrough = checkpoint.learnedThroughSequence ?? "none";
    const newestPending = checkpoint.newestPendingSequence ?? "none";
    return `${checkpoint.teacherIdentity}:${learnedThrough}->${newestPending}`;
}
function formatStructuralOps(report) {
    const structuralOps = report.graph.structuralOps;
    return structuralOps === null
        ? "none"
        : `split:${structuralOps.split},merge:${structuralOps.merge},prune:${structuralOps.prune},connect:${structuralOps.connect}`;
}
function formatGraphConnectDiagnostics(diagnostics) {
    if (diagnostics === null) {
        return "none";
    }
    return `budget:${diagnostics.requestedBudget},threshold:${diagnostics.scoreThreshold},pairs:${diagnostics.appliedPairCount}/${diagnostics.candidatePairCount},edges:${diagnostics.createdEdgeCount}`;
}
function formatCompactGraphConnectDiagnostics(diagnostics) {
    if (diagnostics === null) {
        return "none";
    }
    return `pairs:${diagnostics.appliedPairCount},edges:${diagnostics.createdEdgeCount}`;
}
function formatGraphSummary(report) {
    return (report.graph.latestMaterialization.operatorSummary ??
        report.graph.operatorSummary ??
        report.graph.latestMaterialization.detail ??
        report.graph.detail);
}
function formatScannerSurfaces(report) {
    return report.supervision.scanSurfaces.length === 0 ? "none" : report.supervision.scanSurfaces.join("|");
}
function formatLearningBuckets(report) {
    const buckets = report.learning.pendingByBucket;
    if (buckets === null) {
        return "none";
    }
    return `pi:${buckets.principal_immediate},pb:${buckets.principal_backfill},live:${buckets.live},backfill:${buckets.backfill}`;
}
function formatLearningWarnings(report) {
    const warnings = report.learning.warningStates.filter((warning) => warning !== "teacher_snapshot_unavailable");
    return warnings.length === 0 ? "none" : warnings.join("|");
}
function formatLabelFlowSummary(labelFlow) {
    return `source=${labelFlow.source} human=${labelFlow.humanLabelCount ?? "none"} self=${labelFlow.selfLabelCount ?? "none"} implicitPositive=${labelFlow.implicitPositiveCount ?? "none"} teacherArtifacts=${labelFlow.asyncTeacherArtifactCount ?? "none"}`;
}
function formatLearningPathSummary(learningPath) {
    return `source=${learningPath.source} pg=${learningPath.policyGradientVersion} method=${learningPath.policyGradientMethod ?? "none"} target=${learningPath.targetConstruction ?? "none"} connect=${learningPath.connectOpsFired ?? "none"} trajectories=${learningPath.reconstructedTrajectoryCount ?? "none"}`;
}
function formatTeacherLoopSummary(report) {
    const parts = [
        `snapshot=${report.teacherLoop.sourcePath ?? "none"}`,
        `kind=${report.teacherLoop.sourceKind}`,
        `lastRun=${report.teacherLoop.lastRunAt ?? "none"}`,
        `artifacts=${report.teacherLoop.artifactCount ?? "none"}`,
        `freshness=${report.teacherLoop.latestFreshness}`,
        `queue=${report.teacherLoop.queueDepth ?? "none"}/${report.teacherLoop.queueCapacity ?? "none"}`,
        `running=${yesNo(report.teacherLoop.running)}`
    ];
    if (report.teacherLoop.lastNoOpReason !== "none") {
        parts.push(`noOp=${report.teacherLoop.lastNoOpReason}`);
    }
    if (report.teacherLoop.failureMode !== "none") {
        const failureDetail = report.teacherLoop.failureDetail === null
            ? report.teacherLoop.failureMode
            : `${report.teacherLoop.failureMode}(${report.teacherLoop.failureDetail})`;
        parts.push(`failure=${failureDetail}`);
    }
    return parts.join(" ");
}
function formatCompactValue(value, maxLength = 64) {
    return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}...`;
}
function formatCompactList(values, maxItems = 2, maxLength = 64) {
    if (values.length === 0) {
        return "none";
    }
    const visible = values.slice(0, maxItems).map((value) => formatCompactValue(value, maxLength));
    return values.length > maxItems ? `${visible.join("|")}+${values.length - maxItems}more` : visible.join("|");
}
const SERVICE_RISK_FINDING_CODES = new Set([
    "hook_desynced",
    "current_profile_not_attached",
    "activation_broken_install",
    "activation_stale_incomplete",
    "active_missing",
    "active_unhealthy",
    "learned_route_missing",
    "serve_path_fail_open",
    "serve_path_hard_fail",
    "serve_path_route_evidence_missing"
]);
const DEGRADED_BRAIN_FINDING_CODES = new Set([
    "attachment_scope_partial",
    "bootstrap_waiting_for_first_export",
    "serve_path_unprobed",
    "brain_context_kernel_only",
    "candidate_unhealthy",
    "promotion_blocked",
    "supervision_not_flowing",
    "scan_surfaces_missing"
]);
const COSMETIC_FINDING_CODES = new Set([
    "last_promotion_unknown",
    "rollback_blocked",
    "supervision_unavailable",
    "turn_attribution_partial",
    "teacher_snapshot_unavailable"
]);
const LEARNING_WARNING_MESSAGES = {
    awaiting_first_export: "awaiting first export",
    principal_live_backlog: "principal live backlog is ahead of serving",
    principal_backfill_pending: "principal backfill is still queued",
    active_pack_behind_latest_principal: "active pack is behind the latest principal correction",
    passive_backfill_pending: "passive backfill remains queued",
    teacher_queue_full: "teacher queue is full",
    teacher_labels_stale: "teacher labels are stale",
    teacher_no_artifacts: "teacher produced no artifacts",
    teacher_snapshot_unavailable: "teacher snapshot is unavailable"
};
const TEACHER_NO_OP_MESSAGES = {
    none: "the latest processed export produced teacher artifacts",
    duplicate_export: "the latest cycle was a no-op because the export was already seen",
    queue_full: "the latest cycle was a no-op because the teacher queue was full",
    no_teacher_artifacts: "the latest cycle was a no-op because no teacher artifacts were produced",
    empty_scan: "the latest cycle was a no-op because the scanner did not produce any events",
    unavailable: "the latest cycle is not visible from the current operator snapshot"
};
function summarizeStatusInstallHook(openclawHome) {
    const hook = inspectOpenClawBrainHookStatus(openclawHome);
    return {
        state: hook.installState === "unverified" ? "unknown" : hook.installState,
        loadability: hook.loadability,
        installLayout: hook.installLayout,
        hookPath: hook.hookPath,
        detail: hook.detail
    };
}
function summarizeStatusHookLoad(installHook, status) {
    return {
        installState: installHook.state === "unknown" ? "unverified" : installHook.state,
        loadability: installHook.loadability,
        loadProof: status.hook.loadProof,
        guardSeverity: status.hook.guardSeverity,
        guardActionability: status.hook.guardActionability,
        guardSummary: status.hook.guardSummary,
        guardAction: status.hook.guardAction,
        detail: status.hook.detail
    };
}
function summarizeStatusConfigLoad(openclawHome) {
    if (openclawHome === null) {
        return {
            state: "unverified",
            detail: "plugin allowlist state is unknown from activation-root-only status; pin --openclaw-home to prove config load state"
        };
    }
    const allowlist = inspectOpenClawBrainPluginAllowlist(openclawHome);
    if (allowlist.state === "blocked") {
        return {
            state: "blocked",
            detail: allowlist.detail
        };
    }
    if (allowlist.state === "invalid") {
        return {
            state: "invalid",
            detail: allowlist.detail
        };
    }
    return {
        state: "allows_load",
        detail: allowlist.detail
    };
}
function summarizeStatusHookFilesState(installHook) {
    if (installHook.state === "installed" || installHook.state === "blocked_by_allowlist") {
        return "present";
    }
    if (installHook.state === "not_installed") {
        return "missing";
    }
    return "unverified";
}
function summarizeStatusAttachmentWatcher(status) {
    if (status.passiveLearning.watchState === "watching") {
        return "alive";
    }
    if (status.passiveLearning.watchState === "stale_snapshot") {
        return "stale";
    }
    return "not_visible";
}
function formatAttachedProfileIdentity(reference) {
    const profileLabel = reference.inspection.profileId ?? "current_profile";
    return `${profileLabel}@${shortenPath(path.resolve(reference.openclawHome))}`;
}
function buildStatusAttachedProfileTruth(input) {
    const resolvedOpenClawHome = canonicalizeExistingCliPath(input.reference.openclawHome);
    const installHook = summarizeStatusInstallHook(resolvedOpenClawHome);
    const configLoad = summarizeStatusConfigLoad(resolvedOpenClawHome);
    const runtimeProof = input.runtimeProofByHome.get(resolvedOpenClawHome);
    const currentOpenClawHome = input.currentOpenClawHome === null ? null : canonicalizeExistingCliPath(input.currentOpenClawHome);
    const hookFiles = summarizeStatusHookFilesState(installHook);
    return {
        label: formatAttachedProfileIdentity(input.reference),
        openclawHome: resolvedOpenClawHome,
        current: currentOpenClawHome !== null && currentOpenClawHome === resolvedOpenClawHome,
        hookFiles,
        configLoad: configLoad.state,
        runtimeLoad: input.runtimeProofError !== null
            ? "proof_error"
            : hookFiles !== "present"
                ? "not_proven"
                : runtimeProof !== undefined
                    ? "proven"
                    : "not_proven",
        runtimeLoadedAt: runtimeProof?.loadedAt ?? null
    };
}
function buildCurrentStatusAttachmentTruthDetail(input) {
    const attachedProfileCount = input.attachedProfiles.length;
    const attachedProfilesDetail = attachedProfileCount === 0
        ? "no attached profiles were discovered for this activation root"
        : `attached profile set has ${attachedProfileCount} discoverable ${attachedProfileCount === 1 ? "profile" : "profiles"}`;
    if (input.openclawHome === null) {
        return ("current-profile hook/config/runtime load truth is unverified without --openclaw-home; " +
            `${attachedProfilesDetail}`);
    }
    if (input.runtimeProofError !== null) {
        return (`current profile ${input.currentProfileLabel} could not prove runtime load because ${shortenPath(input.runtimeProofPath)} is unreadable: ${input.runtimeProofError}; ` +
            `${attachedProfilesDetail}`);
    }
    const hookDetail = input.hookFiles === "present"
        ? "hook files are present"
        : input.hookFiles === "missing"
            ? "hook files are missing"
            : "hook files are unverified";
    const configDetail = input.configLoad === "allows_load"
        ? "config allows load"
        : input.configLoad === "blocked"
            ? "config blocks load"
            : input.configLoad === "invalid"
                ? "config is invalid for load proof"
                : "config load is unverified";
    const runtimeDetail = input.runtimeLoad === "proven"
        ? "runtime load is proven"
        : input.runtimeLoad === "not_proven"
            ? "runtime load is not yet proven"
            : input.runtimeLoad === "proof_error"
                ? "runtime load proof is broken"
                : "runtime load is unverified";
    const watcherDetail = input.watcher === "alive"
        ? "watcher is alive"
        : input.watcher === "stale"
            ? "watcher visibility is stale"
            : "watcher is not visible";
    return `current profile ${input.currentProfileLabel}: ${hookDetail}, ${configDetail}, ${runtimeDetail}, ${watcherDetail}; ${attachedProfilesDetail}`;
}
function summarizeStatusAttachmentTruth(input) {
    const runtimeProofs = listOpenClawProfileRuntimeLoadProofs(input.activationRoot);
    const runtimeProofByHome = new Map();
    for (const proof of runtimeProofs.proofs?.profiles ?? []) {
        runtimeProofByHome.set(canonicalizeExistingCliPath(proof.openclawHome), {
            loadedAt: proof.loadedAt
        });
    }
    const attachedProfiles = findInstalledHookReferencesForActivationRoot({
        activationRoot: input.activationRoot
    }).map((reference) => buildStatusAttachedProfileTruth({
        reference,
        currentOpenClawHome: input.openclawHome,
        runtimeProofByHome,
        runtimeProofError: runtimeProofs.error
    }));
    const currentInspection = input.openclawHome === null ? null : inspectOpenClawHome(input.openclawHome);
    const currentProfileLabel = currentInspection?.profileId ?? "current_profile";
    const installHook = summarizeStatusInstallHook(input.openclawHome);
    const configLoad = summarizeStatusConfigLoad(input.openclawHome);
    const currentRuntimeProof = input.openclawHome === null ? undefined : runtimeProofByHome.get(canonicalizeExistingCliPath(input.openclawHome));
    const hookFiles = summarizeStatusHookFilesState(installHook);
    const watcher = summarizeStatusAttachmentWatcher(input.status);
    const runtimeLoad = input.openclawHome === null
        ? "unverified"
        : runtimeProofs.error !== null
            ? "proof_error"
            : hookFiles !== "present"
                ? "not_proven"
                : currentRuntimeProof !== undefined
                    ? "proven"
                    : "not_proven";
    return {
        currentProfileLabel,
        hookFiles,
        configLoad: configLoad.state,
        runtimeLoad,
        watcher,
        attachedProfiles,
        runtimeProofPath: runtimeProofs.path,
        runtimeProofError: runtimeProofs.error,
        detail: buildCurrentStatusAttachmentTruthDetail({
            openclawHome: input.openclawHome,
            currentProfileLabel,
            hookFiles,
            configLoad: configLoad.state,
            runtimeLoad,
            watcher,
            attachedProfiles,
            runtimeProofPath: runtimeProofs.path,
            runtimeProofError: runtimeProofs.error
        })
    };
}
function normalizeAttachmentPolicyMode(value) {
    return value === "undeclared" || value === "dedicated" || value === "shared"
        ? value
        : null;
}
function applyAttachmentPolicyTruth(status, report) {
    const referenceCount = findInstalledHookReferencesForActivationRoot({
        activationRoot: status.host.activationRoot
    }).length;
    const declaration = loadAttachmentPolicyDeclaration(status.host.activationRoot);
    const resolvedPolicy = resolveEffectiveAttachmentPolicyTruth({
        statusPolicy: normalizeAttachmentPolicyMode(status.attachment.policyMode),
        reportPolicy: report === null
            ? null
            : normalizeAttachmentPolicyMode(report.manyProfile.declaredAttachmentPolicy),
        declaredPolicy: declaration.declaration?.policy ?? null,
        referenceCount
    });
    const effectivePolicy = resolvedPolicy.effectivePolicy;
    if (effectivePolicy === null) {
        return {
            status,
            report
        };
    }
    const nextStatusPolicy = resolvedPolicy.statusPolicy;
    const nextReportPolicy = report === null
        ? null
        : resolvedPolicy.reportPolicy;
    return {
        status: nextStatusPolicy === status.attachment.policyMode
            ? status
            : {
                ...status,
                attachment: {
                    ...status.attachment,
                    policyMode: nextStatusPolicy
                }
            },
        report: report === null || nextReportPolicy === report.manyProfile.declaredAttachmentPolicy
            ? report
            : {
                ...report,
                manyProfile: {
                    ...report.manyProfile,
                    declaredAttachmentPolicy: nextReportPolicy
                }
            }
    };
}
function runOllamaProbe(args, baseUrl) {
    try {
        execFileSync("ollama", [...args], {
            stdio: "pipe",
            timeout: 2_000,
            env: {
                ...process.env,
                OLLAMA_HOST: baseUrl
            }
        });
        return {
            detected: true,
            detail: `ollama responded to ${args.join(" ")} at ${baseUrl}`
        };
    }
    catch (error) {
        if (error instanceof Error && "code" in error && error.code === "ENOENT") {
            return {
                detected: false,
                detail: "ollama CLI was not found on PATH"
            };
        }
        return {
            detected: true,
            detail: describeExecFailure(error)
        };
    }
}
function summarizeStatusEmbeddings(report, providerConfig) {
    let embeddedEntryCount = null;
    let totalEntryCount = null;
    let models = [];
    let liveState = "unknown";
    let liveDetail = "no activation-ready active pack is available for embedding inspection";
    if (report.active !== null && report.active.activationReady) {
        try {
            const activePack = loadPackFromActivation(report.activationRoot, "active", {
                requireActivationReady: true
            });
            if (activePack !== null) {
                const summary = summarizePackVectorEmbeddingState(activePack.vectors);
                totalEntryCount = summary.vectorEntryCount;
                embeddedEntryCount = summary.numericEmbeddingEntryCount;
                models = summary.embeddingModels;
                liveState = embeddedEntryCount === null ? "unknown" : embeddedEntryCount > 0 ? "yes" : "no";
                liveDetail = embeddedEntryCount === null || totalEntryCount === null
                    ? "active pack vector entries were unreadable during embedding inspection"
                    : `active pack stores ${embeddedEntryCount}/${totalEntryCount} numeric embeddings`;
            }
        }
        catch (error) {
            liveDetail = `embedding inspection failed: ${toErrorMessage(error)}`;
        }
    }
    if (providerConfig.embedder.provider === "off") {
        return {
            provider: providerConfig.embedder.provider,
            model: providerConfig.embedder.model,
            provisionedState: "off",
            liveState,
            embeddedEntryCount,
            totalEntryCount,
            models,
            detail: `${liveDetail}; embedder provider is off`
        };
    }
    if (providerConfig.embedder.provider === "keywords") {
        return {
            provider: providerConfig.embedder.provider,
            model: providerConfig.embedder.model,
            provisionedState: "builtin",
            liveState,
            embeddedEntryCount,
            totalEntryCount,
            models,
            detail: `${liveDetail}; keyword embedder needs no Ollama model provision`
        };
    }
    const modelProbe = runOllamaProbe(["show", providerConfig.embedder.model], providerConfig.embedderBaseUrl);
    return {
        provider: providerConfig.embedder.provider,
        model: providerConfig.embedder.model,
        provisionedState: modelProbe.detected && /responded to/.test(modelProbe.detail) ? "confirmed" : "not_confirmed",
        liveState,
        embeddedEntryCount,
        totalEntryCount,
        models,
        detail: `${liveDetail}; ollama model check: ${modelProbe.detail}`
    };
}
function summarizeStatusLocalLlm(providerConfig) {
    const detection = runOllamaProbe(["--version"], providerConfig.teacherBaseUrl);
    const enabled = providerConfig.teacher.provider === "ollama";
    if (enabled) {
        return {
            detected: detection.detected,
            enabled,
            provider: providerConfig.teacher.provider,
            model: providerConfig.teacher.model,
            detail: detection.detected
                ? `teacher provider is ollama and the local LLM surface answered at ${providerConfig.teacherBaseUrl}`
                : `teacher provider is ollama but the local LLM surface was not detected (${detection.detail})`
        };
    }
    return {
        detected: detection.detected,
        enabled,
        provider: providerConfig.teacher.provider,
        model: providerConfig.teacher.model,
        detail: detection.detected
            ? `local Ollama is detectable, but teacher labeling is ${providerConfig.teacher.provider}`
            : `teacher labeling is ${providerConfig.teacher.provider}; no local Ollama CLI was detected`
    };
}
function summarizeStatusTeacher(report, providerConfig, localLlm) {
    const enabled = providerConfig.teacher.provider === "ollama";
    const latestCycle = report.teacherLoop.lastNoOpReason === "unavailable"
        ? "unknown"
        : report.teacherLoop.lastNoOpReason === "none"
            ? "teacher_artifact"
            : "no_op";
    if (!enabled) {
        return {
            model: providerConfig.teacher.model,
            enabled,
            healthy: false,
            stale: false,
            idle: false,
            latestCycle,
            detail: `${providerConfig.teacher.model} is not enabled because teacher labeling is ${providerConfig.teacher.provider}`
        };
    }
    if (!localLlm.detected) {
        return {
            model: providerConfig.teacher.model,
            enabled,
            healthy: false,
            stale: null,
            idle: false,
            latestCycle,
            detail: `${providerConfig.teacher.model} is configured on Ollama, but the local LLM surface is not answering at ${providerConfig.teacherBaseUrl}`
        };
    }
    if (!report.teacherLoop.available) {
        return {
            model: providerConfig.teacher.model,
            enabled,
            healthy: null,
            stale: null,
            idle: null,
            latestCycle,
            detail: `${providerConfig.teacher.model} is enabled on Ollama, but no watch teacher snapshot is visible yet`
        };
    }
    const stale = report.teacherLoop.latestFreshness === "stale" || report.teacherLoop.watchState === "stale_snapshot";
    const idle = report.teacherLoop.running === false &&
        (report.teacherLoop.queueDepth ?? 0) === 0 &&
        report.teacherLoop.failureMode === "none";
    const healthy = report.teacherLoop.failureMode === "none" &&
        stale === false &&
        report.teacherLoop.watchState !== "not_visible";
    const cycleDetail = TEACHER_NO_OP_MESSAGES[report.teacherLoop.lastNoOpReason] ?? "the latest teacher cycle detail is unavailable";
    if (report.teacherLoop.failureMode !== "none" && report.teacherLoop.failureMode !== "unavailable") {
        return {
            model: providerConfig.teacher.model,
            enabled,
            healthy: false,
            stale,
            idle,
            latestCycle,
            detail: report.teacherLoop.failureDetail === null
                ? `${providerConfig.teacher.model} is enabled, but the watch loop recorded ${report.teacherLoop.failureMode}`
                : `${providerConfig.teacher.model} is enabled, but the watch loop recorded ${report.teacherLoop.failureMode}: ${report.teacherLoop.failureDetail}`
        };
    }
    return {
        model: providerConfig.teacher.model,
        enabled,
        healthy,
        stale,
        idle,
        latestCycle,
        detail: `${providerConfig.teacher.model} is enabled on Ollama; ${cycleDetail}`
    };
}
function summarizeStatusEmbedder(embeddings) {
    const provisioned = embeddings.provisionedState === "confirmed" || embeddings.provisionedState === "builtin"
        ? true
        : embeddings.provisionedState === "not_confirmed" || embeddings.provisionedState === "off"
            ? false
            : null;
    const live = embeddings.liveState === "yes" ? true : embeddings.liveState === "no" ? false : null;
    if (embeddings.provider === "off") {
        return {
            model: embeddings.model,
            provisioned,
            live,
            detail: `${embeddings.model} is not provisioned because the embedder provider is off`
        };
    }
    if (embeddings.provider === "keywords") {
        return {
            model: embeddings.model,
            provisioned,
            live,
            detail: "keyword embeddings are builtin, so there is no Ollama model to provision"
        };
    }
    if (provisioned === true && live === true) {
        return {
            model: embeddings.model,
            provisioned,
            live,
            detail: `${embeddings.model} is confirmed on Ollama and the active pack stores live numeric embeddings`
        };
    }
    if (provisioned === true && live === false) {
        return {
            model: embeddings.model,
            provisioned,
            live,
            detail: `${embeddings.model} is confirmed on Ollama, but the active pack still has no live numeric embeddings`
        };
    }
    if (provisioned === false && live === true) {
        return {
            model: embeddings.model,
            provisioned,
            live,
            detail: `${embeddings.model} is not confirmed on Ollama, but the active pack already carries numeric embeddings from an earlier materialization`
        };
    }
    return {
        model: embeddings.model,
        provisioned,
        live,
        detail: embeddings.detail
    };
}
function summarizeStatusRouteFn(status, report) {
    const freshness = report.servePath.refreshStatus ?? status.brain.routeFreshness;
    if (!report.routeFn.available) {
        return {
            available: false,
            freshness,
            trainedAt: report.routeFn.trainedAt,
            updatedAt: report.routeFn.updatedAt,
            usedAt: report.routeFn.usedAt,
            detail: report.routeFn.detail
        };
    }
    let detail = report.routeFn.detail;
    if (report.servePath.usedLearnedRouteFn === true) {
        detail = `current serve proof used the learned route_fn; ${report.routeFn.detail}`;
    }
    else if (report.routeFn.usedAt !== null) {
        detail = `current serve proof did not use the learned route_fn, but the active route_fn last served a learned turn at ${report.routeFn.usedAt}`;
    }
    else if (report.routeFn.updatedAt !== null) {
        detail = `active route_fn was last updated at ${report.routeFn.updatedAt}, but no learned serve use is visible yet for the current pack`;
    }
    return {
        available: true,
        freshness,
        trainedAt: report.routeFn.trainedAt,
        updatedAt: report.routeFn.updatedAt,
        usedAt: report.routeFn.usedAt,
        detail
    };
}
function pushUniqueAlert(target, value) {
    const normalized = value.trim();
    if (normalized.length === 0) {
        return;
    }
    if (target.includes(normalized) === false) {
        target.push(normalized);
    }
}
function summarizeStatusAlerts(report, providerConfig, embeddings, localLlm) {
    const buckets = {
        serviceRisk: [],
        degradedBrain: [],
        cosmeticNoise: []
    };
    for (const finding of report.findings) {
        if (finding.severity === "pass") {
            continue;
        }
        if (SERVICE_RISK_FINDING_CODES.has(finding.code)) {
            pushUniqueAlert(buckets.serviceRisk, finding.summary);
            continue;
        }
        if (DEGRADED_BRAIN_FINDING_CODES.has(finding.code)) {
            pushUniqueAlert(buckets.degradedBrain, finding.summary);
            continue;
        }
        if (COSMETIC_FINDING_CODES.has(finding.code)) {
            pushUniqueAlert(buckets.cosmeticNoise, finding.summary);
            continue;
        }
        pushUniqueAlert(finding.severity === "fail" ? buckets.serviceRisk : buckets.degradedBrain, finding.summary);
    }
    for (const warningState of report.learning.warningStates) {
        const message = LEARNING_WARNING_MESSAGES[warningState];
        if (message === undefined) {
            continue;
        }
        if (warningState === "teacher_snapshot_unavailable") {
            pushUniqueAlert(buckets.cosmeticNoise, message);
        }
        else {
            pushUniqueAlert(buckets.degradedBrain, message);
        }
    }
    if (providerConfig.warnings.length > 0) {
        pushUniqueAlert(buckets.cosmeticNoise, "provider env warnings forced fallback defaults");
    }
    if (localLlm.enabled && !localLlm.detected) {
        pushUniqueAlert(buckets.degradedBrain, "local LLM is enabled but not detected");
    }
    if (embeddings.provider === "ollama" && embeddings.provisionedState !== "confirmed") {
        pushUniqueAlert(buckets.degradedBrain, `embedder model ${embeddings.model} is not confirmed on Ollama`);
    }
    if (embeddings.provider === "ollama" && embeddings.liveState === "no") {
        pushUniqueAlert(buckets.degradedBrain, "embedder is provisioned but the active pack has no live numeric embeddings");
    }
    return buckets;
}
function summarizeStatusWatchState(status) {
    return status.passiveLearning.watchState;
}
function summarizeStatusServeReality(status) {
    if (status.brainStatus.serveState === "serving_active_pack") {
        return "proven_active_pack";
    }
    return status.brainStatus.serveState;
}
function summarizeStatusPromotionState(status) {
    if (status.brain.state === "pg_promoted_pack_authoritative") {
        return "promoted";
    }
    if (status.brain.state === "seed_state_authoritative") {
        return status.passiveLearning.firstExportOccurred ? "seed_authoritative" : "awaiting_first_export";
    }
    return status.brain.state;
}
function formatStatusAlertLine(values) {
    const normalized = values.map((value) => value.trim()).filter((value) => value.length > 0);
    return normalized.length === 0 ? "none" : formatCompactList(normalized, 2, 64);
}
function formatStatusNullableNumber(value, unknown = "unknown") {
    return value === null ? unknown : String(value);
}
function formatStatusNullableYesNo(value) {
    return value === null ? "unknown" : yesNo(value);
}
function formatStatusNullableMilliseconds(value) {
    return value === null ? "none" : `${value.toFixed(2)}ms`;
}
function formatStatusHotPathTiming(timing) {
    return [
        `hotPath=${formatStatusNullableMilliseconds(timing.totalMs)}`,
        `route=${formatStatusNullableMilliseconds(timing.routeSelectionMs)}`,
        `prompt=${formatStatusNullableMilliseconds(timing.promptAssemblyMs)}`,
        `other=${formatStatusNullableMilliseconds(timing.otherMs)}`,
        `background=${timing.backgroundWorkIncluded ? "included" : "excluded"}`
    ].join(" ");
}
function formatStatusObservedDeltaTransition(delta) {
    if (delta.latestPackTransition === null) {
        return "none";
    }
    return `${delta.latestPackTransition.kind}:${delta.latestPackTransition.fromPackId ?? "none"}->${delta.latestPackTransition.toPackId}`;
}
function formatAttachedProfileTruthCompact(entry) {
    const currentPrefix = entry.current ? "*" : "";
    return (`${currentPrefix}${entry.label}` +
        `[hook=${entry.hookFiles} config=${entry.configLoad} runtime=${entry.runtimeLoad}` +
        `${entry.runtimeLoadedAt === null ? "" : `@${entry.runtimeLoadedAt}`}]`);
}
function formatAttachedProfileTruthCompactList(entries) {
    return entries.length === 0
        ? "none"
        : formatCompactList(entries.map((entry) => formatAttachedProfileTruthCompact(entry)), 2, 80);
}
function formatAttachedProfileTruthDetailedList(entries) {
    return entries.length === 0
        ? "none"
        : entries
            .map((entry) => `${entry.current ? "*" : ""}${entry.label}` +
            `[hook=${entry.hookFiles} config=${entry.configLoad} runtime=${entry.runtimeLoad} loadedAt=${entry.runtimeLoadedAt ?? "none"}]`)
            .join(" ");
}
function summarizeDisplayedStatus(status, installHook) {
    return installHook.state === "blocked_by_allowlist" || status.hook.loadability === "blocked"
        ? "fail"
        : status.brainStatus.status;
}
function formatTracedLearningSurface(surface) {
    const detail = surface.error === null ? surface.detail : `${surface.detail}: ${surface.error}`;
    return `present=${yesNo(surface.present)} updated=${surface.updatedAt ?? "none"} routes=${surface.routeTraceCount} supervision=${surface.supervisionCount} updates=${surface.routerUpdateCount} teacher=${surface.teacherArtifactCount} pg=${surface.pgVersionUsed ?? "none"} pack=${surface.materializedPackId ?? "none"} detail=${detail}`;
}
function buildCompactStatusHeader(status, report, options) {
    const installHook = summarizeStatusInstallHook(options.openclawHome);
    const hookLoad = summarizeStatusHookLoad(installHook, status);
    const embeddings = summarizeStatusEmbeddings(report, options.providerConfig);
    const localLlm = summarizeStatusLocalLlm(options.providerConfig);
    const teacher = summarizeStatusTeacher(report, options.providerConfig, localLlm);
    const embedder = summarizeStatusEmbedder(embeddings);
    const routeFn = summarizeStatusRouteFn(status, report);
    const alerts = summarizeStatusAlerts(report, options.providerConfig, embeddings, localLlm);
    const liveModels = embeddings.models.length === 0 ? "none" : embeddings.models.join("|");
    const attachmentTruth = summarizeStatusAttachmentTruth({
        activationRoot: status.host.activationRoot,
        openclawHome: options.openclawHome,
        status
    });
    const tracedLearning = options.tracedLearning ?? buildTracedLearningStatusSurface(status.host.activationRoot);
    return [
        `lifecycle   attach=${status.attachment.state} learner=${yesNo(status.passiveLearning.learnerRunning)} watch=${summarizeStatusWatchState(status)} export=${status.passiveLearning.exportState} promote=${summarizeStatusPromotionState(status)} serve=${summarizeStatusServeReality(status)}`,
        `hook        install=${hookLoad.installState} loadability=${hookLoad.loadability} loadProof=${hookLoad.loadProof} layout=${status.hook.installLayout ?? "unverified"} additional=${status.hook.additionalInstallCount ?? 0} severity=${hookLoad.guardSeverity} actionability=${hookLoad.guardActionability} summary=${hookLoad.guardSummary}`,
        `attachTruth current=${attachmentTruth.currentProfileLabel} hook=${attachmentTruth.hookFiles} config=${attachmentTruth.configLoad} runtime=${attachmentTruth.runtimeLoad} watcher=${attachmentTruth.watcher} attachedSet=${formatAttachedProfileTruthCompactList(attachmentTruth.attachedProfiles)} why=${attachmentTruth.detail}`,
        `passive     firstExport=${yesNo(status.passiveLearning.firstExportOccurred)} backlog=${status.passiveLearning.backlogState} pending=${formatStatusNullableNumber(status.passiveLearning.pendingLive)}/${formatStatusNullableNumber(status.passiveLearning.pendingBackfill)}`,
        `serving     pack=${status.passiveLearning.currentServingPackId ?? "none"} lastExport=${status.passiveLearning.lastExportAt ?? "none"} lastPromotion=${status.passiveLearning.lastPromotionAt ?? "none"}`,
        `timing      ${formatStatusHotPathTiming(status.brainStatus.timing)}`,
        `delta       observed=${status.passiveLearning.lastObservedDelta.observedAt ?? "none"} exported=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.exported)} labeled=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.labeled)} promoted=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.promoted)} served=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.served)} transition=${formatStatusObservedDeltaTransition(status.passiveLearning.lastObservedDelta)}`,
        `changed     ${status.passiveLearning.lastObservedDelta.explanation}`,
        `explain     ${status.brain.summary}`,
        `graph       blocks=${report.graph.blockCount ?? "none"} strongest=${report.graph.strongestBlockId ?? "none"} latest=${report.graph.latestMaterialization.packId ?? "none"} latestChanged=${yesNo(report.graph.latestMaterialization.changed)} connect=${formatCompactGraphConnectDiagnostics(report.graph.latestMaterialization.connectDiagnostics ?? report.graph.connectDiagnostics)}`,
        `teacher     model=${teacher.model} enabled=${yesNo(teacher.enabled)} healthy=${yesNo(teacher.healthy)} stale=${yesNo(teacher.stale)} idle=${yesNo(teacher.idle)} cycle=${teacher.latestCycle} why=${teacher.detail}`,
        `embedder    model=${embedder.model} provisioned=${yesNo(embedder.provisioned)} live=${yesNo(embedder.live)} why=${embedder.detail}`,
        `routeFn     available=${yesNo(routeFn.available)} freshness=${routeFn.freshness} trained=${routeFn.trainedAt ?? "none"} updated=${routeFn.updatedAt ?? "none"} used=${routeFn.usedAt ?? "none"} why=${routeFn.detail}`,
        `traced      ${formatTracedLearningSurface(tracedLearning)}`,
        `embeddings  provider=${embeddings.provider} provisioned=${embeddings.provisionedState} live=${embeddings.liveState} stored=${embeddings.embeddedEntryCount ?? "none"}/${embeddings.totalEntryCount ?? "none"} models=${liveModels}`,
        `localLLM    detected=${yesNo(localLlm.detected)} enabled=${yesNo(localLlm.enabled)} provider=${localLlm.provider} model=${localLlm.model}`,
        `alerts      service_risk=${formatStatusAlertLine(alerts.serviceRisk)} degraded_brain=${formatStatusAlertLine(alerts.degradedBrain)} cosmetic_noise=${formatStatusAlertLine(alerts.cosmeticNoise)}`
    ];
}
function formatCurrentProfileStatusSummary(status, report, targetInspection, options) {
    const installHook = summarizeStatusInstallHook(options.openclawHome);
    const displayedStatus = summarizeDisplayedStatus(status, installHook);
    const embeddings = summarizeStatusEmbeddings(report, options.providerConfig);
    const localLlm = summarizeStatusLocalLlm(options.providerConfig);
    const liveModels = embeddings.models.length === 0 ? "none" : embeddings.models.join("|");
    const attachmentTruth = summarizeStatusAttachmentTruth({
        activationRoot: status.host.activationRoot,
        openclawHome: options.openclawHome,
        status
    });
    const tracedLearning = options.tracedLearning ?? buildTracedLearningStatusSurface(status.host.activationRoot);
    const profileIdSuffix = status.profile.profileId === null ? "" : ` id=${status.profile.profileId}`;
    const targetLine = targetInspection === null
        ? `target      activation=${status.host.activationRoot} source=activation_root_only`
        : `target      activation=${status.host.activationRoot} ${formatOpenClawTargetLine(targetInspection)} hook=${status.hook.hookPath === null ? "unverified" : shortenPath(status.hook.hookPath)}`;
    return [
        `STATUS ${displayedStatus}`,
        ...buildCompactStatusHeader(status, report, options),
        `answer      ${status.brain.summary}`,
        targetLine,
        ...(targetInspection === null ? [] : [`preflight   ${formatOpenClawTargetExplanation(targetInspection)}`]),
        `next        ${buildStatusNextStep(status, report, {
            openclawHome: options.openclawHome,
            installHook
        })}`,
        `host        runtime=${status.host.runtimeOwner} activation=${status.host.activationRoot}`,
        `profile     selector=${status.profile.selector}${profileIdSuffix} attachment=${status.attachment.state} policy=${status.attachment.policyMode}`,
        `guard       severity=${status.hook.guardSeverity} actionability=${status.hook.guardActionability} action=${status.hook.guardAction} summary=${status.hook.guardSummary}`,
        `attachTruth current=${attachmentTruth.currentProfileLabel} hook=${attachmentTruth.hookFiles} config=${attachmentTruth.configLoad} runtime=${attachmentTruth.runtimeLoad} watcher=${attachmentTruth.watcher} detail=${attachmentTruth.detail}`,
        `attachedSet ${formatAttachedProfileTruthDetailedList(attachmentTruth.attachedProfiles)} proofPath=${shortenPath(attachmentTruth.runtimeProofPath)} proofError=${attachmentTruth.runtimeProofError ?? "none"}`,
        `manyProfile surface=${report.manyProfile.operatorSurface} policy=${report.manyProfile.declaredAttachmentPolicy} intent=${report.manyProfile.sameGatewayIntent} checkedProof=${report.manyProfile.checkedInProofTopology} sameGatewayProof=${yesNo(report.manyProfile.sameGatewayProof)} sharedWriteProof=${yesNo(report.manyProfile.sharedWriteSafetyProof)}`,
        `activation  state=${status.brainStatus.activationState} detail=${status.brain.detail}`,
        `brain       pack=${status.brain.activePackId ?? "none"} state=${status.brain.state} init=${status.brain.initMode ?? "unknown"} routeFreshness=${status.brain.routeFreshness} lastPromotion=${status.brain.lastPromotionAt ?? "none"} router=${status.brain.routerIdentity ?? "none"}`,
        `serve       state=${status.brainStatus.serveState} failOpen=${yesNo(status.brainStatus.failOpen)} hardFail=${yesNo(report.servePath.hardRequirementViolated)} usedRouteFn=${yesNo(status.brainStatus.usedLearnedRouteFn)} awaitingFirstExport=${yesNo(status.brainStatus.awaitingFirstExport)} detail=${status.brainStatus.detail}`,
        `route       router=${report.servePath.routerIdentity ?? status.brain.routerIdentity ?? "none"} supervision=${report.servePath.refreshStatus ?? status.brain.routeFreshness} freshness=${report.servePath.freshnessChecksum ?? "none"}`,
        `budget      requested=${report.servePath.requestedBudgetStrategy ?? "none"} resolved=${report.servePath.resolvedBudgetStrategy ?? "none"} maxBlocks=${report.servePath.resolvedMaxContextBlocks ?? "none"} source=${report.servePath.structuralBudgetSource ?? "none"} origin=${status.brainStatus.structuralDecision.origin} basis=${status.brainStatus.structuralDecision.basis}`,
        `decision    ${status.brainStatus.structuralDecision.detail}`,
        `principal   latest=${formatPrincipalLatest(report)} pending=${report.principal.pendingCount ?? report.learning.pendingPrincipalCount ?? "none"} checkpoint=${formatPrincipalCheckpointFrontier(report)} downstream=${yesNo(report.principal.servingDownstreamOfLatestCorrection)} lag=${report.learning.principalLagToPromotion.sequenceLag ?? "none"}`,
        `passive     learner=${yesNo(status.passiveLearning.learnerRunning)} firstExport=${yesNo(status.passiveLearning.firstExportOccurred)} watch=${status.passiveLearning.watchState} export=${status.passiveLearning.exportState} backlog=${status.passiveLearning.backlogState} pending=${formatStatusNullableNumber(status.passiveLearning.pendingLive)}/${formatStatusNullableNumber(status.passiveLearning.pendingBackfill)} detail=${status.passiveLearning.detail}`,
        `delta       observed=${status.passiveLearning.lastObservedDelta.observedAt ?? "none"} exported=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.exported)} labeled=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.labeled)} promoted=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.promoted)} served=${formatStatusNullableYesNo(status.passiveLearning.lastObservedDelta.served)} transition=${formatStatusObservedDeltaTransition(status.passiveLearning.lastObservedDelta)} detail=${status.passiveLearning.lastObservedDelta.explanation}`,
        `scanner     flowing=${yesNo(report.supervision.flowing)} scan=${report.supervision.scanPolicy ?? "none"} surfaces=${formatScannerSurfaces(report)} labels=${report.supervision.humanLabelCount ?? "none"}/${report.supervision.selfLabelCount ?? "none"} attributable=${report.supervision.attributedEventCount ?? "none"}/${report.supervision.totalEventCount ?? "none"} digests=${report.supervision.selectionDigestCount ?? "none"}`,
        `labels      ${formatLabelFlowSummary(report.labelFlow)}`,
        `graph       source=${report.graph.runtimePlasticitySource ?? "none"} blocks=${report.graph.blockCount ?? "none"} strongest=${report.graph.strongestBlockId ?? "none"} ops=${formatStructuralOps(report)} latest=${report.graph.latestMaterialization.packId ?? "none"} latestChanged=${yesNo(report.graph.latestMaterialization.changed)} connect=${formatGraphConnectDiagnostics(report.graph.latestMaterialization.connectDiagnostics ?? report.graph.connectDiagnostics)} summary=${formatGraphSummary(report)}`,
        `path        ${formatOperatorLearningPathSummary({
            status,
            learningPath: report.learningPath,
            tracedLearning
        })}`,
        `learning    state=${report.learning.backlogState} bootstrapped=${yesNo(report.learning.bootstrapped)} mode=${report.learning.mode} next=${report.learning.nextPriorityLane} priority=${report.learning.nextPriorityBucket} pending=${report.learning.pendingLive ?? "none"}/${report.learning.pendingBackfill ?? "none"} buckets=${formatLearningBuckets(report)} warn=${formatLearningWarnings(report)} lastPack=${report.learning.lastMaterializedPackId ?? "none"} detail=${report.learning.detail}`,
        `traced      ${formatTracedLearningSurface(tracedLearning)}`,
        `teacherProof ${formatTeacherLoopSummary(report)}`,
        `watch       cadence=${report.teacherLoop.learningCadence} scan=${report.teacherLoop.scanPolicy} heartbeat=${report.teacherLoop.lastHeartbeatAt ?? "none"} interval=${report.teacherLoop.pollIntervalSeconds ?? "none"} replayed=${report.teacherLoop.replayedBundleCount ?? "none"}/${report.teacherLoop.replayedEventCount ?? "none"} exported=${report.teacherLoop.exportedBundleCount ?? "none"}/${report.teacherLoop.exportedEventCount ?? "none"} tail=${report.teacherLoop.sessionTailSessionsTracked ?? "none"}/${report.teacherLoop.sessionTailBridgedEventCount ?? "none"} tailState=${report.teacherLoop.localSessionTailNoopReason ?? "none"} lastJob=${report.teacherLoop.lastAppliedMaterializationJobId ?? "none"} lastPack=${report.teacherLoop.lastMaterializedPackId ?? "none"}`,
        `embeddings  provider=${embeddings.provider} provisioned=${embeddings.provisionedState} live=${embeddings.liveState} stored=${embeddings.embeddedEntryCount ?? "none"}/${embeddings.totalEntryCount ?? "none"} models=${liveModels}`,
        `localLLM    detected=${yesNo(localLlm.detected)} enabled=${yesNo(localLlm.enabled)} provider=${localLlm.provider} model=${localLlm.model}`,
        `rollback    ready=${yesNo(report.rollback.allowed)} state=${report.rollback.state} previous=${report.rollback.previousPackId ?? "none"}`,
        `proof       lastExport=${status.brain.lastExportAt ?? "none"} lastLearningUpdate=${status.brain.lastLearningUpdateAt ?? "none"} lastPromotion=${status.brain.lastPromotionAt ?? "none"}`,
        `logs        root=${status.brain.logRoot ?? "none"}`,
        `turn        attribution=${status.currentTurnAttribution === null ? "none" : status.currentTurnAttribution.contract}`
    ].join("\n");
}
// Auto-detection of activation root is now handled by the shared
// resolveActivationRoot() helper in resolve-activation-root.ts.
// It is imported at the top and used by requireActivationRoot below.
function shortenPath(fullPath) {
    const homeDir = process.env.HOME ?? "";
    if (homeDir.length > 0 && fullPath.startsWith(homeDir)) {
        return "~" + fullPath.slice(homeDir.length);
    }
    return fullPath;
}
function formatOpenClawTargetLine(inspection) {
    const profilePart = inspection.profileId === null
        ? "profile=current_profile"
        : `profile=${inspection.profileId} via ${formatOpenClawHomeProfileSource(inspection.profileSource)}`;
    return `home=${shortenPath(inspection.openclawHome)} layout=${formatOpenClawHomeLayout(inspection.layout)} ${profilePart}`;
}
function formatOpenClawTargetExplanation(inspection) {
    return describeOpenClawHomeInspection(inspection);
}
function buildInstallStatusCommand(activationRoot) {
    return `openclawbrain status --activation-root ${quoteShellArg(activationRoot)}`;
}
function buildLearnerServiceStatusCommand(activationRoot) {
    return `openclawbrain daemon status --activation-root ${quoteShellArg(activationRoot)}`;
}
function buildGatewayRestartCommand(profileId) {
    return `env -i HOME="$HOME" PATH="$PATH" openclaw --profile ${quoteShellArg(profileId)} gateway restart`;
}
function buildGatewayStatusCommand(profileId) {
    return `env -i HOME="$HOME" PATH="$PATH" openclaw --profile ${quoteShellArg(profileId)} gateway status`;
}
function buildInstallCommand(openclawHome) {
    return `openclawbrain install --openclaw-home ${quoteShellArg(openclawHome)}`;
}
function buildAttachCommand(openclawHome, activationRoot = null) {
    const parts = ["openclawbrain", "attach", "--openclaw-home", quoteShellArg(openclawHome)];
    if (activationRoot !== null) {
        parts.push("--activation-root", quoteShellArg(activationRoot));
    }
    return parts.join(" ");
}
function buildInstallEmbedderProvisionCommand(baseUrl, model) {
    return `OLLAMA_HOST=${quoteShellArg(baseUrl)} ollama pull ${quoteShellArg(model)}`;
}
function describeExecOutput(value) {
    if (typeof value === "string") {
        const normalized = value.trim();
        return normalized.length > 0 ? normalized : null;
    }
    if (value instanceof Buffer) {
        const normalized = value.toString("utf8").trim();
        return normalized.length > 0 ? normalized : null;
    }
    return null;
}
function describeExecFailure(error) {
    if (error instanceof Error) {
        const childError = error;
        if (childError.code === "ENOENT") {
            return "ollama was not found on PATH";
        }
        const stderr = describeExecOutput(childError.stderr);
        if (stderr !== null) {
            return stderr;
        }
        const stdout = describeExecOutput(childError.stdout);
        if (stdout !== null) {
            return stdout;
        }
        const message = childError.message.trim();
        if (message.length > 0) {
            return message;
        }
    }
    return String(error);
}
function toErrorMessage(error) {
    return error instanceof Error ? error.message : String(error);
}
function ensureInstallEmbedderReady(parsed) {
    const providerConfig = readOpenClawBrainProviderConfig(process.env);
    const model = DEFAULT_OLLAMA_EMBEDDING_MODEL;
    const baseUrl = providerConfig.embedderBaseUrl;
    if (parsed.skipEmbedderProvision) {
        const skipReason = parsed.skipEmbedderProvisionSource === "flag"
            ? "--skip-embedder-provision"
            : `${OPENCLAWBRAIN_INSTALL_SKIP_EMBEDDER_PROVISION_ENV}=1`;
        return {
            state: "skipped",
            model,
            baseUrl,
            detail: `Skipped default embedder provisioning (${skipReason}); ${parsed.command} continued only because the operator explicitly opted out. ` +
                `Provision it later with ${buildInstallEmbedderProvisionCommand(baseUrl, model)}.`
        };
    }
    try {
        execFileSync("ollama", ["pull", model], {
            stdio: "pipe",
            env: {
                ...process.env,
                OLLAMA_HOST: baseUrl
            }
        });
    }
    catch (error) {
        const detail = describeExecFailure(error);
        throw new Error(`Default embedder provisioning failed before brain init. Tried ${buildInstallEmbedderProvisionCommand(baseUrl, model)}. ` +
            `${parsed.command === "install" ? "Install" : "Attach"} stops here so the bootstrap path does not quietly continue without ${model}. ` +
            `Fix Ollama and rerun ${parsed.command}, or explicitly skip with --skip-embedder-provision or ${OPENCLAWBRAIN_INSTALL_SKIP_EMBEDDER_PROVISION_ENV}=1. ` +
            `Detail: ${detail}`);
    }
    return {
        state: "ensured",
        model,
        baseUrl,
        detail: `Ensured default embedder before brain bootstrap: ${buildInstallEmbedderProvisionCommand(baseUrl, model)}`
    };
}
function parseOllamaListModelNames(output) {
    return output
        .split(/\r?\n/u)
        .map((line) => line.trim())
        .filter((line) => line.length > 0 && !/^name\s+/iu.test(line))
        .map((line) => line.split(/\s+/u)[0] ?? "")
        .filter((name) => name.length > 0);
}
function selectCompatibleLocalTeacherModel(models) {
    const normalized = models.map((model) => model.trim()).filter((model) => model.length > 0);
    for (const prefix of INSTALL_COMPATIBLE_LOCAL_TEACHER_MODEL_PREFIXES) {
        const exact = normalized.find((model) => model === prefix);
        if (exact !== undefined) {
            return exact;
        }
        const variant = normalized.find((model) => model.startsWith(`${prefix}-`) ||
            model.startsWith(`${prefix}_`) ||
            model.startsWith(`${prefix}.`));
        if (variant !== undefined) {
            return variant;
        }
    }
    return null;
}
function detectInstallTeacherDefaults(baseUrl) {
    try {
        const output = execFileSync("ollama", ["list"], {
            stdio: "pipe",
            env: {
                ...process.env,
                OLLAMA_HOST: baseUrl
            }
        }).toString("utf8");
        const availableModels = parseOllamaListModelNames(output);
        const model = selectCompatibleLocalTeacherModel(availableModels);
        if (model === null) {
            return {
                provider: "heuristic",
                model: null,
                baseUrl,
                availableModels,
                detectionDetail: availableModels.length === 0
                    ? `No compatible local Ollama teacher model detected on ${baseUrl}; watch keeps heuristic teacher defaults.`
                    : `No compatible local Ollama teacher model detected on ${baseUrl}; saw ${availableModels.join(", ")} and kept heuristic teacher defaults.`
            };
        }
        return {
            provider: "ollama",
            model,
            baseUrl,
            availableModels,
            detectionDetail: `Detected compatible local Ollama teacher model ${model} on ${baseUrl}; watch will enable it by default from the installed activation root.`
        };
    }
    catch (error) {
        const detail = describeExecFailure(error);
        return {
            provider: "heuristic",
            model: null,
            baseUrl,
            availableModels: [],
            detectionDetail: `Local Ollama teacher autodetect failed on ${baseUrl}; kept heuristic teacher defaults. Detail: ${detail}`
        };
    }
}
function writeInstallProviderDefaults(parsed) {
    const providerConfig = readOpenClawBrainProviderConfig(process.env);
    const teacherDetection = detectInstallTeacherDefaults(providerConfig.teacherBaseUrl);
    const defaultsPath = resolveOpenClawBrainProviderDefaultsPath(parsed.activationRoot);
    const defaults = {
        contract: "openclawbrain_provider_defaults.v1",
        writtenAt: new Date().toISOString(),
        source: "install",
        teacherBaseUrl: providerConfig.teacherBaseUrl,
        embedderBaseUrl: providerConfig.embedderBaseUrl,
        teacher: {
            provider: teacherDetection.provider,
            model: teacherDetection.model,
            detectedLocally: teacherDetection.provider === "ollama",
            detectedFromModel: teacherDetection.model
        },
        embedder: {
            provider: "ollama",
            model: DEFAULT_OLLAMA_EMBEDDING_MODEL
        }
    };
    writeFileSync(defaultsPath, JSON.stringify(defaults, null, 2) + "\n", "utf8");
    return {
        path: defaultsPath,
        defaults,
        detail: `Wrote local provider defaults: ${teacherDetection.detectionDetail}`,
        lifecycleSummary: teacherDetection.provider === "ollama" && teacherDetection.model !== null
            ? `Teacher: auto-enabled local Ollama model ${teacherDetection.model} from install-written defaults`
            : "Teacher: no compatible local Ollama model detected; watch stays heuristic unless explicitly overridden"
    };
}
function shouldWriteProfileHookProviderDefaults(parsed, activationPlan, isInstall) {
    if (isInstall || activationPlan.action === "bootstrap") {
        return true;
    }
    return !existsSync(resolveOpenClawBrainProviderDefaultsPath(parsed.activationRoot));
}
function buildInstallBrainFeedbackSummary(input) {
    const providerDefaultsPath = resolveOpenClawBrainProviderDefaultsPath(input.parsed.activationRoot);
    const hookLayout = describeOpenClawBrainInstallLayout(input.hookLayout);
    const embedderState = input.embedderProvision === null ? "unchanged" : input.embedderProvision.state;
    const teacherDefaults = input.providerDefaults?.defaults.teacher;
    const teacherProvider = teacherDefaults?.provider ?? "unknown";
    const teacherModel = teacherDefaults?.model ?? null;
    const detectedLocalLlm = teacherDefaults?.detectedLocally ?? null;
    const profileName = input.targetInspection.profileId;
    const profileSource = input.targetInspection.profileSource;
    const casingGuidance = profileName === null
        ? "Exact OpenClaw --profile casing is unresolved here because this target stays on the host-selected current_profile boundary."
        : `Use the exact OpenClaw profile casing shown here for host-side restart/status commands: ${quoteShellArg(profileName)}.`;
    const attachment = input.parsed.shared
        ? {
            policy: "shared",
            activationRootMode: "shared_root_declared",
            sameGatewayProof: "not_checked_in",
            detail: "Shared activation root declared. Other profiles may point at this same root, but same-gateway many-profile load/serve proof is not checked in."
        }
        : {
            policy: "dedicated",
            activationRootMode: "dedicated_per_profile",
            sameGatewayProof: "not_applicable",
            detail: "Dedicated activation root for this profile/home boundary."
        };
    const restart = profileName === null
        ? {
            exactProfile: false,
            profile: null,
            profileSource,
            guidance: `Operator-owned restart step: this install did not infer an exact --profile token from ${shortenPath(input.targetInspection.openclawHome)}. ` +
                "If immediate load matters, restart the host-selected current_profile from OpenClaw itself; otherwise the next natural launch will pick up the hook.",
            restartCommand: null,
            gatewayStatusCommand: null
        }
        : {
            exactProfile: true,
            profile: profileName,
            profileSource,
            guidance: `Operator-owned restart step: if immediate load matters and profile ${quoteShellArg(profileName)} is already running, run ${buildGatewayRestartCommand(profileName)}. ` +
                `If it is stopped, the next launch of profile ${quoteShellArg(profileName)} will pick up the hook. ${casingGuidance}`,
            restartCommand: buildGatewayRestartCommand(profileName),
            gatewayStatusCommand: buildGatewayStatusCommand(profileName)
        };
    const provedNow = input.activationPlan.action === "bootstrap"
        ? `${hookLayout} prepared, activation root ready, seed/current-profile attach bootstrapped, learner service ${input.learnerService.state}, provider defaults ${input.providerDefaults === null ? "kept" : "written"}`
        : `${hookLayout} prepared, activation root kept, active pack ${input.activationPlan.activePackId ?? "unknown"} preserved, learner service ${input.learnerService.state}${input.providerDefaults === null ? "" : ", provider defaults written"}`;
    const notYetProved = input.learnerService.state === "deferred"
        ? `OpenClaw has not reloaded this hook yet, and passive learner auto-start was deferred; restart plus status still must prove serve-path load, while learner-service start remains a separate operator check`
        : input.activationPlan.action === "bootstrap"
            ? `Passive learning is wired for this activation root, but OpenClaw has not reloaded the hook yet; restart plus status still must prove live startup/load and the first exported turn`
            : `Passive learning is wired for this activation root, but this ${input.parsed.command} run does not itself prove live startup/load after restart`;
    return {
        hookPath: input.hookPath,
        hookLayout: input.hookLayout,
        providerDefaultsPath,
        profile: {
            exactProfileName: profileName,
            profileSource,
            casingGuidance
        },
        attachment,
        restart,
        embedder: {
            provider: "ollama",
            model: DEFAULT_OLLAMA_EMBEDDING_MODEL,
            state: embedderState
        },
        teacher: {
            provider: teacherProvider,
            model: teacherModel,
            detectedLocalLlm
        },
        learnerService: {
            state: input.learnerService.state,
            detail: input.learnerService.detail,
            plistPath: input.learnerService.plistPath,
            logPath: input.learnerService.logPath,
            configuredActivationRoot: input.learnerService.configuredActivationRoot,
            matchesRequestedActivationRoot: input.learnerService.matchesRequestedActivationRoot
        },
        startup: {
            token: "BRAIN_NOT_YET_LOADED",
            proof: "restart_required"
        },
        provedNow,
        notYetProved,
        lines: [
            `target      ${formatOpenClawTargetLine(input.targetInspection)} source=${formatInstallOpenClawHomeSource(input.parsed.openclawHomeSource)}`,
            profileName === null
                ? "profile     exactName=unresolved selector=current_profile casing=not_available"
                : `profile     exactName=${quoteShellArg(profileName)} source=${profileSource} casing=preserved`,
            `hook        layout=${input.hookLayout} path=${shortenPath(input.hookPath)}`,
            `activation  root=${shortenPath(input.parsed.activationRoot)} source=${formatInstallActivationRootSource(input.parsed.activationRootSource)}`,
            `attachment  policy=${attachment.policy} rootMode=${attachment.activationRootMode} sameGatewayProof=${attachment.sameGatewayProof} detail=${attachment.detail}`,
            `defaults    provider-defaults=${shortenPath(providerDefaultsPath)} state=${input.providerDefaults === null ? "unchanged" : "written"}`,
            `embedder    provider=ollama model=${DEFAULT_OLLAMA_EMBEDDING_MODEL} state=${embedderState}`,
            `teacher     provider=${teacherProvider} model=${teacherModel ?? "none"} localLLM=${detectedLocalLlm === null ? "unknown" : yesNo(detectedLocalLlm)}`,
            `learner     state=${input.learnerService.state} detail=${input.learnerService.detail}`,
            `restart     operator=manual exactProfile=${yesNo(restart.exactProfile)} command=${restart.restartCommand ?? "unavailable"}`,
            "startup     BRAIN_NOT_YET_LOADED proof=restart_required",
            `provedNow   ${provedNow}`,
            `notYet      ${notYetProved}`
        ]
    };
}
function buildInstallReloadGuidance(input) {
    if (input.targetInspection.profileId === null) {
        return `Restart later from OpenClaw for the host-selected current_profile behind ${shortenPath(input.targetInspection.openclawHome)} if immediate load matters; this install did not infer an exact --profile token.`;
    }
    return `Restart now if immediate load matters: ${buildGatewayRestartCommand(input.targetInspection.profileId)}`;
}
const LEGACY_PROFILE_NOTE_FILENAMES = ["BRAIN.md", "brain.md"];
const LEGACY_BRAIN_AGENTS_LINE = "5. Read `BRAIN.md` — your learning brain context";
function isLegacyBrainAdvisoryContent(content) {
    return content.includes("## OpenClawBrain")
        && content.includes("You have a learning brain attached at ")
        && content.includes("openclawbrain status --activation-root")
        && content.includes("openclawbrain rollback --activation-root");
}
function writeUpdatedTextFile(filePath, nextText, previousText) {
    const normalizedNextText = previousText.endsWith("\n") ? `${nextText}\n` : nextText;
    writeFileSync(filePath, normalizedNextText, "utf8");
}
function collectProfileResidueDirs(openclawHome) {
    const directories = [path.resolve(openclawHome)];
    try {
        const entries = readdirSync(openclawHome, { withFileTypes: true });
        for (const entry of entries) {
            if (entry.isDirectory() && entry.name.startsWith("workspace-")) {
                directories.push(path.join(openclawHome, entry.name));
            }
        }
    }
    catch {
        // Residue cleanup stays best-effort.
    }
    return directories;
}
function removeLegacyProfileResidue(openclawHome) {
    const removedNotes = [];
    const updatedAgents = [];
    for (const directory of collectProfileResidueDirs(openclawHome)) {
        for (const fileName of LEGACY_PROFILE_NOTE_FILENAMES) {
            const notePath = path.join(directory, fileName);
            if (!existsSync(notePath)) {
                continue;
            }
            try {
                const content = readFileSync(notePath, "utf8");
                if (!isLegacyBrainAdvisoryContent(content)) {
                    continue;
                }
            }
            catch {
                continue;
            }
            rmSync(notePath, { force: true });
            removedNotes.push(notePath);
        }
        const agentsPath = path.join(directory, "AGENTS.md");
        if (!existsSync(agentsPath)) {
            continue;
        }
        let agentsContent;
        try {
            agentsContent = readFileSync(agentsPath, "utf8");
        }
        catch {
            continue;
        }
        const nextContent = agentsContent
            .split("\n")
            .filter((line) => line.trim() !== LEGACY_BRAIN_AGENTS_LINE)
            .join("\n");
        if (nextContent !== agentsContent) {
            writeUpdatedTextFile(agentsPath, nextContent, agentsContent);
            updatedAgents.push(agentsPath);
        }
    }
    return {
        removedNotes,
        updatedAgents
    };
}
function buildCleanupRestartGuidance(restart) {
    if (restart === "never") {
        return "No restart requested. If this OpenClaw profile is currently running, it may keep the previous hook state until the next restart.";
    }
    if (restart === "external") {
        return "Restart this OpenClaw profile externally if it is currently running. If it is stopped, the next launch will pick up the new hook state.";
    }
    return "If this OpenClaw profile is currently running, restart it before expecting the new hook state to take effect. If it is stopped, the next launch will pick it up.";
}
function buildStatusNextStep(status, report, options) {
    const activationRootArg = quoteShellArg(status.host.activationRoot);
    const attachmentTruth = summarizeStatusAttachmentTruth({
        activationRoot: status.host.activationRoot,
        openclawHome: options.openclawHome,
        status
    });
    if (options.installHook.state === "blocked_by_allowlist") {
        if (options.openclawHome === null) {
            return "Repair the OpenClaw plugin allowlist mismatch before trusting serve-path status again.";
        }
        return ("Repair the OpenClaw plugin allowlist mismatch " +
            `(rerun ${buildInstallCommand(options.openclawHome)} or ${buildAttachCommand(options.openclawHome, status.host.activationRoot)}) ` +
            "before trusting serve-path status again.");
    }
    if (status.hook.loadability === "blocked") {
        if (options.openclawHome === null) {
            return "Repair the installed hook so it pins a real activation root before trusting serve-path status again.";
        }
        return (`Repair the installed ${status.hook.installLayout === "native_package_plugin" ? "native package plugin" : "profile hook"} ` +
            `(rerun ${buildInstallCommand(options.openclawHome)} or ${buildAttachCommand(options.openclawHome, status.host.activationRoot)}) ` +
            "before trusting serve-path status again.");
    }
    if (status.brainStatus.activationState === "broken_install") {
        return "Repair or replace the activation root before trusting serve-path status again.";
    }
    if (status.brainStatus.activationState === "stale_incomplete") {
        return "Clean up or repair the retained activation state before reattaching or promoting packs.";
    }
    if (status.brainStatus.status === "fail") {
        return `Run \`openclawbrain status --activation-root ${activationRootArg} --detailed\` before changing lifecycle state so the serve-path failure is explicit.`;
    }
    if (options.openclawHome !== null && options.installHook.state === "not_installed") {
        return `Run \`${buildInstallCommand(options.openclawHome)}\` before expecting this OpenClaw home to load the brain hook.`;
    }
    if (options.openclawHome !== null &&
        attachmentTruth.hookFiles === "present" &&
        attachmentTruth.configLoad === "allows_load" &&
        attachmentTruth.runtimeLoad === "not_proven") {
        return "Restart the exact OpenClaw profile or wait for its next launch, then rerun status until runtime load becomes proven instead of assumed from on-disk state.";
    }
    if (status.brainStatus.awaitingFirstExport) {
        return `Let the attached OpenClaw profile emit a real export, then rerun \`openclawbrain status --activation-root ${activationRootArg}\`.`;
    }
    if (options.openclawHome === null) {
        return `Pin \`--openclaw-home <path>\` when you need exact hook-install truth; activation-root-only status only proves this root's serve-path state.`;
    }
    if (attachmentTruth.runtimeLoad === "proof_error") {
        return "Repair the runtime-load proof file before trusting attach truth again; status now knows the exact file that broke.";
    }
    if (options.installHook.state === "installed" && status.brainStatus.serveState === "serving_active_pack") {
        return "Check the OpenClaw startup log for the `[openclawbrain] BRAIN LOADED` breadcrumb when you need live hook-load proof.";
    }
    if (report.learning.warningStates.includes("principal_live_backlog") ||
        report.learning.warningStates.includes("active_pack_behind_latest_principal")) {
        return "A newer principal correction is still pending promotion; keep the current pack conservative until learner promotion lands.";
    }
    if (report.rollback.allowed) {
        return `Use \`openclawbrain rollback --activation-root ${activationRootArg} --dry-run\` before restoring the previous pack.`;
    }
    return `Use \`openclawbrain status --activation-root ${activationRootArg} --detailed\` when you need the full lifecycle, serve-path, and backlog proof.`;
}
function formatHumanFriendlyStatus(status, report, targetInspection, options) {
    const installHook = summarizeStatusInstallHook(options.openclawHome);
    const displayedStatus = summarizeDisplayedStatus(status, installHook);
    const lines = [
        `STATUS ${displayedStatus}`,
        ...buildCompactStatusHeader(status, report, options),
        ...(targetInspection === null ? [] : [
            `target      ${formatOpenClawTargetLine(targetInspection)}`,
            `preflight   ${formatOpenClawTargetExplanation(targetInspection)}`
        ]),
        `next        ${buildStatusNextStep(status, report, {
            openclawHome: options.openclawHome,
            installHook
        })}`
    ];
    return lines.join("\n");
}
function requireActivationRoot(input, openclawHome, command) {
    const explicitActivationRoot = input.activationRoot.trim().length > 0 ? input.activationRoot : null;
    if (explicitActivationRoot !== null) {
        return path.resolve(explicitActivationRoot);
    }
    if (openclawHome !== null) {
        return resolveActivationRoot({
            openclawHome
        });
    }
    throw new Error(`${command} requires --activation-root <path> or --openclaw-home <path>; unpinned host auto-resolution is no longer supported for ${command}.`);
}
function resolveCliActivationRoot(explicitActivationRoot, openclawHome) {
    return resolveActivationRoot({
        explicit: explicitActivationRoot,
        openclawHome
    });
}
function readJsonFile(filePath) {
    return JSON.parse(readFileSync(path.resolve(filePath), "utf8"));
}
function loadCliScanLiveExport(livePath) {
    const resolvedPath = path.resolve(livePath);
    const stats = statSync(resolvedPath);
    if (stats.isDirectory()) {
        return loadRuntimeEventExportBundle(resolvedPath).normalizedEventExport;
    }
    return readJsonFile(resolvedPath);
}
function formatScanSessionSummary(result) {
    return [
        "SCAN session ok",
        `trace       ${result.bundle.traceId}`,
        `winner      ${result.bundle.summary.winnerMode ?? "none"}`,
        `scores      ${result.bundle.modes.map((mode) => `${mode.mode}=${mode.summary.qualityScore}`).join(" ")}`,
        `turns       ${result.bundle.modes[0]?.turns.length ?? 0}`,
        `hashes      fixture=${result.fixtureHash} score=${result.bundle.scoreHash}`,
        `root        ${result.rootDir}`
    ].join("\n");
}
function formatScanLiveSummary(result, snapshotOutPath) {
    const materializedPackId = result.snapshot.learner.lastMaterialization?.candidate.summary.packId ?? "none";
    const materializationReason = result.snapshot.learner.lastMaterialization?.reason ?? "none";
    const teacherSummary = [
        `artifacts=${result.snapshot.teacher.artifactCount}`,
        `freshness=${result.snapshot.teacher.latestFreshness}`,
        `humanLabels=${result.supervision.humanLabelCount}`
    ];
    if (result.snapshot.diagnostics.lastNoOpReason !== "none") {
        teacherSummary.push(`noop=${result.snapshot.diagnostics.lastNoOpReason}`);
    }
    return [
        "SCAN live ok",
        `source      digest=${result.supervision.exportDigest} session=${result.supervision.sessionId ?? "none"} channel=${result.supervision.channel ?? "none"} range=${result.supervision.eventRange.start}-${result.supervision.eventRange.end}/${result.supervision.eventRange.count}`,
        `teacher     ${teacherSummary.join(" ")}`,
        `labels      source=${result.labelFlow.source} human=${result.labelFlow.humanLabelCount ?? "none"} self=${result.labelFlow.selfLabelCount ?? "none"} implicitPositive=${result.labelFlow.implicitPositiveCount ?? "none"} teacherArtifacts=${result.labelFlow.asyncTeacherArtifactCount ?? "none"}`,
        `path        source=${result.learningPath.source} pg=${result.learningPath.policyGradientVersion} method=${result.learningPath.policyGradientMethod ?? "none"} target=${result.learningPath.targetConstruction ?? "none"} connect=${result.learningPath.connectOpsFired ?? "none"} trajectories=${result.learningPath.reconstructedTrajectoryCount ?? "none"}`,
        `learner     packLabel=${result.packLabel} materialized=${materializedPackId} reason=${materializationReason}`,
        `observed    ${result.observedAt}`,
        `snapshot    ${snapshotOutPath ?? "none"}`
    ].join("\n");
}
export function parseOperatorCliArgs(argv) {
    let command = "status";
    let activationRoot = null;
    let eventExportPath = null;
    let teacherSnapshotPath = null;
    let updatedAt = null;
    let brainAttachmentPolicy = null;
    let sessionPath = null;
    let livePath = null;
    let rootDir = null;
    let workspacePath = null;
    let packLabel = null;
    let workspaceId = null;
    let observedAt = null;
    let snapshotOutPath = null;
    let openclawHome = null;
    let shared = false;
    let skipEmbedderProvision = false;
    let keepData = false;
    let purgeData = false;
    let restart = "safe";
    let restartExplicitlySet = false;
    let json = false;
    let help = false;
    let dryRun = false;
    let detailed = false;
    const args = [...argv];
    if (args[0] === "doctor") {
        throw new Error(buildDoctorDeletedMessage(args.slice(1)));
    }
    if (args[0] === "setup") {
        throw new Error(buildSetupDeletedMessage());
    }
    if (args[0] === "daemon") {
        args.shift();
        return parseDaemonArgs(args);
    }
    if (args[0] === "status" || args[0] === "rollback" || args[0] === "scan" || args[0] === "attach" || args[0] === "install" || args[0] === "detach" || args[0] === "uninstall" || args[0] === "context" || args[0] === "history" || args[0] === "learn" || args[0] === "watch" || args[0] === "export" || args[0] === "import" || args[0] === "reset") {
        command = args.shift();
    }
    if (command === "learn") {
        for (let index = 0; index < args.length; index += 1) {
            const arg = args[index];
            if (arg === "--help" || arg === "-h") {
                help = true;
                continue;
            }
            if (arg === "--json") {
                json = true;
                continue;
            }
            if (arg === "--activation-root") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--activation-root requires a value");
                }
                activationRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--openclaw-home") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--openclaw-home requires a value");
                }
                openclawHome = next;
                index += 1;
                continue;
            }
            if (arg.startsWith("--")) {
                throw new Error(`unknown argument for learn: ${arg}`);
            }
        }
        if (help) {
            return { command, activationRoot: "", json, help };
        }
        return {
            command,
            activationRoot: resolveCliActivationRoot(activationRoot, openclawHome),
            json,
            help
        };
    }
    if (command === "watch") {
        let watchScanRoot = null;
        let watchInterval = 30;
        for (let index = 0; index < args.length; index += 1) {
            const arg = args[index];
            if (arg === "--help" || arg === "-h") {
                help = true;
                continue;
            }
            if (arg === "--json") {
                json = true;
                continue;
            }
            if (arg === "--activation-root") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--activation-root requires a value");
                }
                activationRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--scan-root") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--scan-root requires a value");
                }
                watchScanRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--interval") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--interval requires a value");
                }
                const parsed = Number.parseInt(next, 10);
                if (!Number.isInteger(parsed) || parsed < 1) {
                    throw new Error("--interval must be a positive integer (seconds)");
                }
                watchInterval = parsed;
                index += 1;
                continue;
            }
            if (arg.startsWith("--")) {
                throw new Error(`unknown argument for watch: ${arg}`);
            }
        }
        if (help) {
            return { command, activationRoot: "", scanRoot: null, interval: 30, json, help };
        }
        if (activationRoot === null || activationRoot.trim().length === 0) {
            throw new Error("watch requires --activation-root <path>");
        }
        return {
            command,
            activationRoot: path.resolve(activationRoot),
            scanRoot: watchScanRoot,
            interval: watchInterval,
            json,
            help
        };
    }
    if (command === "context") {
        const messageParts = [];
        for (let index = 0; index < args.length; index += 1) {
            const arg = args[index];
            if (arg === "--help" || arg === "-h") {
                help = true;
                continue;
            }
            if (arg === "--json") {
                json = true;
                continue;
            }
            if (arg === "--activation-root") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--activation-root requires a value");
                }
                activationRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--openclaw-home") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--openclaw-home requires a value");
                }
                openclawHome = next;
                index += 1;
                continue;
            }
            if (arg.startsWith("--")) {
                throw new Error(`unknown argument for context: ${arg}`);
            }
            messageParts.push(arg);
        }
        if (help) {
            return { command, message: "", activationRoot: "", json, help };
        }
        if (messageParts.length === 0) {
            throw new Error("context requires a message argument: openclawbrain context \"your message\"");
        }
        return {
            command,
            message: messageParts.join(" "),
            activationRoot: resolveCliActivationRoot(activationRoot, openclawHome),
            json,
            help
        };
    }
    if (command === "history") {
        let historyLimit = 20;
        for (let index = 0; index < args.length; index += 1) {
            const arg = args[index];
            if (arg === "--help" || arg === "-h") {
                help = true;
                continue;
            }
            if (arg === "--json") {
                json = true;
                continue;
            }
            if (arg === "--activation-root") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--activation-root requires a value");
                }
                activationRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--openclaw-home") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--openclaw-home requires a value");
                }
                openclawHome = next;
                index += 1;
                continue;
            }
            if (arg === "--limit") {
                const next = args[index + 1];
                if (next === undefined) {
                    throw new Error("--limit requires a value");
                }
                const parsed = Number.parseInt(next, 10);
                if (!Number.isInteger(parsed) || parsed <= 0) {
                    throw new Error("--limit must be a positive integer");
                }
                historyLimit = parsed;
                index += 1;
                continue;
            }
            if (arg.startsWith("--")) {
                throw new Error(`unknown argument for history: ${arg}`);
            }
        }
        if (help) {
            return { command, activationRoot: "", limit: historyLimit, json, help };
        }
        return {
            command,
            activationRoot: resolveCliActivationRoot(activationRoot, openclawHome),
            limit: historyLimit,
            json,
            help
        };
    }
    if (command === "export") {
        let outputPath = null;
        for (let index = 0; index < args.length; index += 1) {
            const arg = args[index];
            if (arg === "--help" || arg === "-h") {
                help = true;
                continue;
            }
            if (arg === "--json") {
                json = true;
                continue;
            }
            if (arg === "--activation-root") {
                const next = args[index + 1];
                if (next === undefined)
                    throw new Error("--activation-root requires a value");
                activationRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--openclaw-home") {
                const next = args[index + 1];
                if (next === undefined)
                    throw new Error("--openclaw-home requires a value");
                openclawHome = next;
                index += 1;
                continue;
            }
            if (arg === "-o" || arg === "--output") {
                const next = args[index + 1];
                if (next === undefined)
                    throw new Error("-o / --output requires a value");
                outputPath = next;
                index += 1;
                continue;
            }
            if (arg.startsWith("--"))
                throw new Error(`unknown argument for export: ${arg}`);
        }
        if (help)
            return { command, activationRoot: "", outputPath: "", json, help };
        if (outputPath === null)
            throw new Error("export requires -o <output.tar.gz>");
        return {
            command,
            activationRoot: resolveCliActivationRoot(activationRoot, openclawHome),
            outputPath: path.resolve(outputPath),
            json,
            help,
        };
    }
    if (command === "import") {
        let archivePath = null;
        let force = false;
        for (let index = 0; index < args.length; index += 1) {
            const arg = args[index];
            if (arg === "--help" || arg === "-h") {
                help = true;
                continue;
            }
            if (arg === "--json") {
                json = true;
                continue;
            }
            if (arg === "--force") {
                force = true;
                continue;
            }
            if (arg === "--activation-root") {
                const next = args[index + 1];
                if (next === undefined)
                    throw new Error("--activation-root requires a value");
                activationRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--openclaw-home") {
                const next = args[index + 1];
                if (next === undefined)
                    throw new Error("--openclaw-home requires a value");
                openclawHome = next;
                index += 1;
                continue;
            }
            if (arg.startsWith("--"))
                throw new Error(`unknown argument for import: ${arg}`);
            if (archivePath === null) {
                archivePath = arg;
            }
            else {
                throw new Error(`unexpected positional argument: ${arg}`);
            }
        }
        if (help)
            return { command, archivePath: "", activationRoot: "", force: false, json, help };
        if (archivePath === null)
            throw new Error("import requires <backup.tar.gz> argument");
        return {
            command,
            archivePath: path.resolve(archivePath),
            activationRoot: resolveCliActivationRoot(activationRoot, openclawHome),
            force,
            json,
            help,
        };
    }
    if (command === "reset") {
        let yes = false;
        for (let index = 0; index < args.length; index += 1) {
            const arg = args[index];
            if (arg === "--help" || arg === "-h") {
                help = true;
                continue;
            }
            if (arg === "--json") {
                json = true;
                continue;
            }
            if (arg === "--yes" || arg === "-y") {
                yes = true;
                continue;
            }
            if (arg === "--activation-root") {
                const next = args[index + 1];
                if (next === undefined)
                    throw new Error("--activation-root requires a value");
                activationRoot = next;
                index += 1;
                continue;
            }
            if (arg === "--openclaw-home") {
                const next = args[index + 1];
                if (next === undefined)
                    throw new Error("--openclaw-home requires a value");
                openclawHome = next;
                index += 1;
                continue;
            }
            throw new Error(`unknown argument for reset: ${arg}`);
        }
        if (help)
            return { command, activationRoot: "", yes: false, json, help };
        return {
            command,
            activationRoot: resolveCliActivationRoot(activationRoot, openclawHome),
            yes,
            json,
            help
        };
    }
    for (let index = 0; index < args.length; index += 1) {
        const arg = args[index];
        if (arg === "--help" || arg === "-h") {
            help = true;
            continue;
        }
        if (arg === "--json") {
            json = true;
            continue;
        }
        if (arg === "--dry-run") {
            dryRun = true;
            continue;
        }
        if (arg === "--shared") {
            shared = true;
            continue;
        }
        if (arg === "--skip-embedder-provision") {
            skipEmbedderProvision = true;
            continue;
        }
        if (arg === "--keep-data") {
            keepData = true;
            continue;
        }
        if (arg === "--purge-data") {
            purgeData = true;
            continue;
        }
        if (arg === "--restart") {
            const next = args[index + 1];
            if (next === undefined) {
                throw new Error("--restart requires a value");
            }
            if (next !== "never" && next !== "safe" && next !== "external") {
                throw new Error(`invalid --restart value: ${next}`);
            }
            restart = next;
            restartExplicitlySet = true;
            index += 1;
            continue;
        }
        if (arg === "--detailed") {
            detailed = true;
            continue;
        }
        const next = args[index + 1];
        if (arg === "--openclaw-home") {
            if (next === undefined) {
                throw new Error("--openclaw-home requires a value");
            }
            openclawHome = next;
            index += 1;
            continue;
        }
        if (arg === "--activation-root") {
            if (next === undefined) {
                throw new Error("--activation-root requires a value");
            }
            activationRoot = next;
            index += 1;
            continue;
        }
        if (arg === "--event-export") {
            if (next === undefined) {
                throw new Error("--event-export requires a value");
            }
            eventExportPath = next;
            index += 1;
            continue;
        }
        if (arg === "--teacher-snapshot") {
            if (next === undefined) {
                throw new Error("--teacher-snapshot requires a value");
            }
            teacherSnapshotPath = next;
            index += 1;
            continue;
        }
        if (arg === "--updated-at") {
            if (next === undefined) {
                throw new Error("--updated-at requires a value");
            }
            updatedAt = next;
            index += 1;
            continue;
        }
        if (arg === "--brain-attachment-policy") {
            if (next === undefined) {
                throw new Error("--brain-attachment-policy requires a value");
            }
            if (next !== "undeclared" && next !== "dedicated" && next !== "shared") {
                throw new Error(`invalid --brain-attachment-policy value: ${next}`);
            }
            brainAttachmentPolicy = next;
            index += 1;
            continue;
        }
        if (arg === "--session") {
            if (next === undefined) {
                throw new Error("--session requires a value");
            }
            sessionPath = next;
            index += 1;
            continue;
        }
        if (arg === "--live") {
            if (next === undefined) {
                throw new Error("--live requires a value");
            }
            livePath = next;
            index += 1;
            continue;
        }
        if (arg === "--root") {
            if (next === undefined) {
                throw new Error("--root requires a value");
            }
            rootDir = next;
            index += 1;
            continue;
        }
        if (arg === "--workspace") {
            if (next === undefined) {
                throw new Error("--workspace requires a value");
            }
            workspacePath = next;
            index += 1;
            continue;
        }
        if (arg === "--pack-label") {
            if (next === undefined) {
                throw new Error("--pack-label requires a value");
            }
            packLabel = next;
            index += 1;
            continue;
        }
        if (arg === "--observed-at") {
            if (next === undefined) {
                throw new Error("--observed-at requires a value");
            }
            observedAt = next;
            index += 1;
            continue;
        }
        if (arg === "--snapshot-out") {
            if (next === undefined) {
                throw new Error("--snapshot-out requires a value");
            }
            snapshotOutPath = next;
            index += 1;
            continue;
        }
        if (arg === "--workspace-id") {
            if (next === undefined) {
                throw new Error("--workspace-id requires a value");
            }
            workspaceId = next;
            index += 1;
            continue;
        }
        throw new Error(`unknown argument: ${arg}`);
    }
    if (command !== "detach" && command !== "uninstall" && restartExplicitlySet) {
        throw new Error("--restart only applies to detach/uninstall");
    }
    if (command !== "install" && command !== "attach" && shared) {
        throw new Error("--shared only applies to install/attach");
    }
    if (command !== "install" && command !== "attach" && skipEmbedderProvision) {
        throw new Error("--skip-embedder-provision only applies to install/attach");
    }
    if (command !== "uninstall" && keepData) {
        throw new Error("--keep-data only applies to uninstall; use detach to preserve activation data");
    }
    if (command !== "uninstall" && purgeData) {
        throw new Error("--purge-data only applies to uninstall");
    }
    if (command !== "install" && command !== "attach" && workspaceId !== null) {
        throw new Error("--workspace-id only applies to install/attach");
    }
    if (command !== "scan" && packLabel !== null) {
        throw new Error("--pack-label only applies to scan --live");
    }
    if ((command === "install" || command === "attach") && brainAttachmentPolicy !== null) {
        throw new Error(`${command} uses dedicated by default or --shared for shared mode; --brain-attachment-policy only applies to status/rollback inspection`);
    }
    if (command === "install") {
        if (help) {
            return {
                command,
                openclawHome: "",
                openclawHomeSource: "explicit",
                activationRoot: "",
                activationRootSource: "explicit",
                shared: false,
                workspaceId: "",
                workspaceIdSource: "explicit",
                skipEmbedderProvision: false,
                skipEmbedderProvisionSource: null,
                json,
                help
            };
        }
        const resolvedOpenclawHome = resolveInstallOpenClawHome(openclawHome);
        const resolvedActivationRoot = resolveInstallActivationRoot(resolvedOpenclawHome.openclawHome, activationRoot);
        const resolvedWorkspaceId = resolveInstallWorkspaceId(resolvedOpenclawHome.openclawHome, workspaceId);
        const resolvedEmbedderProvisionSkip = resolveInstallEmbedderProvisionSkip(skipEmbedderProvision);
        return {
            command,
            openclawHome: resolvedOpenclawHome.openclawHome,
            openclawHomeSource: resolvedOpenclawHome.openclawHomeSource,
            activationRoot: resolvedActivationRoot.activationRoot,
            activationRootSource: resolvedActivationRoot.source,
            shared,
            workspaceId: resolvedWorkspaceId.workspaceId,
            workspaceIdSource: resolvedWorkspaceId.source,
            skipEmbedderProvision: resolvedEmbedderProvisionSkip.skipEmbedderProvision,
            skipEmbedderProvisionSource: resolvedEmbedderProvisionSkip.skipEmbedderProvisionSource,
            json,
            help
        };
    }
    if (command === "attach") {
        if (help) {
            return {
                command,
                openclawHome: "",
                openclawHomeSource: "explicit",
                activationRoot: "",
                activationRootSource: "explicit",
                shared: false,
                workspaceId: "",
                workspaceIdSource: "explicit",
                skipEmbedderProvision: false,
                skipEmbedderProvisionSource: null,
                json,
                help
            };
        }
        if (openclawHome === null || openclawHome.trim().length === 0) {
            throw new Error("--openclaw-home is required for attach; use install for the first-time default path");
        }
        const resolvedOpenclawHome = path.resolve(openclawHome);
        const resolvedActivationRoot = resolveInstallActivationRoot(resolvedOpenclawHome, activationRoot);
        const resolvedWorkspaceId = resolveInstallWorkspaceId(resolvedOpenclawHome, workspaceId);
        const resolvedEmbedderProvisionSkip = resolveInstallEmbedderProvisionSkip(skipEmbedderProvision);
        return {
            command,
            openclawHome: resolvedOpenclawHome,
            openclawHomeSource: "explicit",
            activationRoot: resolvedActivationRoot.activationRoot,
            activationRootSource: resolvedActivationRoot.source,
            shared,
            workspaceId: resolvedWorkspaceId.workspaceId,
            workspaceIdSource: resolvedWorkspaceId.source,
            skipEmbedderProvision: resolvedEmbedderProvisionSkip.skipEmbedderProvision,
            skipEmbedderProvisionSource: resolvedEmbedderProvisionSkip.skipEmbedderProvisionSource,
            json,
            help
        };
    }
    if (command === "detach") {
        if (help) {
            return { command, openclawHome: "", activationRoot: null, restart: "safe", json, help };
        }
        if (openclawHome === null || openclawHome.trim().length === 0) {
            throw new Error("--openclaw-home is required for detach");
        }
        if (purgeData) {
            throw new Error("detach always preserves activation data; use uninstall --purge-data to remove it");
        }
        const resolvedOpenclawHome = path.resolve(openclawHome);
        const resolvedActivationRoot = resolveActivationRoot({
            explicit: activationRoot,
            openclawHome: resolvedOpenclawHome,
            quiet: true
        });
        return {
            command,
            openclawHome: resolvedOpenclawHome,
            activationRoot: resolvedActivationRoot.trim().length === 0 ? null : path.resolve(resolvedActivationRoot),
            restart,
            json,
            help
        };
    }
    if (command === "uninstall") {
        if (help) {
            return { command, openclawHome: "", activationRoot: null, dataMode: "keep", restart: "safe", json, help };
        }
        if (openclawHome === null || openclawHome.trim().length === 0) {
            throw new Error("--openclaw-home is required for uninstall");
        }
        if (!keepData && !purgeData) {
            throw new Error("uninstall requires exactly one of --keep-data or --purge-data");
        }
        if (keepData && purgeData) {
            throw new Error("--keep-data and --purge-data are mutually exclusive");
        }
        const resolvedOpenclawHome = path.resolve(openclawHome);
        const resolvedActivationRoot = resolveActivationRoot({
            explicit: activationRoot,
            openclawHome: resolvedOpenclawHome,
            quiet: true
        });
        if (purgeData && resolvedActivationRoot.trim().length === 0) {
            throw new Error("--purge-data requires a resolvable activation root from the installed profile hook or --activation-root <path>");
        }
        return {
            command,
            openclawHome: resolvedOpenclawHome,
            activationRoot: resolvedActivationRoot.trim().length === 0 ? null : path.resolve(resolvedActivationRoot),
            dataMode: purgeData ? "purge" : "keep",
            restart,
            json,
            help
        };
    }
    if (command === "scan") {
        if ((sessionPath === null && livePath === null) || (sessionPath !== null && livePath !== null)) {
            throw new Error("scan requires exactly one of --session or --live");
        }
        if (sessionPath !== null) {
            if (rootDir === null) {
                throw new Error("--root is required for scan --session");
            }
            if (workspacePath !== null || packLabel !== null || observedAt !== null || snapshotOutPath !== null) {
                throw new Error("--workspace, --pack-label, --observed-at, and --snapshot-out only apply to scan --live");
            }
        }
        if (livePath !== null) {
            if (workspacePath === null) {
                throw new Error("--workspace is required for scan --live");
            }
            if (rootDir !== null) {
                throw new Error("--root only applies to scan --session");
            }
        }
        return {
            command,
            json,
            help,
            sessionPath,
            livePath,
            rootDir,
            workspacePath,
            packLabel,
            observedAt,
            snapshotOutPath
        };
    }
    return {
        command: command,
        input: {
            activationRoot: activationRoot ?? "",
            eventExportPath,
            teacherSnapshotPath,
            updatedAt,
            brainAttachmentPolicy
        },
        openclawHome: normalizeOptionalCliString(openclawHome),
        json,
        help,
        dryRun,
        detailed
    };
}
function isDirectCliRun(entryArg, moduleUrl) {
    if (entryArg === undefined) {
        return false;
    }
    try {
        return pathToFileURL(realpathSync(entryArg)).href === moduleUrl;
    }
    catch {
        return pathToFileURL(path.resolve(entryArg)).href === moduleUrl;
    }
}
/**
 * Resolve the path to the pre-built extension template shipped with this package.
 * Falls back to a generated string if the template file is missing (e.g. in tests).
 */
function resolveExtensionTemplatePath() {
    const candidates = [
        path.resolve(__dirname, "..", "extension", "index.ts"),
        path.resolve(__dirname, "..", "..", "extension", "index.ts"),
    ];
    for (const candidate of candidates) {
        if (existsSync(candidate)) {
            return candidate;
        }
    }
    throw new Error("Pre-built extension template not found. Searched:\n" +
        candidates.map((c) => `  - ${c}`).join("\n"));
}
function resolveExtensionRuntimeGuardPath() {
    const tsCandidates = [
        path.resolve(__dirname, "..", "extension", "runtime-guard.ts"),
        path.resolve(__dirname, "..", "..", "extension", "runtime-guard.ts"),
    ];
    const jsCandidates = [
        path.resolve(__dirname, "..", "dist", "extension", "runtime-guard.js"),
        path.resolve(__dirname, "extension", "runtime-guard.js"),
        path.resolve(__dirname, "..", "extension", "runtime-guard.js"),
    ];
    const tsPath = tsCandidates.find((c) => existsSync(c)) ?? null;
    const jsPath = jsCandidates.find((c) => existsSync(c));
    if (!jsPath) {
        throw new Error("Pre-built extension runtime-guard.js not found. Searched:\n" +
            jsCandidates.map((c) => `  - ${c}`).join("\n"));
    }
    return { ts: tsPath, js: jsPath };
}
const LOCAL_WORKSPACE_EXTENSION_PACKAGES = [
    "activation",
    "compiler",
    "contracts",
    "event-export",
    "events",
    "learner",
    "openclaw",
    "pack-format",
    "provenance",
    "workspace-metadata"
];
const OPENCLAWBRAIN_EXTENSION_TARBALL_DIR_ENV = "OPENCLAWBRAIN_EXTENSION_TARBALL_DIR";
function resolveNpmCommand() {
    return process.platform === "win32" ? "npm.cmd" : "npm";
}
function resolveExtensionInstallReleaseTarballs() {
    const configuredDir = normalizeOptionalCliString(process.env[OPENCLAWBRAIN_EXTENSION_TARBALL_DIR_ENV]);
    if (configuredDir === null) {
        return null;
    }
    const artifactDir = path.resolve(configuredDir);
    let entries;
    try {
        entries = readdirSync(artifactDir, { withFileTypes: true });
    }
    catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(`${OPENCLAWBRAIN_EXTENSION_TARBALL_DIR_ENV} is unreadable: ${artifactDir} (${detail})`);
    }
    const tarballs = entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".tgz"))
        .map((entry) => path.join(artifactDir, entry.name))
        .sort((left, right) => left.localeCompare(right));
    if (tarballs.length === 0) {
        throw new Error(`${OPENCLAWBRAIN_EXTENSION_TARBALL_DIR_ENV} has no .tgz release artifacts: ${artifactDir}`);
    }
    return {
        artifactDir,
        tarballs
    };
}
function resolveLocalWorkspaceRootForExtensionInstall() {
    const candidates = [
        path.resolve(__dirname, "..", "..", "..", ".."),
        path.resolve(__dirname, "..", "..", "..")
    ];
    for (const candidate of candidates) {
        const packageRoot = path.join(candidate, "packages", "openclaw");
        const distEntry = path.join(packageRoot, "dist", "src", "index.js");
        if (existsSync(packageRoot) && existsSync(distEntry)) {
            return candidate;
        }
    }
    return null;
}
function installExtensionFromLocalWorkspaceBuild(extensionDir) {
    const workspaceRoot = resolveLocalWorkspaceRootForExtensionInstall();
    if (workspaceRoot === null) {
        return null;
    }
    const nodeModulesRoot = path.join(extensionDir, "node_modules", "@openclawbrain");
    mkdirSync(nodeModulesRoot, { recursive: true });
    for (const packageName of LOCAL_WORKSPACE_EXTENSION_PACKAGES) {
        const packageDir = path.join(workspaceRoot, "packages", packageName);
        const packageDistEntry = path.join(packageDir, "dist", "src", "index.js");
        if (!existsSync(packageDir) || !existsSync(packageDistEntry)) {
            return null;
        }
    }
    for (const packageName of LOCAL_WORKSPACE_EXTENSION_PACKAGES) {
        const packageDir = path.join(workspaceRoot, "packages", packageName);
        const linkPath = path.join(nodeModulesRoot, packageName);
        rmSync(linkPath, { recursive: true, force: true });
        symlinkSync(packageDir, linkPath, "dir");
    }
    return [...LOCAL_WORKSPACE_EXTENSION_PACKAGES];
}
let cachedOpenClawPackageMetadata = null;
function resolveOpenClawPackageManifestPath() {
    const candidates = [
        path.resolve(__dirname, "..", "package.json"),
        path.resolve(__dirname, "..", "..", "package.json"),
    ];
    for (const candidate of candidates) {
        if (existsSync(candidate)) {
            return candidate;
        }
    }
    throw new Error("OpenClawBrain package manifest not found. Searched:\n" +
        candidates.map((candidate) => `  - ${candidate}`).join("\n"));
}
function readOpenClawPackageMetadata() {
    if (cachedOpenClawPackageMetadata !== null) {
        return cachedOpenClawPackageMetadata;
    }
    const manifestPath = resolveOpenClawPackageManifestPath();
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    const name = typeof manifest.name === "string" && manifest.name.trim().length > 0
        ? manifest.name.trim()
        : "@openclawbrain/openclaw";
    const version = typeof manifest.version === "string" && manifest.version.trim().length > 0
        ? manifest.version.trim()
        : "0.0.0";
    cachedOpenClawPackageMetadata = { name, version };
    return cachedOpenClawPackageMetadata;
}
function buildExtensionIndexTs(activationRoot) {
    const templatePath = resolveExtensionTemplatePath();
    const template = readFileSync(templatePath, "utf8");
    return template.replace(/const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";/, `const ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};`);
}
function buildExtensionPackageJson() {
    const packageMetadata = readOpenClawPackageMetadata();
    return JSON.stringify({
        name: "openclawbrain",
        version: packageMetadata.version,
        private: true,
        type: "module",
        openclaw: {
            extensions: ["index.ts"]
        },
        dependencies: {
            [packageMetadata.name]: packageMetadata.version
        }
    }, null, 2) + "\n";
}
function buildExtensionPluginManifest() {
    const packageMetadata = readOpenClawPackageMetadata();
    return JSON.stringify({
        id: "openclawbrain",
        name: "OpenClawBrain",
        description: "Learned memory and context from OpenClawBrain",
        version: packageMetadata.version,
        configSchema: {
            type: "object",
            additionalProperties: false,
            properties: {}
        }
    }, null, 2) + "\n";
}
function formatContextForHuman(result) {
    if (!result.ok) {
        if (result.fallbackToStaticContext) {
            return "No learned context yet. Talk to your agent and check back.";
        }
        return `Brain error: ${result.error}`;
    }
    if (result.brainContext.trim().length === 0) {
        return "No learned context yet. Talk to your agent and check back.";
    }
    return result.brainContext;
}
function runContextCommand(parsed) {
    const result = compileRuntimeContext({
        activationRoot: parsed.activationRoot,
        message: parsed.message
    });
    if (parsed.json) {
        console.log(JSON.stringify({
            ok: result.ok,
            activationRoot: result.activationRoot,
            activePackId: result.ok ? result.activePackId : null,
            brainContext: result.brainContext,
            fallbackToStaticContext: result.ok ? false : result.fallbackToStaticContext,
            hardRequirementViolated: result.ok ? false : result.hardRequirementViolated,
            error: result.ok ? null : result.error
        }, null, 2));
    }
    else {
        console.log(formatContextForHuman(result));
    }
    return 0;
}
function formatHistoryTimestamp(iso) {
    const date = new Date(iso);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${year}-${month}-${day} ${hours}:${minutes}`;
}
function loadManifestSafe(manifestPath) {
    try {
        if (!existsSync(manifestPath)) {
            return null;
        }
        return JSON.parse(readFileSync(manifestPath, "utf8"));
    }
    catch {
        return null;
    }
}
function buildHistoryEntry(record, slot, isActive) {
    const manifest = loadManifestSafe(record.manifestPath);
    const eventCount = record.eventRange.count;
    // Count corrections from the learning surface in the manifest provenance
    let correctionCount = 0;
    if (manifest !== null) {
        const learningSurface = manifest.provenance?.learningSurface;
        if (learningSurface?.labelHarvest) {
            correctionCount = learningSurface.labelHarvest.humanLabels;
        }
    }
    // Determine the label: seed packs have 0 events, promoted packs have events
    const label = eventCount === 0 ? "seed" : "promoted";
    return {
        packId: record.packId,
        slot,
        label,
        builtAt: record.builtAt,
        updatedAt: record.updatedAt,
        eventCount,
        correctionCount,
        current: isActive
    };
}
function ensureLifecycleLearnerService(activationRoot) {
    const outcome = ensureManagedLearnerServiceForActivationRoot(activationRoot);
    return {
        state: outcome.state,
        detail: outcome.detail,
        plistPath: outcome.inspection.plistPath,
        logPath: outcome.inspection.logPath,
        configuredActivationRoot: outcome.inspection.configuredActivationRoot,
        matchesRequestedActivationRoot: outcome.inspection.matchesRequestedActivationRoot
    };
}
function resolveCleanupLearnerServiceOutcome(activationRoot, openclawHome) {
    if (activationRoot === null) {
        return {
            state: "unresolved",
            detail: "Learner service preservation is unresolved because the activation root could not be resolved from the installed profile hook.",
            plistPath: null,
            logPath: null,
            configuredActivationRoot: null,
            matchesRequestedActivationRoot: null
        };
    }
    const remainingProfiles = findOtherInstalledHookReferencesForActivationRoot({
        activationRoot,
        excludingOpenClawHome: openclawHome
    });
    const partitioned = partitionSharedActivationRootHookReferences(remainingProfiles);
    if (partitioned.attached.length > 0) {
        const inspection = inspectManagedLearnerService(activationRoot);
        const attachedProfiles = partitioned.attached
            .map(({ openclawHome: profileHome }) => shortenPath(path.resolve(profileHome)))
            .join(", ");
        const halfAttachedNote = partitioned.halfAttached.length === 0
            ? ""
            : ` Half-attached hooks still point at this root but were not counted as attached: ${partitioned.halfAttached
                .map((reference) => `${shortenPath(path.resolve(reference.openclawHome))} (${summarizeSharedActivationRootReferenceProof(reference)})`)
                .join(", ")}.`;
        return {
            state: "preserved",
            detail: `Preserved the background learner service for ${path.resolve(activationRoot)} because other attached OpenClaw profiles still share this activation root: ${attachedProfiles}.${halfAttachedNote}`,
            plistPath: inspection.plistPath,
            logPath: inspection.logPath,
            configuredActivationRoot: inspection.configuredActivationRoot,
            matchesRequestedActivationRoot: inspection.matchesRequestedActivationRoot
        };
    }
    const outcome = removeManagedLearnerServiceForActivationRoot(activationRoot);
    const halfAttachedNote = partitioned.halfAttached.length === 0
        ? ""
        : ` Ignored half-attached OpenClaw profile hooks that still point at this activation root because they do not prove serve-path attachment: ${partitioned.halfAttached
            .map((reference) => `${shortenPath(path.resolve(reference.openclawHome))} (${summarizeSharedActivationRootReferenceProof(reference)})`)
            .join(", ")}.`;
    return {
        state: outcome.state,
        detail: `${outcome.detail}${halfAttachedNote}`,
        plistPath: outcome.inspection.plistPath,
        logPath: outcome.inspection.logPath,
        configuredActivationRoot: outcome.inspection.configuredActivationRoot,
        matchesRequestedActivationRoot: outcome.inspection.matchesRequestedActivationRoot
    };
}
function formatInspectionFindings(findings) {
    return findings.join("; ");
}
function buildInstallRefusalError(parsed, detail) {
    const purgeCommand = `openclawbrain uninstall --openclaw-home ${quoteShellArg(parsed.openclawHome)} ` +
        `--activation-root ${quoteShellArg(parsed.activationRoot)} --purge-data`;
    return new Error(`Refusing to reuse activation root ${path.resolve(parsed.activationRoot)}: ${detail}. ` +
        "Install only repairs an empty first-state root; it will not overwrite populated or broken activation state. " +
        `Inspect: ${buildInstallStatusCommand(parsed.activationRoot)}. ` +
        `Reset: ${purgeCommand}.`);
}
function inspectInstallActivationPlan(parsed) {
    const resolvedActivationRoot = path.resolve(parsed.activationRoot);
    const activationPointersPath = path.join(resolvedActivationRoot, "activation-pointers.json");
    if (!existsSync(resolvedActivationRoot)) {
        return {
            createActivationRoot: true,
            action: "bootstrap",
            resolution: "new_root",
            inspectionStep: "Activation state inspection: activation root is missing; bootstrapping first state.",
            activePackId: null
        };
    }
    const activationRootStats = statSync(resolvedActivationRoot);
    if (!activationRootStats.isDirectory()) {
        throw buildInstallRefusalError(parsed, "activation root path exists but is not a directory");
    }
    if (!existsSync(activationPointersPath)) {
        return {
            createActivationRoot: false,
            action: "bootstrap",
            resolution: "missing_pointers",
            inspectionStep: "Activation state inspection: activation root exists but activation-pointers.json is missing; bootstrapping first state.",
            activePackId: null
        };
    }
    let inspection;
    try {
        inspection = inspectActivationState(resolvedActivationRoot, new Date().toISOString());
    }
    catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw buildInstallRefusalError(parsed, `activation pointers could not be inspected (${detail})`);
    }
    if (inspection.active === null && inspection.candidate === null && inspection.previous === null) {
        return {
            createActivationRoot: false,
            action: "bootstrap",
            resolution: "empty_pointers",
            inspectionStep: "Activation state inspection: activation pointers are present but all slots are empty; bootstrapping first state.",
            activePackId: null
        };
    }
    const unhealthySlots = [inspection.active, inspection.candidate, inspection.previous]
        .filter((slot) => slot !== null && !slot.activationReady)
        .map((slot) => `${slot.slot}: ${formatInspectionFindings(slot.findings)}`);
    if (unhealthySlots.length > 0) {
        throw buildInstallRefusalError(parsed, `activation state contains unhealthy slots (${unhealthySlots.join(" | ")})`);
    }
    if (inspection.active === null) {
        const populatedSlots = [inspection.candidate, inspection.previous]
            .filter((slot) => slot !== null)
            .map((slot) => slot.slot);
        throw buildInstallRefusalError(parsed, `activation state is populated without an active pack (${populatedSlots.join(", ")})`);
    }
    if (inspection.candidate !== null && !inspection.promotion.allowed) {
        throw buildInstallRefusalError(parsed, `candidate slot is stale or incoherent (${formatInspectionFindings(inspection.promotion.findings)})`);
    }
    if (inspection.previous !== null && !inspection.rollback.allowed) {
        throw buildInstallRefusalError(parsed, `previous slot is stale or incoherent (${formatInspectionFindings(inspection.rollback.findings)})`);
    }
    return {
        createActivationRoot: false,
        action: "keep",
        resolution: "healthy_existing",
        inspectionStep: `Activation state inspection: active pack ${inspection.active.packId} is healthy; keeping existing activation state.`,
        activePackId: inspection.active.packId
    };
}
function runHistoryCommand(parsed) {
    const activationRoot = parsed.activationRoot;
    const pointersPath = path.join(activationRoot, "activation-pointers.json");
    if (!existsSync(pointersPath)) {
        if (parsed.json) {
            console.log(JSON.stringify({ entries: [], empty: true, message: "No history yet. Run: openclawbrain install" }, null, 2));
        }
        else {
            console.log("No history yet. Run: openclawbrain install");
        }
        return 0;
    }
    let pointers;
    try {
        pointers = JSON.parse(readFileSync(pointersPath, "utf8"));
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`Failed to read activation pointers: ${message}`);
        return 1;
    }
    // Build history entries from pointers: active is most recent, then previous
    const entries = [];
    if (pointers.active !== null) {
        entries.push(buildHistoryEntry(pointers.active, "active", true));
    }
    if (pointers.previous !== null) {
        // Only add if different from active
        if (pointers.active === null || pointers.previous.packId !== pointers.active.packId) {
            entries.push(buildHistoryEntry(pointers.previous, "previous", false));
        }
    }
    if (pointers.candidate !== null) {
        // Only add if different from active and previous
        const isDuplicate = entries.some((e) => e.packId === pointers.candidate.packId);
        if (!isDuplicate) {
            entries.push(buildHistoryEntry(pointers.candidate, "candidate", false));
        }
    }
    if (entries.length === 0) {
        if (parsed.json) {
            console.log(JSON.stringify({ entries: [], empty: true, message: "No history yet. Run: openclawbrain install" }, null, 2));
        }
        else {
            console.log("No history yet. Run: openclawbrain install");
        }
        return 0;
    }
    // Sort by updatedAt descending (most recent first)
    entries.sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
    // Apply limit
    const limited = entries.slice(0, parsed.limit);
    if (parsed.json) {
        console.log(JSON.stringify({
            entries: limited,
            activationRoot,
            empty: false
        }, null, 2));
        return 0;
    }
    // Human-readable output
    for (const entry of limited) {
        const packShort = entry.packId.length > 9 ? entry.packId.slice(0, 9) : entry.packId;
        const timestamp = formatHistoryTimestamp(entry.updatedAt);
        const tag = entry.current ? "(current)" : "(previous)";
        let line = `${packShort.padEnd(10)} ${entry.label.padEnd(10)} ${timestamp}  ${tag}`;
        // Add stats suffix for promoted packs
        if (entry.label === "promoted" && (entry.correctionCount > 0 || entry.eventCount > 0)) {
            const parts = [];
            if (entry.correctionCount > 0) {
                parts.push(`${entry.correctionCount} corrections`);
            }
            if (entry.eventCount > 0) {
                parts.push(`${entry.eventCount} events`);
            }
            line += ` — ${parts.join(", ")}`;
        }
        console.log(line);
    }
    return 0;
}
function runProfileHookAttachCommand(parsed) {
    const steps = [];
    const commandLabel = parsed.command.toUpperCase();
    const isInstall = parsed.command === "install";
    const targetInspection = inspectOpenClawHome(parsed.openclawHome);
    const installTarget = resolveOpenClawBrainInstallTarget(parsed.openclawHome);
    const extensionDir = installTarget.extensionDir;
    steps.push(`Target OpenClaw home: ${parsed.openclawHome} (${formatInstallOpenClawHomeSource(parsed.openclawHomeSource)})`);
    steps.push(isInstall
        ? "Lifecycle mode: install is the safe first-time default for wiring one profile to one activation root."
        : "Lifecycle mode: attach is the explicit reattach/manual profile-hook path; use install for first-time setup.");
    steps.push(`Detected layout: ${formatOpenClawTargetExplanation(targetInspection)}`);
    steps.push(`Target hook path: ${installTarget.hookPath} (${describeOpenClawBrainInstallLayout(installTarget.installLayout)})`);
    if (installTarget.selectedInstall !== null && installTarget.additionalInstalls.length > 0) {
        steps.push(`Kept ${describeOpenClawBrainInstallLayout(installTarget.selectedInstall.installLayout)} authoritative at ${shortenPath(installTarget.selectedInstall.extensionDir)} (${describeOpenClawBrainInstallIdentity(installTarget.selectedInstall)}); additional installs remain at ${installTarget.additionalInstalls.map((install) => `${shortenPath(install.extensionDir)} (${describeOpenClawBrainInstallLayout(install.installLayout)})`).join(", ")}`);
    }
    // 1. Validate --openclaw-home exists and has openclaw.json
    validateOpenClawHome(parsed.openclawHome);
    // 2. Inspect the activation root before writing profile hook artifacts.
    const activationPlan = inspectInstallActivationPlan(parsed);
    // 3. Ensure the default embedder exists before bootstrap unless the operator explicitly opts out.
    const embedderProvision = activationPlan.action === "bootstrap"
        ? ensureInstallEmbedderReady(parsed)
        : null;
    if (embedderProvision === null) {
        steps.push("Skipped bootstrap-time embedder provisioning because attach/install is reusing healthy activation state.");
    }
    else {
        steps.push(embedderProvision.detail);
    }
    // 4. Create activation root if needed
    if (activationPlan.createActivationRoot) {
        mkdirSync(parsed.activationRoot, { recursive: true });
        steps.push(`Created activation root: ${parsed.activationRoot}`);
    }
    else {
        steps.push(`Activation root exists: ${parsed.activationRoot}`);
    }
    steps.push(activationPlan.inspectionStep);
    // 5. Persist install-written local provider defaults so watch/learning surfaces do not depend on gateway env wiring.
    const providerDefaults = shouldWriteProfileHookProviderDefaults(parsed, activationPlan, isInstall)
        ? writeInstallProviderDefaults(parsed)
        : null;
    if (providerDefaults === null) {
        steps.push("Preserved existing provider-defaults.json because explicit attach is reusing existing activation data.");
    }
    else {
        steps.push(providerDefaults.detail);
    }
    // 6. Bootstrap only for safe empty first-state roots; otherwise keep the inspected healthy state.
    if (activationPlan.action === "bootstrap") {
        const packRoot = path.resolve(parsed.activationRoot, "packs", "initial");
        mkdirSync(packRoot, { recursive: true });
        const brainAttachmentPolicy = parsed.shared ? "shared" : "dedicated";
        const result = bootstrapRuntimeAttach({
            profileSelector: "current_profile",
            brainAttachmentPolicy,
            activationRoot: parsed.activationRoot,
            packRoot,
            packLabel: isInstall ? "install-cli" : "attach-cli",
            workspace: {
                workspaceId: parsed.workspaceId,
                snapshotId: `${parsed.workspaceId}@${parsed.command}-${new Date().toISOString().slice(0, 10)}`,
                capturedAt: new Date().toISOString(),
                rootDir: parsed.openclawHome,
                revision: isInstall ? "cli-install-v1" : "cli-attach-v1"
            },
            interactionEvents: [],
            feedbackEvents: []
        });
        steps.push(`Bootstrapped brain attach: state=${result.currentProfile.brain.state} awaitingFirstExport=${yesNo(result.currentProfile.brainStatus.awaitingFirstExport)}`);
    }
    else {
        steps.push(isInstall
            ? `Kept inspected activation state: active pack ${activationPlan.activePackId}`
            : `Reused inspected activation state for explicit attach: active pack ${activationPlan.activePackId}`);
    }
    // 7-10. Prepare the hook layout that OpenClaw will actually load.
    if (installTarget.writeMode === "pin_native_package") {
        pinInstalledOpenClawBrainPluginActivationRoot(installTarget.hookPath, parsed.activationRoot);
        steps.push(`Pinned native package plugin loader: ${installTarget.hookPath}`);
    }
    else {
        mkdirSync(extensionDir, { recursive: true });
        const indexTsPath = path.join(extensionDir, "index.ts");
        writeFileSync(indexTsPath, buildExtensionIndexTs(parsed.activationRoot), "utf8");
        steps.push(`Wrote extension: ${indexTsPath}`);
        const runtimeGuardPaths = resolveExtensionRuntimeGuardPath();
        if (runtimeGuardPaths.ts !== null) {
            const runtimeGuardTsPath = path.join(extensionDir, "runtime-guard.ts");
            writeFileSync(runtimeGuardTsPath, readFileSync(runtimeGuardPaths.ts, "utf8"), "utf8");
            steps.push(`Wrote extension runtime-guard source: ${runtimeGuardTsPath}`);
        }
        const runtimeGuardJsPath = path.join(extensionDir, "runtime-guard.js");
        writeFileSync(runtimeGuardJsPath, readFileSync(runtimeGuardPaths.js, "utf8"), "utf8");
        steps.push(`Wrote extension runtime-guard: ${runtimeGuardJsPath}`);
        const packageJsonPath = path.join(extensionDir, "package.json");
        writeFileSync(packageJsonPath, buildExtensionPackageJson(), "utf8");
        steps.push(`Wrote package.json: ${packageJsonPath}`);
        const releaseTarballInstall = resolveExtensionInstallReleaseTarballs();
        try {
            if (releaseTarballInstall !== null) {
                execFileSync(resolveNpmCommand(), ["install", "--ignore-scripts", "--no-save", ...releaseTarballInstall.tarballs], { cwd: extensionDir, stdio: "pipe" });
                steps.push(`Installed extension dependencies from release artifacts: ${releaseTarballInstall.tarballs.length} tarballs from ${releaseTarballInstall.artifactDir}`);
            }
            else {
                execSync("npm install --ignore-scripts", { cwd: extensionDir, stdio: "pipe" });
                steps.push("Ran npm install --ignore-scripts");
                const linkedPackages = installExtensionFromLocalWorkspaceBuild(extensionDir);
                if (linkedPackages !== null) {
                    steps.push(`Linked coherent local workspace packages: ${linkedPackages.join(", ")}`);
                }
            }
        }
        catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            if (releaseTarballInstall !== null) {
                throw new Error(`Extension dependency install from release artifacts failed: ${message}`);
            }
            const linkedPackages = installExtensionFromLocalWorkspaceBuild(extensionDir);
            if (linkedPackages !== null) {
                steps.push(`Linked coherent local workspace packages: ${linkedPackages.join(", ")}`);
            }
            else {
                steps.push(`npm install failed (non-fatal): ${message}`);
            }
        }
        const manifestPath = path.join(extensionDir, "openclaw.plugin.json");
        writeFileSync(manifestPath, buildExtensionPluginManifest(), "utf8");
        steps.push(`Wrote manifest: ${manifestPath}`);
    }
    const pluginConfigRepair = ensureOpenClawBrainPluginConfig(parsed.openclawHome);
    steps.push(pluginConfigRepair.detail);
    const learnerService = ensureLifecycleLearnerService(parsed.activationRoot);
    steps.push(learnerService.detail);
    const attachmentPolicyDeclaration = writeAttachmentPolicyDeclaration({
        activationRoot: parsed.activationRoot,
        policy: parsed.shared ? "shared" : "dedicated",
        source: parsed.command,
        openclawHome: parsed.openclawHome
    });
    steps.push(`Recorded attachment policy declaration: ${attachmentPolicyDeclaration.declaration.policy} at ${shortenPath(attachmentPolicyDeclaration.path)}`);
    const brainFeedback = buildInstallBrainFeedbackSummary({
        parsed,
        targetInspection,
        hookPath: installTarget.hookPath,
        hookLayout: installTarget.installLayout,
        activationPlan,
        learnerService,
        embedderProvision,
        providerDefaults
    });
    const restartGuidance = buildInstallReloadGuidance({
        targetInspection
    });
    const nextSteps = [
        restartGuidance,
        brainFeedback.restart.gatewayStatusCommand === null
            ? null
            : `Confirm gateway after restart: ${brainFeedback.restart.gatewayStatusCommand}`,
        `Check status: ${buildInstallStatusCommand(parsed.activationRoot)}`,
        `Check learner service: ${buildLearnerServiceStatusCommand(parsed.activationRoot)}`,
        embedderProvision !== null && embedderProvision.state === "skipped"
            ? `Provision default embedder later: ${buildInstallEmbedderProvisionCommand(embedderProvision.baseUrl, embedderProvision.model)}`
            : null
    ].filter((step) => step !== null);
    const preflightSummary = [
        `Hook: ${describeOpenClawBrainInstallLayout(installTarget.installLayout)} at ${shortenPath(installTarget.hookPath)}`,
        parsed.shared
            ? "Attachment policy: shared activation root declared; same-gateway many-profile load/serve proof is still not checked in"
            : "Attachment policy: dedicated activation root for this profile/home boundary",
        activationPlan.action === "bootstrap"
            ? "Attachment: seed/current-profile attach created; restart plus status will prove later serve-path use"
            : `Attachment: existing active pack ${activationPlan.activePackId} kept in place; restart plus status will prove later serve-path use`,
        embedderProvision === null
            ? "Embedder: unchanged because no bootstrap was needed"
            : embedderProvision.state === "ensured"
                ? `Embedder: default Ollama model ${embedderProvision.model} was ensured before bootstrap`
                : `Embedder: default Ollama model ${embedderProvision.model} was intentionally skipped`,
        `Learner: background service ${learnerService.state} for the exact activation root/profile boundary`,
        `Serve path: install alone does not prove serving; restart the profile and run ${buildInstallStatusCommand(parsed.activationRoot)}`
    ];
    const lifecycleSummary = [
        isInstall
            ? "Lifecycle mode: install (safe first-time/default profile hookup)"
            : "Lifecycle mode: attach (explicit reattach/manual profile hookup)",
        `OpenClaw target: ${shortenPath(parsed.openclawHome)} (${formatInstallOpenClawHomeSource(parsed.openclawHomeSource)})`,
        `Detected layout: ${formatOpenClawTargetExplanation(targetInspection)}`,
        brainFeedback.profile.exactProfileName === null
            ? "Profile token: current_profile only; this install did not infer an exact --profile token"
            : `Profile token: use exact OpenClaw profile casing ${quoteShellArg(brainFeedback.profile.exactProfileName)} for host-side restart/status commands`,
        `Activation root: ${shortenPath(parsed.activationRoot)} (${formatInstallActivationRootSource(parsed.activationRootSource)})`,
        `Attachment policy: ${brainFeedback.attachment.policy} (${brainFeedback.attachment.detail})`,
        `Workspace ID: ${parsed.workspaceId} (${formatInstallWorkspaceIdSource(parsed.workspaceIdSource)})`,
        embedderProvision === null
            ? "Embedder: unchanged because no bootstrap was needed"
            : embedderProvision.state === "ensured"
                ? `Embedder: ensured default Ollama model ${embedderProvision.model} before brain init`
                : `Embedder: skipped default Ollama model ${embedderProvision.model} via ${parsed.skipEmbedderProvisionSource === "flag" ? "--skip-embedder-provision" : OPENCLAWBRAIN_INSTALL_SKIP_EMBEDDER_PROVISION_ENV}`,
        ...(providerDefaults === null ? [] : [`${providerDefaults.lifecycleSummary} (${shortenPath(providerDefaults.path)})`]),
        `Profile hook: ${describeOpenClawBrainInstallLayout(installTarget.installLayout)} at ${shortenPath(installTarget.hookPath)}`,
        `Learner service: ${learnerService.state} for ${shortenPath(parsed.activationRoot)}`,
        activationPlan.resolution === "new_root"
            ? `Activation data: initialized at ${shortenPath(parsed.activationRoot)}`
            : activationPlan.resolution === "missing_pointers"
                ? `Activation data: repaired missing pointers at ${shortenPath(parsed.activationRoot)}`
                : activationPlan.resolution === "empty_pointers"
                    ? `Activation data: repaired empty pointers at ${shortenPath(parsed.activationRoot)}`
                    : `Activation data: reused healthy state at ${shortenPath(parsed.activationRoot)}`,
        activationPlan.action === "bootstrap"
            ? activationPlan.resolution === "new_root"
                ? `${isInstall ? "Install" : "Attach"}: bootstrapped a seed/current-profile brain`
                : activationPlan.resolution === "missing_pointers"
                    ? `${isInstall ? "Install" : "Attach"}: repaired missing activation pointers and bootstrapped a seed/current-profile brain`
                    : `${isInstall ? "Install" : "Attach"}: repaired empty activation pointers and bootstrapped a seed/current-profile brain`
            : isInstall
                ? `Install: kept healthy active pack ${activationPlan.activePackId} in place`
                : `Attach: rewired the profile hook to healthy active pack ${activationPlan.activePackId}`
    ];
    // 9. Print summary
    if (parsed.json) {
        console.log(JSON.stringify({
            command: parsed.command,
            openclawHome: parsed.openclawHome,
            openclawHomeSource: parsed.openclawHomeSource,
            openclawTarget: {
                layout: targetInspection.layout,
                detail: describeOpenClawHomeInspection(targetInspection),
                profileId: targetInspection.profileId,
                profileSource: targetInspection.profileSource,
                configuredProfileIds: targetInspection.configuredProfileIds
            },
            activationRoot: parsed.activationRoot,
            resolvedInputs: {
                activationRoot: {
                    value: parsed.activationRoot,
                    source: parsed.activationRootSource
                },
                workspaceId: {
                    value: parsed.workspaceId,
                    source: parsed.workspaceIdSource
                }
            },
            workspaceId: parsed.workspaceId,
            shared: parsed.shared,
            embedderProvision: embedderProvision === null
                ? null
                : {
                    skipped: parsed.skipEmbedderProvision,
                    source: parsed.skipEmbedderProvisionSource,
                    model: embedderProvision.model,
                    baseUrl: embedderProvision.baseUrl
                },
            providerDefaults: providerDefaults === null
                ? null
                : {
                    path: providerDefaults.path,
                    teacher: providerDefaults.defaults.teacher === undefined
                        ? null
                        : {
                            provider: providerDefaults.defaults.teacher.provider ?? null,
                            model: providerDefaults.defaults.teacher.model ?? null,
                            detectedLocally: providerDefaults.defaults.teacher.detectedLocally ?? false
                        },
                    embedder: providerDefaults.defaults.embedder === undefined
                        ? null
                        : {
                            provider: providerDefaults.defaults.embedder.provider ?? null,
                            model: providerDefaults.defaults.embedder.model ?? null
                        },
                    teacherBaseUrl: providerDefaults.defaults.teacherBaseUrl ?? null,
                    embedderBaseUrl: providerDefaults.defaults.embedderBaseUrl ?? null
                },
            pluginConfigRepair,
            learnerService,
            brainFeedback: {
                hookPath: brainFeedback.hookPath,
                hookLayout: brainFeedback.hookLayout,
                providerDefaultsPath: brainFeedback.providerDefaultsPath,
                profile: brainFeedback.profile,
                attachment: brainFeedback.attachment,
                restart: brainFeedback.restart,
                embedder: brainFeedback.embedder,
                teacher: brainFeedback.teacher,
                learnerService: brainFeedback.learnerService,
                startup: brainFeedback.startup,
                provedNow: brainFeedback.provedNow,
                notYetProved: brainFeedback.notYetProved,
                lines: brainFeedback.lines
            },
            extensionDir,
            lifecycleSummary,
            preflightSummary,
            restartGuidance,
            nextSteps,
            steps
        }, null, 2));
    }
    else {
        console.log(`${commandLabel} complete\n`);
        console.log("Brain feedback:");
        for (const line of brainFeedback.lines) {
            console.log(`  ${line}`);
        }
        console.log(`Restart:    ${restartGuidance}`);
        if (brainFeedback.restart.gatewayStatusCommand !== null) {
            console.log(`Gateway:    Confirm OpenClaw after restart: ${brainFeedback.restart.gatewayStatusCommand}`);
        }
        console.log(`Check:      ${buildInstallStatusCommand(parsed.activationRoot)}`);
        console.log(`Learner:    ${buildLearnerServiceStatusCommand(parsed.activationRoot)}`);
        if (embedderProvision !== null && embedderProvision.state === "skipped") {
            console.log(`Embedder:   ${buildInstallEmbedderProvisionCommand(embedderProvision.baseUrl, embedderProvision.model)}`);
        }
    }
    return 0;
}
function runInstallCommand(parsed) {
    return runProfileHookAttachCommand(parsed);
}
function runAttachCommand(parsed) {
    return runProfileHookAttachCommand(parsed);
}
function validateOpenClawHome(openclawHome) {
    if (!existsSync(openclawHome)) {
        throw new Error(`--openclaw-home directory does not exist: ${openclawHome}`);
    }
    const openclawJsonPath = path.join(openclawHome, "openclaw.json");
    if (!existsSync(openclawJsonPath)) {
        throw new Error(`openclaw.json not found in ${openclawHome}`);
    }
}
function readJsonObjectRecord(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    return value;
}
function readOpenClawJsonConfig(openclawHome) {
    const openclawJsonPath = path.join(openclawHome, "openclaw.json");
    let parsed;
    try {
        parsed = JSON.parse(readFileSync(openclawJsonPath, "utf8"));
    }
    catch (error) {
        throw new Error(`Failed to read ${openclawJsonPath}: ${toErrorMessage(error)}`);
    }
    const config = readJsonObjectRecord(parsed);
    if (config === null) {
        throw new Error(`Failed to read ${openclawJsonPath}: openclaw.json must contain a top-level object`);
    }
    return {
        path: openclawJsonPath,
        config
    };
}
function ensureOpenClawBrainPluginConfig(openclawHome) {
    const { path: openclawJsonPath, config } = readOpenClawJsonConfig(openclawHome);
    const selectedInstall = findInstalledOpenClawBrainPlugin(openclawHome).selectedInstall;
    const plugins = readJsonObjectRecord(config.plugins);
    if (plugins === null) {
        return {
            path: openclawJsonPath,
            changed: false,
            detail: `Left ${shortenPath(openclawJsonPath)} unchanged because plugins config is not configured`
        };
    }
    const normalizedPlugins = normalizeOpenClawBrainPluginsConfig(plugins, selectedInstall);
    if (!normalizedPlugins.changed) {
        const details = [];
        if (!Object.prototype.hasOwnProperty.call(plugins, "allow")) {
            details.push("plugins.allow is not configured");
        }
        else if (!Array.isArray(plugins.allow)) {
            details.push("plugins.allow is not an array");
        }
        else {
            details.push(`plugins.allow already includes ${normalizedPlugins.allowedPluginIds.join(", ")}`);
        }
        const entries = readJsonObjectRecord(plugins.entries);
        if (entries !== null && Object.prototype.hasOwnProperty.call(entries, normalizedPlugins.canonicalEntryId)) {
            details.push(`plugins.entries already uses ${normalizedPlugins.canonicalEntryId}`);
        }
        return {
            path: openclawJsonPath,
            changed: false,
            detail: `Verified ${shortenPath(openclawJsonPath)} ${details.join("; ")}`
        };
    }
    config.plugins = normalizedPlugins.pluginsConfig;
    writeFileSync(openclawJsonPath, `${JSON.stringify(config, null, 2)}\n`, "utf8");
    return {
        path: openclawJsonPath,
        changed: true,
        detail: `Repaired ${shortenPath(openclawJsonPath)} by ${normalizedPlugins.changes.join("; ")}`
    };
}
function scrubOpenClawBrainPluginConfig(openclawHome) {
    const { path: openclawJsonPath, config } = readOpenClawJsonConfig(openclawHome);
    const selectedInstall = findInstalledOpenClawBrainPlugin(openclawHome).selectedInstall;
    const knownPluginIds = new Set(selectedInstall === null
        ? ["openclawbrain", "openclaw"]
        : getOpenClawBrainKnownPluginIds(selectedInstall));
    const plugins = readJsonObjectRecord(config.plugins);
    if (plugins === null) {
        return {
            path: openclawJsonPath,
            changed: false,
            detail: `No stale openclawbrain plugin config found in ${openclawJsonPath}`
        };
    }
    const changes = [];
    let changed = false;
    if (Array.isArray(plugins.allow)) {
        const filteredAllow = plugins.allow.filter((entry) => !knownPluginIds.has(entry));
        if (filteredAllow.length !== plugins.allow.length) {
            changed = true;
            changes.push("removed plugins.allow entries");
            if (filteredAllow.length > 0) {
                plugins.allow = filteredAllow;
            }
            else {
                delete plugins.allow;
            }
        }
    }
    const entries = readJsonObjectRecord(plugins.entries);
    if (entries !== null) {
        for (const pluginId of knownPluginIds) {
            if (!Object.prototype.hasOwnProperty.call(entries, pluginId)) {
                continue;
            }
            delete entries[pluginId];
            changed = true;
            changes.push(`removed plugins.entries.${pluginId}`);
        }
    }
    if (entries !== null && Object.keys(entries).length === 0 && Object.prototype.hasOwnProperty.call(plugins, "entries")) {
        delete plugins.entries;
        changed = true;
        changes.push("removed empty plugins.entries container");
    }
    if (Object.keys(plugins).length === 0 && Object.prototype.hasOwnProperty.call(config, "plugins")) {
        delete config.plugins;
        changed = true;
        changes.push("removed empty plugins container");
    }
    if (changed) {
        writeFileSync(openclawJsonPath, JSON.stringify(config, null, 2) + "\n", "utf8");
        return {
            path: openclawJsonPath,
            changed: true,
            detail: `Scrubbed stale openclawbrain plugin config in ${openclawJsonPath}: ${changes.join(", ")}`
        };
    }
    return {
        path: openclawJsonPath,
        changed: false,
        detail: `No stale openclawbrain plugin config found in ${openclawJsonPath}`
    };
}
function resolveCleanupActivationRoot(openclawHome, explicitActivationRoot) {
    if (explicitActivationRoot !== null) {
        return path.resolve(explicitActivationRoot);
    }
    const resolved = resolveActivationRoot({
        openclawHome,
        quiet: true
    });
    return resolved.trim().length === 0 ? null : path.resolve(resolved);
}
function removeProfileHookup(openclawHome, steps) {
    const installedPlugin = findInstalledOpenClawBrainPlugin(openclawHome);
    const installs = installedPlugin.selectedInstall === null
        ? []
        : [installedPlugin.selectedInstall, ...installedPlugin.additionalInstalls];
    if (installs.length === 0) {
        steps.push(`Profile hookup already absent under ${installedPlugin.extensionsDir}`);
        return {
            primaryPath: path.join(openclawHome, "extensions", "openclawbrain"),
            removedPaths: []
        };
    }
    const removedPaths = [];
    for (const install of installs
        .slice()
        .sort((left, right) => right.extensionDir.length - left.extensionDir.length)) {
        rmSync(install.extensionDir, { recursive: true, force: true });
        removedPaths.push(install.extensionDir);
        steps.push(`Removed ${describeOpenClawBrainInstallLayout(install.installLayout)}: ${install.extensionDir}`);
    }
    return {
        primaryPath: removedPaths[0],
        removedPaths
    };
}
function summarizeKeptActivationData(activationRoot) {
    if (activationRoot === null) {
        return {
            activationRoot: null,
            activationDataState: "unresolved",
            activationDataDetail: "Activation data preserved, but the activation root could not be resolved from the profile hook."
        };
    }
    return {
        activationRoot,
        activationDataState: "kept",
        activationDataDetail: `Activation data preserved at ${activationRoot}`
    };
}
function buildRestartGuidance(restart) {
    return buildCleanupRestartGuidance(restart);
}
function clearCleanupRuntimeLoadProof(activationRoot, openclawHome, steps) {
    if (activationRoot === null) {
        return;
    }
    try {
        const cleared = clearOpenClawProfileRuntimeLoadProof({
            activationRoot,
            openclawHome
        });
        if (cleared) {
            steps.push(`Cleared runtime-load proof for ${shortenPath(openclawHome)} from ${shortenPath(resolveAttachmentRuntimeLoadProofsPath(activationRoot))}`);
        }
    }
    catch (error) {
        steps.push(`Runtime-load proof cleanup failed open at ${shortenPath(resolveAttachmentRuntimeLoadProofsPath(activationRoot))}: ${toErrorMessage(error)}`);
    }
}
function runDetachCommand(parsed) {
    const steps = [];
    validateOpenClawHome(parsed.openclawHome);
    const targetInspection = inspectOpenClawHome(parsed.openclawHome);
    steps.push(`Detected layout: ${formatOpenClawTargetExplanation(targetInspection)}`);
    const activationRoot = resolveCleanupActivationRoot(parsed.openclawHome, parsed.activationRoot);
    clearCleanupRuntimeLoadProof(activationRoot, parsed.openclawHome, steps);
    const learnerService = resolveCleanupLearnerServiceOutcome(activationRoot, parsed.openclawHome);
    const pluginConfigCleanup = scrubOpenClawBrainPluginConfig(parsed.openclawHome);
    const removedHookup = removeProfileHookup(parsed.openclawHome, steps);
    const legacyResidue = removeLegacyProfileResidue(parsed.openclawHome);
    const activationData = summarizeKeptActivationData(activationRoot);
    const restartGuidance = buildRestartGuidance(parsed.restart);
    const nextSteps = [
        restartGuidance,
        activationRoot === null ? null : `Inspect preserved data: ${buildInstallStatusCommand(activationRoot)}`,
        activationRoot === null ? null : `Inspect learner service: ${buildLearnerServiceStatusCommand(activationRoot)}`,
        `Reattach later: ${buildAttachCommand(parsed.openclawHome, activationRoot)}`
    ].filter((step) => step !== null);
    steps.push(pluginConfigCleanup.detail);
    if (legacyResidue.removedNotes.length > 0) {
        steps.push(`Removed legacy profile notes: ${legacyResidue.removedNotes.map((notePath) => shortenPath(notePath)).join(", ")}`);
    }
    if (legacyResidue.updatedAgents.length > 0) {
        steps.push(`Removed legacy AGENTS.md brain references: ${legacyResidue.updatedAgents.map((agentsPath) => shortenPath(agentsPath)).join(", ")}`);
    }
    steps.push(learnerService.detail);
    steps.push(activationData.activationDataDetail);
    steps.push("Detach only removes the OpenClaw profile hook; it does not delete OpenClawBrain data.");
    if (parsed.json) {
        console.log(JSON.stringify({
            command: "detach",
            openclawHome: parsed.openclawHome,
            openclawTarget: {
                layout: targetInspection.layout,
                detail: describeOpenClawHomeInspection(targetInspection),
                profileId: targetInspection.profileId,
                profileSource: targetInspection.profileSource,
                configuredProfileIds: targetInspection.configuredProfileIds
            },
            extensionDir: removedHookup.primaryPath,
            removedHookDirs: removedHookup.removedPaths,
            activationRoot,
            dataAction: "kept",
            activationDataState: activationData.activationDataState,
            pluginConfigCleanup,
            learnerService,
            removedLegacyNotes: legacyResidue.removedNotes,
            updatedAgents: legacyResidue.updatedAgents,
            restartMode: parsed.restart,
            restartGuidance,
            nextSteps,
            steps
        }, null, 2));
    }
    else {
        console.log("DETACH complete\n");
        for (const step of steps) {
            console.log(`  ✓ ${step}`);
        }
        console.log("");
        console.log(`Lifecycle:     OpenClaw home ${shortenPath(parsed.openclawHome)} is detached from the brain hook.`);
        console.log(`Target:        ${formatOpenClawTargetExplanation(targetInspection)}`);
        if (activationRoot !== null) {
            console.log(`Brain data:    ${shortenPath(activationRoot)} remains available for inspection or reattach.`);
        }
        else {
            console.log("Brain data:    preserved, but the activation root could not be resolved from the removed hook.");
        }
        console.log(`Config:        ${pluginConfigCleanup.detail}`);
        console.log(`Learner:       ${learnerService.detail}`);
        console.log(`Next:          ${restartGuidance}`);
        if (activationRoot !== null) {
            console.log(`Check:         ${buildInstallStatusCommand(activationRoot)}`);
            console.log(`Service:       ${buildLearnerServiceStatusCommand(activationRoot)}`);
        }
        console.log(`Reattach:      ${buildAttachCommand(parsed.openclawHome, activationRoot)}`);
    }
    return 0;
}
function runUninstallCommand(parsed) {
    const steps = [];
    validateOpenClawHome(parsed.openclawHome);
    const targetInspection = inspectOpenClawHome(parsed.openclawHome);
    steps.push(`Detected layout: ${formatOpenClawTargetExplanation(targetInspection)}`);
    const activationRoot = resolveCleanupActivationRoot(parsed.openclawHome, parsed.activationRoot);
    clearCleanupRuntimeLoadProof(activationRoot, parsed.openclawHome, steps);
    if (parsed.dataMode === "purge" && activationRoot !== null) {
        assertActivationRootPurgeIsNotShared({
            activationRoot,
            openclawHome: parsed.openclawHome
        });
    }
    const learnerService = resolveCleanupLearnerServiceOutcome(activationRoot, parsed.openclawHome);
    const pluginConfigCleanup = scrubOpenClawBrainPluginConfig(parsed.openclawHome);
    if (parsed.dataMode === "purge" &&
        activationRoot !== null &&
        learnerService.state === "preserved" &&
        learnerService.matchesRequestedActivationRoot !== false) {
        throw new Error(`Refusing to purge activation root ${path.resolve(activationRoot)} because the background learner service for this exact root could not be removed. ${learnerService.detail}`);
    }
    const removedHookup = removeProfileHookup(parsed.openclawHome, steps);
    const legacyResidue = removeLegacyProfileResidue(parsed.openclawHome);
    let activationData;
    if (parsed.dataMode === "purge") {
        if (activationRoot === null) {
            throw new Error("--purge-data requires a resolvable activation root from the installed profile hook or --activation-root <path>");
        }
        if (existsSync(activationRoot)) {
            rmSync(activationRoot, { recursive: true, force: true });
            activationData = {
                activationRoot,
                activationDataState: "removed",
                activationDataDetail: `Activation data removed at ${activationRoot}`
            };
        }
        else {
            activationData = {
                activationRoot,
                activationDataState: "already_absent",
                activationDataDetail: `Activation data already absent at ${activationRoot}`
            };
        }
    }
    else {
        activationData = summarizeKeptActivationData(activationRoot);
    }
    const restartGuidance = buildRestartGuidance(parsed.restart);
    const nextSteps = [
        restartGuidance,
        parsed.dataMode === "keep" && activationRoot !== null ? `Inspect preserved data: ${buildInstallStatusCommand(activationRoot)}` : null,
        activationRoot === null ? null : `Inspect learner service: ${buildLearnerServiceStatusCommand(activationRoot)}`,
        parsed.dataMode === "keep"
            ? `Reattach later: ${buildAttachCommand(parsed.openclawHome, activationRoot)}`
            : `Reinstall later: ${buildInstallCommand(parsed.openclawHome)}`
    ].filter((step) => step !== null);
    steps.push(pluginConfigCleanup.detail);
    if (legacyResidue.removedNotes.length > 0) {
        steps.push(`Removed legacy profile notes: ${legacyResidue.removedNotes.map((notePath) => shortenPath(notePath)).join(", ")}`);
    }
    if (legacyResidue.updatedAgents.length > 0) {
        steps.push(`Removed legacy AGENTS.md brain references: ${legacyResidue.updatedAgents.map((agentsPath) => shortenPath(agentsPath)).join(", ")}`);
    }
    steps.push(learnerService.detail);
    steps.push(activationData.activationDataDetail);
    steps.push(parsed.dataMode === "purge"
        ? "Uninstall removed the OpenClaw profile hook and activation data."
        : "Uninstall removed the OpenClaw profile hook and kept activation data explicitly.");
    if (parsed.json) {
        console.log(JSON.stringify({
            command: "uninstall",
            openclawHome: parsed.openclawHome,
            openclawTarget: {
                layout: targetInspection.layout,
                detail: describeOpenClawHomeInspection(targetInspection),
                profileId: targetInspection.profileId,
                profileSource: targetInspection.profileSource,
                configuredProfileIds: targetInspection.configuredProfileIds
            },
            extensionDir: removedHookup.primaryPath,
            removedHookDirs: removedHookup.removedPaths,
            activationRoot,
            dataAction: parsed.dataMode,
            activationDataState: activationData.activationDataState,
            pluginConfigCleanup,
            learnerService,
            removedLegacyNotes: legacyResidue.removedNotes,
            updatedAgents: legacyResidue.updatedAgents,
            restartMode: parsed.restart,
            restartGuidance,
            nextSteps,
            steps
        }, null, 2));
    }
    else {
        console.log("UNINSTALL complete\n");
        for (const step of steps) {
            console.log(`  ✓ ${step}`);
        }
        console.log("");
        console.log(`Lifecycle:     OpenClaw home ${shortenPath(parsed.openclawHome)} no longer has the brain hook installed.`);
        console.log(`Target:        ${formatOpenClawTargetExplanation(targetInspection)}`);
        console.log(`Data mode:     ${parsed.dataMode === "purge" ? "purged" : "kept"}`);
        if (activationRoot !== null) {
            console.log(`Activation:    ${parsed.dataMode === "purge" ? shortenPath(activationRoot) : `${shortenPath(activationRoot)} preserved`}`);
        }
        console.log(`Config:        ${pluginConfigCleanup.detail}`);
        console.log(`Learner:       ${learnerService.detail}`);
        console.log(`Next:          ${restartGuidance}`);
        if (parsed.dataMode === "keep" && activationRoot !== null) {
            console.log(`Check:         ${buildInstallStatusCommand(activationRoot)}`);
        }
        if (activationRoot !== null) {
            console.log(`Service:       ${buildLearnerServiceStatusCommand(activationRoot)}`);
        }
        if (parsed.dataMode === "keep") {
            console.log(`Reattach:      ${buildAttachCommand(parsed.openclawHome, activationRoot)}`);
        }
        else {
            console.log(`Reinstall:     ${buildInstallCommand(parsed.openclawHome)}`);
        }
    }
    return 0;
}
function resolveServeTimeLearningRuntimeInput(activationRoot) {
    const logPath = resolveLearningSpineLogPath(activationRoot, "serveTimeRouteDecisions");
    const { entries: serveTimeDecisions, fallbackReason } = readBoundedJsonlTail(logPath);
    const decisionLogCount = serveTimeDecisions.length;
    const pgVersion = decisionLogCount > 0 ? "v2" : "v1";
    return {
        pgVersion,
        serveTimeDecisions,
        decisionLogCount,
        baselineState: pgVersion === "v2" ? loadOrInitBaseline(activationRoot) : undefined,
        fallbackReason: fallbackReason === null ? null : `serve_time_decision_log_${fallbackReason}`
    };
}
function runLearnCommand(parsed) {
    const learnStatePath = path.join(parsed.activationRoot, "learn-cli-state.json");
    const teacherSnapshotPath = resolveAsyncTeacherLiveLoopSnapshotPath(parsed.activationRoot);
    function isLearnRuntimeStateLike(value) {
        if (typeof value !== "object" || value === null) {
            return false;
        }
        const candidate = value;
        return (candidate.runtimeOwner === "openclaw" &&
            typeof candidate.cursor === "object" &&
            candidate.cursor !== null &&
            typeof candidate.pending === "object" &&
            candidate.pending !== null &&
            Array.isArray(candidate.pending.live) &&
            Array.isArray(candidate.pending.backfill) &&
            typeof candidate.materializationCount === "number" &&
            typeof candidate.sparseFeedback === "object" &&
            candidate.sparseFeedback !== null);
    }
    function loadPersistedLearnCliState() {
        if (!existsSync(learnStatePath)) {
            return {
                state: createAlwaysOnLearningRuntimeState(),
                loaded: false,
                resetReason: null
            };
        }
        try {
            const persisted = readJsonFile(learnStatePath);
            if (persisted.contract !== "openclaw.learn_cli_state.v1" || !isLearnRuntimeStateLike(persisted.state)) {
                throw new Error("persisted learn state shape is invalid");
            }
            return {
                state: persisted.state,
                loaded: true,
                resetReason: null
            };
        }
        catch (error) {
            return {
                state: createAlwaysOnLearningRuntimeState(),
                loaded: false,
                resetReason: error instanceof Error ? error.message : "persisted learn state could not be parsed"
            };
        }
    }
    function persistLearnCliState(state, updatedAt) {
        const payload = {
            contract: "openclaw.learn_cli_state.v1",
            updatedAt,
            state
        };
        mkdirSync(path.dirname(learnStatePath), { recursive: true });
        writeFileSync(learnStatePath, JSON.stringify(payload, null, 2) + "\n", "utf8");
    }
    const activationRoot = parsed.activationRoot;
    const persistedState = loadPersistedLearnCliState();
    const stores = discoverOpenClawSessionStores();
    if (stores.length === 0) {
        const labelFlow = {
            source: "missing",
            humanLabelCount: 0,
            selfLabelCount: 0,
            asyncTeacherArtifactCount: 0,
            implicitPositiveCount: 0,
            detail: "no local session stores were found"
        };
        const learningPath = summarizeLearningPathFromMaterialization(null);
        if (parsed.json) {
            console.log(JSON.stringify({
                command: "learn",
                activationRoot,
                scannedSessions: 0,
                newEvents: 0,
                loadedState: persistedState.loaded,
                statePath: learnStatePath,
                stateResetReason: persistedState.resetReason,
                materialized: null,
                promoted: false,
                graph: null,
                labelFlow,
                learningPath,
                message: "No local session stores found."
            }));
        }
        else {
            console.log("No new session data. Brain is up to date.");
        }
        return 0;
    }
    let totalSessions = 0;
    const allInteractionEvents = [];
    const allFeedbackEvents = [];
    let nextSequence = 1;
    const discoveredSessions = stores
        .flatMap((store) => {
        const sessionIndex = loadOpenClawSessionIndex(store.indexPath);
        return Object.entries(sessionIndex).map(([sessionKey, entry]) => ({
            store,
            sessionKey,
            entry
        }));
    })
        .sort((left, right) => {
        if (left.entry.updatedAt !== right.entry.updatedAt) {
            return left.entry.updatedAt - right.entry.updatedAt;
        }
        if (left.store.indexPath !== right.store.indexPath) {
            return left.store.indexPath.localeCompare(right.store.indexPath);
        }
        return left.sessionKey.localeCompare(right.sessionKey);
    })
        .filter((() => {
        const seenSessionIds = new Set();
        return (session) => {
            const sessionId = session.entry.sessionId;
            if (sessionId !== undefined && seenSessionIds.has(sessionId)) {
                return false;
            }
            if (sessionId !== undefined) {
                seenSessionIds.add(sessionId);
            }
            return true;
        };
    })());
    for (const session of discoveredSessions) {
        const sessionFile = session.entry.sessionFile;
        const records = typeof sessionFile !== "string" || sessionFile.trim().length === 0
            ? []
            : (() => {
                try {
                    return readOpenClawSessionFile(sessionFile);
                }
                catch {
                    return [];
                }
            })();
        const sessionExport = buildPassiveLearningSessionExportFromOpenClawSessionStore({
            sessionKey: session.sessionKey,
            indexEntry: session.entry,
            records,
            agentId: session.store.agentId,
            sequenceStart: nextSequence
        });
        nextSequence = sessionExport.nextSequence;
        totalSessions += 1;
        allInteractionEvents.push(...sessionExport.interactionEvents);
        allFeedbackEvents.push(...sessionExport.feedbackEvents);
    }
    const seenInteractionIds = new Set();
    const dedupedInteractionEvents = [];
    for (const event of allInteractionEvents) {
        if (!seenInteractionIds.has(event.eventId)) {
            seenInteractionIds.add(event.eventId);
            dedupedInteractionEvents.push(event);
        }
    }
    const seenFeedbackIds = new Set();
    const dedupedFeedbackEvents = [];
    for (const event of allFeedbackEvents) {
        if (!seenFeedbackIds.has(event.eventId)) {
            seenFeedbackIds.add(event.eventId);
            dedupedFeedbackEvents.push(event);
        }
    }
    const totalEvents = dedupedInteractionEvents.length + dedupedFeedbackEvents.length;
    const now = new Date().toISOString();
    const normalizedEventExport = totalEvents === 0
        ? null
        : buildNormalizedEventExport({
            interactionEvents: dedupedInteractionEvents,
            feedbackEvents: dedupedFeedbackEvents
        });
    const teacherArtifacts = normalizedEventExport === null
        ? []
        : buildTeacherSupervisionArtifactsFromNormalizedEventExport({
            normalizedEventExport,
            observedAt: now
        });
    const labelFlow = normalizedEventExport === null
        ? {
            source: "missing",
            humanLabelCount: 0,
            selfLabelCount: 0,
            asyncTeacherArtifactCount: 0,
            implicitPositiveCount: 0,
            detail: "no normalized learning export was built"
        }
        : summarizeNormalizedEventExportLabelFlow(normalizedEventExport, teacherArtifacts.length);
    if (totalEvents === 0) {
        if (parsed.json) {
            console.log(JSON.stringify({
                command: "learn",
                activationRoot,
                scannedSessions: totalSessions,
                newEvents: 0,
                loadedState: persistedState.loaded,
                statePath: learnStatePath,
                stateResetReason: persistedState.resetReason,
                materialized: null,
                promoted: false,
                graph: null,
                labelFlow,
                learningPath: summarizeLearningPathFromMaterialization(null),
                message: "No new session data. Brain is up to date."
            }));
        }
        else {
            console.log("No new session data. Brain is up to date.");
        }
        return 0;
    }
    const learningExport = normalizedEventExport;
    const serveTimeLearning = resolveServeTimeLearningRuntimeInput(activationRoot);
    const learnerResult = drainAlwaysOnLearningRuntime({
        packLabel: "learn-cli",
        workspace: {
            workspaceId: "learn-cli",
            snapshotId: `learn-cli@${now.slice(0, 10)}`,
            capturedAt: now,
            rootDir: activationRoot,
            revision: "learn-cli-v1"
        },
        interactionEvents: dedupedInteractionEvents,
        feedbackEvents: dedupedFeedbackEvents,
        teacherSupervisionArtifacts: teacherArtifacts,
        learnedRouting: true,
        state: persistedState.state,
        builtAt: now,
        maxCycles: 16,
        pgVersion: serveTimeLearning.pgVersion,
        ...(serveTimeLearning.decisionLogCount > 0 ? { serveTimeDecisions: serveTimeLearning.serveTimeDecisions } : {}),
        ...(serveTimeLearning.baselineState !== undefined ? { baselineState: serveTimeLearning.baselineState } : {})
    });
    const lastMaterialization = learnerResult.materializations.at(-1) ?? null;
    const plan = describeAlwaysOnLearningRuntimeState(learnerResult.state, lastMaterialization);
    const learningPath = summarizeLearningPathFromMaterialization(lastMaterialization);
    const supervisionCount = lastMaterialization?.candidate.summary.learnedRouter.supervisionCount ?? 0;
    const routerUpdateCount = lastMaterialization?.candidate.summary.learnedRouter.updateCount ?? 0;
    const routerNoOpReason = lastMaterialization?.candidate.summary.learnedRouter.noOpReason ?? null;
    const graphEvolution = lastMaterialization?.candidate.payloads.graph.evolution;
    const graphSummary = graphEvolution === undefined
        ? null
        : {
            structuralOps: graphEvolution.structuralOps,
            connectDiagnostics: graphEvolution.connectDiagnostics ?? null
        };
    const connectSummary = graphSummary?.connectDiagnostics === null || graphSummary?.connectDiagnostics === undefined
        ? ""
        : ` connect candidates=${graphSummary.connectDiagnostics.candidatePairCount} applied=${graphSummary.connectDiagnostics.appliedPairCount} edges=${graphSummary.connectDiagnostics.createdEdgeCount}.`;
    const routingBuild = lastMaterialization?.candidate.routingBuild ?? {
        learnedRoutingPath: serveTimeLearning.pgVersion === "v2" ? "policy_gradient_v2" : "policy_gradient_v1",
        pgVersionRequested: serveTimeLearning.pgVersion,
        pgVersionUsed: serveTimeLearning.pgVersion,
        decisionLogCount: serveTimeLearning.decisionLogCount,
        fallbackReason: serveTimeLearning.pgVersion === "v1" ? serveTimeLearning.fallbackReason ?? "no_serve_time_decisions" : null,
        updatedBaseline: null
    };
    const learnPathReport = {
        ...routingBuild,
        fallbackReason: routingBuild.fallbackReason ??
            (routingBuild.pgVersionUsed === "v1" ? serveTimeLearning.fallbackReason ?? "no_serve_time_decisions" : null)
    };
    let promoted = false;
    let materializedPackId = null;
    let baselinePersisted = false;
    const latestTeacherFreshness = teacherArtifacts.length === 0
        ? "none"
        : teacherArtifacts.some((artifact) => artifact.freshness.status === "fresh")
            ? "fresh"
            : "stale";
    if (lastMaterialization !== null) {
        const candidatePackRoot = path.join(activationRoot, "packs", `learn-cli-${Date.now()}`);
        mkdirSync(candidatePackRoot, { recursive: true });
        const candidateDescriptor = materializeAlwaysOnLearningCandidatePack(candidatePackRoot, lastMaterialization);
        stageCandidatePack(activationRoot, candidatePackRoot, {
            updatedAt: now,
            reason: "learn_cli_stage"
        });
        promoteCandidatePack(activationRoot, {
            updatedAt: now,
            reason: "learn_cli_promote"
        });
        if (learnPathReport.pgVersionUsed === "v2" && learnPathReport.updatedBaseline !== null) {
            persistBaseline(activationRoot, learnPathReport.updatedBaseline);
            baselinePersisted = true;
        }
        materializedPackId = candidateDescriptor.manifest.packId;
        promoted = true;
    }
    persistLearnCliState(learnerResult.state, now);
    writeJsonFile(teacherSnapshotPath, {
        runtimeOwner: "openclaw",
        queue: {
            capacity: 1,
            depth: 0,
            running: false
        },
        teacher: {
            artifactCount: teacherArtifacts.length,
            artifacts: teacherArtifacts,
            latestFreshness: latestTeacherFreshness
        },
        learner: {
            state: learnerResult.state,
            lastMaterialization
        },
        diagnostics: {
            acceptedExportCount: 1,
            processedExportCount: 1,
            duplicateExportCount: 0,
            droppedExportCount: 0,
            emittedArtifactCount: teacherArtifacts.length,
            dedupedArtifactCount: 0,
            lastProcessedAt: now,
            latestFreshness: latestTeacherFreshness,
            lastNoOpReason: teacherArtifacts.length === 0 ? "no_teacher_artifacts" : "none",
            notes: [
                `learn-cli export=${learningExport.provenance.exportDigest} range=${learningExport.range.start}-${learningExport.range.end}/${learningExport.range.count}`,
                `teacher artifacts=${teacherArtifacts.length} freshness=${latestTeacherFreshness}`,
                `last materialized pack=${materializedPackId ?? "none"}`
            ]
        },
        state: {
            interactionEvents: learningExport.interactionEvents,
            feedbackEvents: learningExport.feedbackEvents,
            seenExportDigests: [learningExport.provenance.exportDigest]
        },
        runtime: {
            startedAt: now,
            lastHeartbeatAt: now,
            lastScanAt: now,
            scanRoot: null,
            lastAppliedMaterializationJobId: lastMaterialization?.jobId ?? null
        }
    });
    const tracedLearningBridge = mergeTracedLearningBridgePayload({
        updatedAt: now,
        routeTraceCount: lastMaterialization?.candidate.summary.learnedRouter.routeTraceCount ?? serveTimeLearning.decisionLogCount,
        supervisionCount,
        routerUpdateCount,
        teacherArtifactCount: teacherArtifacts.length,
        pgVersionRequested: learnPathReport.pgVersionRequested,
        pgVersionUsed: learnPathReport.pgVersionUsed,
        decisionLogCount: learnPathReport.decisionLogCount,
        fallbackReason: learnPathReport.fallbackReason,
        routerNoOpReason,
        materializedPackId,
        promoted,
        baselinePersisted,
        source: {
            command: "learn",
            exportDigest: learningExport.provenance.exportDigest,
            teacherSnapshotPath
        }
    }, loadBrainStoreTracedLearningBridge());
    const surfacedSupervisionCount = tracedLearningBridge.supervisionCount;
    const surfacedRouterUpdateCount = tracedLearningBridge.routerUpdateCount;
    const surfacedRouterNoOpReason = tracedLearningBridge.routerNoOpReason;
    persistBrainStoreTracedLearningBridge(tracedLearningBridge);
    writeTracedLearningBridge(activationRoot, tracedLearningBridge);
    const summaryMessage = materializedPackId === null
        ? `Scanned ${totalSessions} sessions, ${totalEvents} new events, no candidate materialized, no promotion.`
        : `Scanned ${totalSessions} sessions, ${totalEvents} new events, materialized ${materializedPackId}, promoted.${connectSummary}`;
    if (parsed.json) {
        console.log(JSON.stringify({
            command: "learn",
            activationRoot,
            scannedSessions: totalSessions,
            newEvents: totalEvents,
            loadedState: persistedState.loaded,
            statePath: learnStatePath,
            stateResetReason: persistedState.resetReason,
            drain: {
                cyclesRun: learnerResult.cycles.length,
                stopReason: learnerResult.stopReason,
                drained: learnerResult.drained,
                materializationCount: learnerResult.materializations.length
            },
            learner: {
                teacherBudget: learnerResult.state.sparseFeedback.teacherBudget,
                eligibleFeedbackCount: learnerResult.state.sparseFeedback.eligibleFeedbackCount,
                budgetedOutFeedbackCount: learnerResult.state.sparseFeedback.budgetedOutFeedbackCount,
                supervisionCount: surfacedSupervisionCount,
                routerUpdateCount: surfacedRouterUpdateCount,
                routerNoOpReason: surfacedRouterNoOpReason,
                pending: plan.pending,
                learnedRange: plan.learnedRange
            },
            materialized: materializedPackId,
            promoted,
            graph: graphSummary,
            labelFlow,
            learningPath,
            learnedRoutingPath: learnPathReport.learnedRoutingPath,
            pgVersionRequested: learnPathReport.pgVersionRequested,
            pgVersionUsed: learnPathReport.pgVersionUsed,
            decisionLogCount: learnPathReport.decisionLogCount,
            fallbackReason: learnPathReport.fallbackReason,
            baselinePersisted,
            message: summaryMessage
        }, null, 2));
    }
    else {
        const text = materializedPackId === null
            ? `Scanned ${totalSessions} sessions, ${totalEvents} new events, no promotion. cycles=${learnerResult.cycles.length} stop=${learnerResult.stopReason} supervision=${surfacedSupervisionCount}.`
            : `Scanned ${totalSessions} sessions, ${totalEvents} new events, materialized ${materializedPackId}, promoted.${connectSummary} cycles=${learnerResult.cycles.length} supervision=${surfacedSupervisionCount}.`;
        console.log(text);
        console.log(`labels: source=${labelFlow.source} human=${labelFlow.humanLabelCount ?? "none"} self=${labelFlow.selfLabelCount ?? "none"} implicitPositive=${labelFlow.implicitPositiveCount ?? "none"} teacherArtifacts=${labelFlow.asyncTeacherArtifactCount ?? "none"}`);
        console.log(`path: source=${learningPath.source} pg=${learningPath.policyGradientVersion} method=${learningPath.policyGradientMethod ?? "none"} target=${learningPath.targetConstruction ?? "none"} connect=${learningPath.connectOpsFired ?? "none"} trajectories=${learningPath.reconstructedTrajectoryCount ?? "none"}`);
        console.log(`learned routing: path=${learnPathReport.learnedRoutingPath} pg=${learnPathReport.pgVersionUsed ?? "n/a"} decisions=${learnPathReport.decisionLogCount}` +
            `${learnPathReport.fallbackReason === null ? "" : ` fallback=${learnPathReport.fallbackReason}`}` +
            `${learnPathReport.pgVersionUsed === "v2" ? ` baseline=${baselinePersisted ? "persisted" : "unchanged"}` : ""}`);
    }
    return 0;
}
function formatTimestamp() {
    const now = new Date();
    return `[${now.toTimeString().slice(0, 8)}]`;
}
function watchLog(message) {
    console.log(`${formatTimestamp()} ${message}`);
}
function formatWatchError(error) {
    return error instanceof Error ? error.message : String(error);
}
function sanitizeWatchPathSegment(value) {
    const sanitized = value
        .replace(/[^a-zA-Z0-9._-]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 96);
    return sanitized.length > 0 ? sanitized : "session";
}
function readOptionalJsonFile(filePath) {
    if (!existsSync(filePath)) {
        return null;
    }
    try {
        return JSON.parse(readFileSync(filePath, "utf8"));
    }
    catch {
        return null;
    }
}
function writeJsonFile(filePath, value) {
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
function loadWatchSessionTailCursor(cursorPath) {
    const parsed = readOptionalJsonFile(cursorPath);
    if (Array.isArray(parsed)) {
        return parsed;
    }
    if (parsed !== null && Array.isArray(parsed.cursor)) {
        return parsed.cursor;
    }
    return [];
}
function persistWatchSessionTailCursor(cursorPath, cursor) {
    writeJsonFile(cursorPath, {
        contract: "openclaw_watch_session_tail_cursor.v1",
        runtimeOwner: "openclaw",
        updatedAt: new Date().toISOString(),
        cursor
    });
}
function countWatchCursorBridgedEvents(cursor) {
    return cursor.reduce((sum, entry) => sum + entry.bridgedEventCount, 0);
}
function listWatchRuntimeEventExportBundleRoots(scanRoot) {
    if (!existsSync(scanRoot)) {
        return [];
    }
    return readdirSync(scanRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => path.join(scanRoot, entry.name))
        .sort((left, right) => left.localeCompare(right));
}
async function replayWatchScanRootIntoTeacherLoop(teacherLoop, scanRoot) {
    const seenExportDigests = new Set();
    const bundles = listWatchRuntimeEventExportBundleRoots(scanRoot)
        .map((rootDir) => {
        try {
            return loadRuntimeEventExportBundle(rootDir);
        }
        catch {
            return null;
        }
    })
        .filter((bundle) => bundle !== null)
        .sort((left, right) => {
        const exportedAtCompare = left.manifest.exportedAt.localeCompare(right.manifest.exportedAt);
        if (exportedAtCompare !== 0) {
            return exportedAtCompare;
        }
        if (left.normalizedEventExport.range.start !== right.normalizedEventExport.range.start) {
            return left.normalizedEventExport.range.start - right.normalizedEventExport.range.start;
        }
        if (left.normalizedEventExport.range.end !== right.normalizedEventExport.range.end) {
            return left.normalizedEventExport.range.end - right.normalizedEventExport.range.end;
        }
        return left.normalizedEventExport.provenance.exportDigest.localeCompare(right.normalizedEventExport.provenance.exportDigest);
    });
    let replayedBundleCount = 0;
    let replayedEventCount = 0;
    for (const bundle of bundles) {
        const exportDigest = bundle.normalizedEventExport.provenance.exportDigest;
        if (seenExportDigests.has(exportDigest)) {
            continue;
        }
        seenExportDigests.add(exportDigest);
        let enqueue = teacherLoop.enqueueNormalizedEventExport(bundle.normalizedEventExport, {
            observedAt: bundle.manifest.exportedAt
        });
        if (!enqueue.accepted && enqueue.reason === "queue_full") {
            await teacherLoop.flush();
            enqueue = teacherLoop.enqueueNormalizedEventExport(bundle.normalizedEventExport, {
                observedAt: bundle.manifest.exportedAt
            });
        }
        if (!enqueue.accepted) {
            continue;
        }
        replayedBundleCount += 1;
        replayedEventCount += bundle.normalizedEventExport.range.count;
    }
    if (replayedBundleCount > 0) {
        await teacherLoop.flush();
    }
    return {
        replayedBundleCount,
        replayedEventCount
    };
}
function exportLocalSessionTailChangesToScanRoot(input) {
    let exportedBundleCount = 0;
    let exportedEventCount = 0;
    const warnings = [];
    for (const change of input.changes) {
        if (change.scannedEventExport === null) {
            continue;
        }
        const built = buildNormalizedEventExportFromScannedEvents(change.scannedEventExport);
        if (!built.ok) {
            warnings.push(`${change.sessionKey}: ${built.error}`);
            continue;
        }
        const exportDigest = built.normalizedEventExport.provenance.exportDigest.replace(/^sha256-/u, "");
        const exportName = `session-tail-${sanitizeWatchPathSegment(change.sessionKey)}-${built.normalizedEventExport.range.start}-${built.normalizedEventExport.range.end}-${exportDigest.slice(0, 12)}`;
        const result = writeScannedEventExportBundle({
            rootDir: path.join(input.scanRoot, exportName),
            exportName,
            exportedAt: input.polledAt,
            scannedEventExport: change.scannedEventExport
        });
        if (!result.ok) {
            warnings.push(`${change.sessionKey}: ${result.error}`);
            continue;
        }
        exportedBundleCount += 1;
        exportedEventCount += result.normalizedEventExport.range.count;
    }
    return {
        exportedBundleCount,
        exportedEventCount,
        warnings
    };
}
function summarizeVectorEmbeddingState(vectors) {
    return summarizePackVectorEmbeddingState(vectors);
}
function buildWatchEmbedTracePoint(input) {
    const summary = summarizeVectorEmbeddingState(input.vectors);
    return {
        slot: input.slot,
        packId: input.packId,
        runtimeEmbedderPresent: input.embedder !== null,
        runtimeEmbedderModel: input.embedder?.model ?? null,
        vectorEntryCount: summary.vectorEntryCount,
        numericEmbeddingEntryCount: summary.numericEmbeddingEntryCount,
        embeddingModels: summary.embeddingModels,
        error: input.error ?? null
    };
}
function buildWatchEmbedTracePointFromPack(input) {
    return buildWatchEmbedTracePoint({
        slot: input.slot,
        packId: input.pack?.manifest.packId ?? null,
        embedder: input.embedder,
        vectors: input.pack?.vectors,
        error: input.error ?? null
    });
}
function formatWatchEmbedTracePoint(label, point) {
    const models = point.embeddingModels.length === 0 ? "none" : point.embeddingModels.join("|");
    const slot = point.slot ?? "build";
    const packId = point.packId ?? "unknown";
    const embedderState = point.runtimeEmbedderPresent ? `present:${point.runtimeEmbedderModel ?? "unknown"}` : "null";
    const counts = point.vectorEntryCount === null || point.numericEmbeddingEntryCount === null
        ? "vectors=unknown numeric=unknown"
        : `vectors=${point.vectorEntryCount} numeric=${point.numericEmbeddingEntryCount}`;
    const error = point.error === null ? "" : ` error=${point.error}`;
    return `embed-trace ${label} slot=${slot} pack=${packId} runtimeEmbedder=${embedderState} ${counts} models=${models}${error}`;
}
async function applyWatchMaterialization(activationRoot, snapshot, lastHandledMaterializationPackId, embedder, log) {
    let materialization = snapshot?.learner?.lastMaterialization ?? null;
    if (materialization === null) {
        return {
            lastHandledMaterializationPackId,
            logLine: null,
            materializedPackId: null,
            embedInstrumentation: null,
            failure: null
        };
    }
    const packId = typeof materialization?.candidate?.summary?.packId === "string"
        ? materialization.candidate.summary.packId
        : null;
    if (packId === null || packId === lastHandledMaterializationPackId) {
        return {
            lastHandledMaterializationPackId,
            logLine: null,
            materializedPackId: packId,
            embedInstrumentation: null,
            failure: null
        };
    }
    if (embedder !== null) {
        materialization = {
            ...materialization,
            candidate: await reindexCandidatePackBuildResultWithEmbedder(materialization.candidate, embedder)
        };
        if (snapshot?.learner !== undefined && snapshot.learner !== null) {
            snapshot.learner.lastMaterialization = materialization;
        }
    }
    const shortPackId = packId.length > 16 ? packId.slice(0, 16) : packId;
    const observedAt = new Date().toISOString();
    const beforeCandidateMaterialization = buildWatchEmbedTracePoint({
        slot: null,
        packId,
        embedder,
        vectors: materialization?.candidate?.payloads?.vectors
    });
    let embedInstrumentation = {
        observedAt,
        candidatePackId: packId,
        promotionAllowed: null,
        promotionFindings: [],
        beforeCandidateMaterialization,
        afterCandidateMaterialization: null,
        afterStage: null,
        afterPromote: null
    };
    log?.(formatWatchEmbedTracePoint("before_materialize", beforeCandidateMaterialization));
    try {
        const candidateRootDir = path.resolve(activationRoot, "packs", packId);
        mkdirSync(candidateRootDir, { recursive: true });
        let activeBeforePack = null;
        try {
            activeBeforePack = loadPackFromActivation(activationRoot, "active", { requireActivationReady: true });
        }
        catch {
            activeBeforePack = null;
        }
        const candidateDescriptor = materializeAlwaysOnLearningCandidatePack(candidateRootDir, materialization);
        embedInstrumentation = {
            ...embedInstrumentation,
            afterCandidateMaterialization: buildWatchEmbedTracePointFromPack({
                slot: "candidate",
                pack: candidateDescriptor,
                embedder
            })
        };
        if (embedInstrumentation.afterCandidateMaterialization !== null) {
            log?.(formatWatchEmbedTracePoint("after_materialize", embedInstrumentation.afterCandidateMaterialization));
        }
        appendLearningUpdateLogs({
            activationRoot,
            materialization,
            activeBeforePack,
            candidateDescriptor
        });
        const now = observedAt;
        stageCandidatePack(activationRoot, candidateRootDir, {
            updatedAt: now,
            reason: `watch_stage:${materialization.reason}:${materialization.lane}`
        });
        const inspection = inspectActivationState(activationRoot, now);
        let stagedPack = null;
        let stagedPackError = null;
        try {
            stagedPack = loadPackFromActivation(activationRoot, "candidate", { requireActivationReady: true });
        }
        catch (error) {
            stagedPackError = formatWatchError(error);
        }
        embedInstrumentation = {
            ...embedInstrumentation,
            promotionAllowed: inspection.promotion.allowed,
            promotionFindings: [...inspection.promotion.findings],
            afterStage: buildWatchEmbedTracePointFromPack({
                slot: "candidate",
                pack: stagedPack,
                embedder,
                error: stagedPackError
            })
        };
        if (embedInstrumentation.afterStage !== null) {
            log?.(formatWatchEmbedTracePoint("after_stage", embedInstrumentation.afterStage));
        }
        if (inspection.promotion.allowed) {
            promoteCandidatePack(activationRoot, {
                updatedAt: now,
                reason: `watch_promote:${materialization.reason}:${materialization.lane}`
            });
            let promotedPack = null;
            let promotedPackError = null;
            try {
                promotedPack = loadPackFromActivation(activationRoot, "active", { requireActivationReady: true });
            }
            catch (error) {
                promotedPackError = formatWatchError(error);
            }
            embedInstrumentation = {
                ...embedInstrumentation,
                afterPromote: buildWatchEmbedTracePointFromPack({
                    slot: "active",
                    pack: promotedPack,
                    embedder,
                    error: promotedPackError
                })
            };
            if (embedInstrumentation.afterPromote !== null) {
                log?.(formatWatchEmbedTracePoint("after_promote", embedInstrumentation.afterPromote));
            }
            return {
                lastHandledMaterializationPackId: packId,
                materializedPackId: packId,
                logLine: `Promoted ${shortPackId} → active`,
                embedInstrumentation,
                failure: null
            };
        }
        return {
            lastHandledMaterializationPackId: packId,
            materializedPackId: packId,
            logLine: `Staged ${shortPackId} (promotion blocked: ${inspection.promotion.findings.join(", ")})`,
            embedInstrumentation,
            failure: null
        };
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        embedInstrumentation = {
            ...embedInstrumentation,
            afterCandidateMaterialization: embedInstrumentation.afterCandidateMaterialization ??
                buildWatchEmbedTracePoint({
                    slot: "candidate",
                    packId,
                    embedder,
                    vectors: null,
                    error: message
                })
        };
        return {
            lastHandledMaterializationPackId,
            materializedPackId: packId,
            logLine: `Promotion failed for ${shortPackId}: ${message}`,
            embedInstrumentation,
            failure: {
                mode: "materialization_failed",
                detail: message,
                at: new Date().toISOString()
            }
        };
    }
}
function resolveWatchTeacherLabelerConfig(input, activationRoot) {
    if (input !== undefined) {
        return {
            teacherLabeler: input,
            warnings: []
        };
    }
    const providerConfig = readOpenClawBrainProviderConfigFromSources({
        env: process.env,
        activationRoot
    });
    const warnings = providerConfig.warnings.filter((warning) => /OPENCLAWBRAIN_TEACHER_|provider defaults/u.test(warning));
    if (providerConfig.teacher.provider !== "ollama") {
        return {
            teacherLabeler: null,
            warnings
        };
    }
    return {
        teacherLabeler: {
            provider: "ollama",
            baseUrl: providerConfig.teacherBaseUrl,
            model: providerConfig.teacher.model,
            ...(providerConfig.teacher.timeoutMs === undefined ? {} : { timeoutMs: providerConfig.teacher.timeoutMs }),
            ...(providerConfig.teacher.maxPromptChars === undefined ? {} : { maxPromptChars: providerConfig.teacher.maxPromptChars }),
            ...(providerConfig.teacher.maxResponseChars === undefined ? {} : { maxResponseChars: providerConfig.teacher.maxResponseChars }),
            ...(providerConfig.teacher.maxOutputTokens === undefined ? {} : { maxOutputTokens: providerConfig.teacher.maxOutputTokens }),
            ...(providerConfig.teacher.maxArtifactsPerExport === undefined
                ? {}
                : { maxArtifactsPerExport: providerConfig.teacher.maxArtifactsPerExport }),
            ...(providerConfig.teacher.maxInteractionsPerExport === undefined
                ? {}
                : { maxInteractionsPerExport: providerConfig.teacher.maxInteractionsPerExport })
        },
        warnings
    };
}
function resolveWatchEmbedderConfig(input, activationRoot) {
    if (input !== undefined) {
        return {
            embedder: input,
            warnings: []
        };
    }
    const defaultsResult = readOpenClawBrainProviderDefaults(activationRoot);
    const providerConfig = readOpenClawBrainProviderConfigFromSources({
        env: process.env,
        activationRoot,
        defaults: defaultsResult.defaults
    });
    const warnings = [...new Set([
            ...defaultsResult.warnings.filter((warning) => /OPENCLAWBRAIN_EMBEDDER_|provider defaults/u.test(warning)),
            ...providerConfig.warnings.filter((warning) => /OPENCLAWBRAIN_EMBEDDER_|provider defaults/u.test(warning))
        ])];
    const explicitEnv = typeof process.env[OPENCLAWBRAIN_EMBEDDER_PROVIDER_ENV] === "string" ||
        typeof process.env[OPENCLAWBRAIN_EMBEDDER_MODEL_ENV] === "string" ||
        typeof process.env[OPENCLAWBRAIN_EMBEDDER_BASE_URL_ENV] === "string";
    // Legacy install-written provider-defaults.json files can predate embedder fields entirely.
    // If a persisted defaults file exists, treat that activation root as explicitly configured and
    // let provider-config resolution fill in the embedder fallback instead of silently dropping to null.
    const explicitDefaults = defaultsResult.defaults !== null;
    if (!explicitEnv && !explicitDefaults) {
        return {
            embedder: null,
            warnings
        };
    }
    if (providerConfig.embedder.provider !== "ollama") {
        return {
            embedder: null,
            warnings
        };
    }
    return {
        embedder: createOllamaEmbedder({
            baseUrl: providerConfig.embedderBaseUrl,
            model: providerConfig.embedder.model
        }),
        warnings
    };
}
function summarizeWatchLatestUserMessage(localPoll) {
    let latest = null;
    for (const change of localPoll.changes) {
        if (change.lastUserMessageAt === null || change.lastUserMessageText === null) {
            continue;
        }
        const candidate = {
            at: change.lastUserMessageAt,
            text: change.lastUserMessageText,
            sessionId: change.sessionId
        };
        if (latest === null || Date.parse(candidate.at) >= Date.parse(latest.at)) {
            latest = candidate;
        }
    }
    return latest;
}
function summarizeWatchPackTransition(input) {
    const beforeActivePackId = input.before?.active?.packId ?? input.before?.pointers.active?.packId ?? null;
    const afterActivePackId = input.after?.active?.packId ?? input.after?.pointers.active?.packId ?? null;
    if (afterActivePackId !== null && beforeActivePackId !== afterActivePackId) {
        return {
            kind: "promoted_active",
            fromPackId: beforeActivePackId,
            toPackId: afterActivePackId
        };
    }
    const beforeCandidatePackId = input.before?.candidate?.packId ?? input.before?.pointers.candidate?.packId ?? null;
    const afterCandidatePackId = input.after?.candidate?.packId ?? input.after?.pointers.candidate?.packId ?? null;
    if (afterCandidatePackId !== null && beforeCandidatePackId !== afterCandidatePackId) {
        return {
            kind: "staged_candidate",
            fromPackId: beforeCandidatePackId,
            toPackId: afterCandidatePackId
        };
    }
    return null;
}
function truncateWatchMessage(text, maxLength = 96) {
    const normalized = text.replace(/\s+/gu, " ").trim();
    if (normalized.length <= maxLength) {
        return normalized;
    }
    return `${normalized.slice(0, maxLength - 1)}…`;
}
function buildWatchLastObservedDelta(input) {
    const exported = input.exported.exportedBundleCount > 0 ||
        input.exported.exportedEventCount > 0;
    const labeled = (input.snapshotAfter.diagnostics.emittedArtifactCount ?? 0) >
        (input.snapshotBefore.diagnostics.emittedArtifactCount ?? 0);
    const latestPackTransition = summarizeWatchPackTransition({
        before: input.beforeInspection,
        after: input.afterInspection
    });
    const promoted = latestPackTransition?.kind === "promoted_active";
    const afterActivePackId = input.afterInspection?.active?.packId ?? input.afterInspection?.pointers.active?.packId ?? null;
    const served = promoted && afterActivePackId === latestPackTransition?.toPackId && input.afterInspection?.active?.activationReady === true;
    const latestUserMessage = summarizeWatchLatestUserMessage(input.localPoll);
    const selectedBackfillOnly = !exported && input.scanResult.selected.length > 0;
    const cycleDidNothing = !exported && !labeled && !promoted && !served;
    let explanation;
    if (latestUserMessage === null) {
        if (selectedBackfillOnly) {
            explanation =
                "No new local user message was exported in this cycle; the learner only revisited stored exports, so this pass does not prove a new last-turn change.";
        }
        else if (cycleDidNothing) {
            explanation = "No new local user message or learner-visible export was observed in this cycle, so nothing changed.";
        }
        else if (promoted && latestPackTransition !== null) {
            explanation =
                `No new local user message was exported in this cycle; pack ${latestPackTransition.toPackId} moved into ${latestPackTransition.kind === "promoted_active" ? "active serving" : "the candidate slot"} from previously accumulated learner state.`;
        }
        else {
            explanation =
                "This cycle observed learner activity, but it did not include a new local user message, so the latest last-turn delta cannot be attributed to a fresh user turn.";
        }
    }
    else {
        const quotedMessage = `"${truncateWatchMessage(latestUserMessage.text)}"`;
        if (exported && labeled && promoted && served && latestPackTransition !== null) {
            explanation =
                `Latest user message ${quotedMessage} was exported, labeled, promoted into pack ${latestPackTransition.toPackId}, and is now served from the active pack.`;
        }
        else if (exported && labeled && !promoted) {
            explanation =
                `Latest user message ${quotedMessage} was exported and labeled, but it has not been promoted into the serving pack yet.`;
        }
        else if (exported && !labeled && !promoted) {
            explanation =
                `Latest user message ${quotedMessage} was exported, but it did not add a new teacher label or change the serving pack in this cycle.`;
        }
        else if (exported && !labeled && promoted && latestPackTransition !== null) {
            explanation =
                `Latest user message ${quotedMessage} was exported, but this cycle's promotion to pack ${latestPackTransition.toPackId} is not backed by a new teacher label from that message alone.`;
        }
        else if (!exported && labeled) {
            explanation =
                `Latest user message ${quotedMessage} was already in stored exports; this cycle only labeled or replayed it, without a new local export.`;
        }
        else if (cycleDidNothing) {
            explanation = `Latest user message ${quotedMessage} did not produce a new export, label, or serving-pack change in this cycle.`;
        }
        else {
            explanation =
                `Latest user message ${quotedMessage} changed learner state this cycle, but the local artifacts do not prove a clean export-to-serve handoff yet.`;
        }
    }
    return {
        available: true,
        observedAt: input.observedAt,
        exported,
        labeled,
        promoted,
        served,
        latestPackTransition,
        explanation
    };
}
export async function createWatchCommandRuntime(input) {
    const activationRoot = path.resolve(input.activationRoot);
    const bootstrapObservedAt = new Date().toISOString();
    const scanRoot = input.scanRoot !== undefined && input.scanRoot !== null
        ? path.resolve(input.scanRoot)
        : path.resolve(activationRoot, "event-exports");
    const pollIntervalSeconds = Number.isInteger(input.pollIntervalSeconds) && (input.pollIntervalSeconds ?? 0) > 0
        ? input.pollIntervalSeconds
        : DEFAULT_WATCH_POLL_INTERVAL_SECONDS;
    const sessionTailCursorPath = resolveWatchSessionTailCursorPath(activationRoot);
    const teacherSnapshotPath = resolveWatchTeacherSnapshotPath(activationRoot);
    const restoredTeacherState = loadWatchTeacherSnapshotState(teacherSnapshotPath);
    const log = input.log ?? watchLog;
    const startupWarnings = [];
    mkdirSync(scanRoot, { recursive: true });
    mkdirSync(resolveWatchStateRoot(activationRoot), { recursive: true });
    log(`Watch starting — activation: ${shortenPath(activationRoot)}`);
    log(`Scan root: ${shortenPath(scanRoot)}`);
    log(`State: cursor=${shortenPath(sessionTailCursorPath)} snapshot=${shortenPath(teacherSnapshotPath)}`);
    const resolvedTeacherLabeler = resolveWatchTeacherLabelerConfig(input.teacherLabeler, activationRoot);
    const resolvedEmbedder = resolveWatchEmbedderConfig(input.embedder, activationRoot);
    const teacherLabeler = resolvedTeacherLabeler.teacherLabeler;
    for (const warning of resolvedTeacherLabeler.warnings) {
        startupWarnings.push(`teacher_config_warning:${warning}`);
        log(`Teacher config warning: ${warning}`);
    }
    for (const warning of resolvedEmbedder.warnings) {
        startupWarnings.push(`embedder_config_warning:${warning}`);
        log(`Embedder config warning: ${warning}`);
    }
    if (teacherLabeler?.provider === "ollama") {
        log(`Teacher labeler: provider=ollama model=${teacherLabeler.model ?? "qwen3.5:9b"}`);
    }
    if (resolvedEmbedder.embedder !== null) {
        log(`Embedder: provider=ollama model=${resolvedEmbedder.embedder.model}`);
    }
    else {
        log("Embedder: numeric pack materialization is not configured; watch will keep keyword/weight vectors only.");
    }
    const scanner = createRuntimeEventExportScanner({ scanRoot });
    let lastServeTimeFallbackReason = null;
    const baseTeacherLoopInput = {
        packLabel: "watch-cli",
        workspace: {
            workspaceId: "watch-cli",
            snapshotId: `watch-cli@${new Date().toISOString().slice(0, 10)}`,
            capturedAt: new Date().toISOString(),
            rootDir: activationRoot,
            revision: "watch-cli-v2"
        },
        learnedRouting: true,
        ...(teacherLabeler !== null ? { teacherLabeler } : {}),
        resolveLearnedRoutingState: () => {
            const resolved = resolveServeTimeLearningRuntimeInput(activationRoot);
            if (resolved.fallbackReason !== null && resolved.fallbackReason !== lastServeTimeFallbackReason) {
                log(`Serve-time routing fallback: ${resolved.fallbackReason}`);
            }
            lastServeTimeFallbackReason = resolved.fallbackReason;
            return {
                pgVersion: resolved.pgVersion,
                ...(resolved.decisionLogCount > 0 ? { serveTimeDecisions: resolved.serveTimeDecisions } : {}),
                ...(resolved.baselineState !== undefined ? { baselineState: resolved.baselineState } : {})
            };
        },
        persistUpdatedBaseline: (state) => {
            try {
                persistBaseline(activationRoot, state);
            }
            catch (error) {
                log(`Baseline persist failed: ${formatWatchError(error)}`);
            }
        }
    };
    let teacherLoop;
    let lastHandledMaterializationPackId = restoredTeacherState.lastHandledMaterializationPackId;
    let lastEmbedInstrumentation = restoredTeacherState.embedInstrumentation;
    let restoredLastObservedDelta = restoredTeacherState.lastObservedDelta;
    if (restoredTeacherState.error !== null) {
        const message = restoredTeacherState.error;
        startupWarnings.push(`teacher_snapshot_reset:${message}`);
        lastHandledMaterializationPackId = null;
        lastEmbedInstrumentation = null;
        restoredLastObservedDelta = {
            available: true,
            observedAt: bootstrapObservedAt,
            exported: false,
            labeled: false,
            promoted: false,
            served: false,
            latestPackTransition: null,
            explanation: "Watch reset an unreadable teacher snapshot, so no prior last-turn delta can be trusted."
        };
        log(`Teacher snapshot reset: ${message}`);
        teacherLoop = createAsyncTeacherLiveLoop(baseTeacherLoopInput);
    }
    else {
        try {
            teacherLoop = createAsyncTeacherLiveLoop({
                ...baseTeacherLoopInput,
                ...(restoredTeacherState.snapshot !== null ? { resumeFromSnapshot: restoredTeacherState.snapshot } : {})
            });
        }
        catch (error) {
            const message = formatWatchError(error);
            startupWarnings.push(`teacher_snapshot_reset:${message}`);
            lastHandledMaterializationPackId = null;
            lastEmbedInstrumentation = null;
            restoredLastObservedDelta = {
                available: true,
                observedAt: bootstrapObservedAt,
                exported: false,
                labeled: false,
                promoted: false,
                served: false,
                latestPackTransition: null,
                explanation: "Watch reset an unusable teacher snapshot, so no prior last-turn delta can be trusted."
            };
            log(`Teacher snapshot reset: ${message}`);
            teacherLoop = createAsyncTeacherLiveLoop(baseTeacherLoopInput);
        }
    }
    if (restoredTeacherState.snapshot !== null && startupWarnings.length === 0) {
        const restoredSeenExportCount = restoredTeacherState.snapshot.state?.seenExportDigests.length ?? 0;
        log(`Restored teacher snapshot: seen=${restoredSeenExportCount} artifacts=${restoredTeacherState.snapshot.teacher.artifactCount}`);
    }
    const resolvedWatchProfileScope = input.profileRoots === undefined
        ? resolveWatchProfileRootsForActivationRoot(activationRoot)
        : {
            attachedProfileRoots: [...new Set(input.profileRoots.map((root) => path.resolve(root)))],
            halfAttachedReferences: []
        };
    const resolvedProfileRoots = resolvedWatchProfileScope.attachedProfileRoots;
    if (input.profileRoots === undefined && resolvedProfileRoots !== undefined && resolvedProfileRoots.length > 0) {
        log(`Session tail scope: attached OpenClaw home${resolvedProfileRoots.length === 1 ? "" : "s"} ${resolvedProfileRoots
            .map((root) => shortenPath(root))
            .join(", ")}`);
    }
    if (input.profileRoots === undefined && resolvedWatchProfileScope.halfAttachedReferences.length > 0) {
        log(`Session tail scope skipped half-attached OpenClaw home${resolvedWatchProfileScope.halfAttachedReferences.length === 1 ? "" : "s"} ${resolvedWatchProfileScope.halfAttachedReferences
            .map((reference) => `${shortenPath(path.resolve(reference.openclawHome))} (${summarizeSharedActivationRootReferenceProof(reference)})`)
            .join(", ")}`);
    }
    let restoredCursor = loadWatchSessionTailCursor(sessionTailCursorPath);
    let localSessionTail;
    try {
        localSessionTail = createOpenClawLocalSessionTail({
            ...(resolvedProfileRoots === undefined ? {} : { profileRoots: resolvedProfileRoots }),
            cursor: restoredCursor,
            emitExistingOnFirstPoll: restoredCursor.length === 0
        });
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        log(`Session tail cursor reset: ${message}`);
        restoredCursor = [];
        localSessionTail = createOpenClawLocalSessionTail({
            ...(resolvedProfileRoots === undefined ? {} : { profileRoots: resolvedProfileRoots }),
            emitExistingOnFirstPoll: true
        });
        persistWatchSessionTailCursor(sessionTailCursorPath, []);
    }
    let replayState = {
        replayedBundleCount: 0,
        replayedEventCount: 0
    };
    try {
        replayState = await replayWatchScanRootIntoTeacherLoop(teacherLoop, scanRoot);
    }
    catch (error) {
        const message = formatWatchError(error);
        startupWarnings.push(`teacher_replay_failed:${message}`);
        log(`Async teacher replay fail-open: ${message}`);
    }
    if (replayState.replayedBundleCount > 0) {
        log(`Replayed ${replayState.replayedBundleCount} stored export bundle${replayState.replayedBundleCount === 1 ? "" : "s"} (${replayState.replayedEventCount} event${replayState.replayedEventCount === 1 ? "" : "s"})`);
    }
    let bootstrapSnapshot = teacherLoop.snapshot();
    const replayPromotion = await applyWatchMaterialization(activationRoot, bootstrapSnapshot, lastHandledMaterializationPackId, resolvedEmbedder.embedder, log);
    lastHandledMaterializationPackId = replayPromotion.lastHandledMaterializationPackId;
    if (replayPromotion.embedInstrumentation !== null) {
        lastEmbedInstrumentation = replayPromotion.embedInstrumentation;
    }
    if (replayPromotion.logLine !== null) {
        log(replayPromotion.logLine);
        bootstrapSnapshot = teacherLoop.snapshot();
    }
    const bootstrapCursor = localSessionTail.snapshot();
    persistWatchTeacherSnapshot(teacherSnapshotPath, {
        lastRunAt: bootstrapObservedAt,
        pollIntervalSeconds,
        scanRoot,
        sessionTailCursorPath,
        sessionTailCursorUpdatedAt: bootstrapObservedAt,
        sessionTailSessionsTracked: bootstrapCursor.length,
        sessionTailBridgedEventCount: countWatchCursorBridgedEvents(bootstrapCursor),
        scannerCheckpointPath: scanner.checkpointPath,
        scannerCheckpoint: scanner.snapshot(),
        replayedBundleCount: replayState.replayedBundleCount,
        replayedEventCount: replayState.replayedEventCount,
        exportedBundleCount: 0,
        exportedEventCount: 0,
        startupWarnings,
        lastTeacherError: null,
        localSessionTailNoopReason: null,
        lastHandledMaterializationPackId,
        lastObservedDelta: restoredLastObservedDelta.available
            ? restoredLastObservedDelta
            : {
                available: true,
                observedAt: bootstrapObservedAt,
                exported: false,
                labeled: false,
                promoted: false,
                served: false,
                latestPackTransition: null,
                explanation: "Watch bootstrapped its state, but no new local user-message delta has been observed yet."
            },
        embedInstrumentation: lastEmbedInstrumentation,
        failure: replayPromotion.failure,
        snapshot: bootstrapSnapshot
    });
    return {
        activationRoot,
        scanRoot,
        pollIntervalSeconds,
        sessionTailCursorPath,
        teacherSnapshotPath,
        startupWarnings,
        lastTeacherError: null,
        replayState,
        lastHandledMaterializationPackId,
        lastEmbedInstrumentation,
        scanner,
        teacherLoop,
        localSessionTail,
        embedder: resolvedEmbedder.embedder
    };
}
export async function runWatchCommandPass(runtime, options = {}) {
    const log = options.log ?? watchLog;
    const observedAt = options.observedAt ?? new Date().toISOString();
    const snapshotBefore = runtime.teacherLoop.snapshot();
    const beforeInspection = inspectActivationState(runtime.activationRoot, observedAt);
    const localPoll = runtime.localSessionTail.pollOnce({
        observedAt
    });
    const scannerCheckpointBeforeScan = runtime.scanner.snapshot();
    const exported = exportLocalSessionTailChangesToScanRoot({
        scanRoot: runtime.scanRoot,
        polledAt: localPoll.polledAt,
        changes: localPoll.changes
    });
    persistWatchSessionTailCursor(runtime.sessionTailCursorPath, localPoll.cursor);
    for (const warning of [...localPoll.warnings, ...exported.warnings]) {
        log(`Session tail warning: ${warning}`);
    }
    if (exported.exportedBundleCount > 0) {
        log(`Session tail exported ${exported.exportedBundleCount} bundle${exported.exportedBundleCount === 1 ? "" : "s"} from ${localPoll.changes.length} changed session${localPoll.changes.length === 1 ? "" : "s"}`);
    }
    const scanResult = runtime.scanner.scanOnce({
        scannedAt: observedAt
    });
    const totalSelected = scanResult.selected.length;
    const totalEvents = scanResult.selected.reduce((sum, hit) => sum + hit.eventRange.count, 0);
    let snapshot = runtime.teacherLoop.snapshot();
    let materializedPackId = null;
    let failure = null;
    if (totalSelected === 0) {
        log("Scanning... no changes");
    }
    else {
        log(`Scanning... ${totalSelected} export bundle${totalSelected === 1 ? "" : "s"} selected, ${totalEvents} event${totalEvents === 1 ? "" : "s"}`);
        try {
            const ingestResult = await runtime.teacherLoop.ingestRuntimeEventExportScannerScan(scanResult);
            runtime.lastTeacherError = null;
            snapshot = ingestResult.snapshot;
            const promotion = await applyWatchMaterialization(runtime.activationRoot, snapshot, runtime.lastHandledMaterializationPackId, runtime.embedder, log);
            runtime.lastHandledMaterializationPackId = promotion.lastHandledMaterializationPackId;
            materializedPackId = promotion.materializedPackId;
            if (promotion.embedInstrumentation !== null) {
                runtime.lastEmbedInstrumentation = promotion.embedInstrumentation;
            }
            failure = promotion.failure;
            if (promotion.logLine !== null) {
                log(promotion.logLine);
                snapshot = runtime.teacherLoop.snapshot();
            }
        }
        catch (error) {
            const message = formatWatchError(error);
            runtime.lastTeacherError = message;
            failure = {
                mode: "teacher_fail_open",
                detail: message,
                at: observedAt
            };
            log(`Async teacher fail-open: ${message}`);
            try {
                runtime.scanner.restoreCheckpoint(scannerCheckpointBeforeScan);
            }
            catch (restoreError) {
                const restoreMessage = formatWatchError(restoreError);
                runtime.lastTeacherError = `${message}; scanner checkpoint restore failed: ${restoreMessage}`;
                failure = {
                    mode: "teacher_fail_open",
                    detail: runtime.lastTeacherError,
                    at: observedAt
                };
                log(`Scanner checkpoint restore failed: ${restoreMessage}`);
            }
            snapshot = runtime.teacherLoop.snapshot();
        }
    }
    const afterInspection = inspectActivationState(runtime.activationRoot, observedAt);
    const lastObservedDelta = buildWatchLastObservedDelta({
        observedAt,
        localPoll,
        exported,
        scanResult,
        snapshotBefore,
        snapshotAfter: snapshot,
        beforeInspection,
        afterInspection
    });
    persistWatchTeacherSnapshot(runtime.teacherSnapshotPath, {
        lastRunAt: observedAt,
        pollIntervalSeconds: runtime.pollIntervalSeconds,
        scanRoot: runtime.scanRoot,
        sessionTailCursorPath: runtime.sessionTailCursorPath,
        sessionTailCursorUpdatedAt: observedAt,
        sessionTailSessionsTracked: localPoll.cursor.length,
        sessionTailBridgedEventCount: countWatchCursorBridgedEvents(localPoll.cursor),
        scannerCheckpointPath: runtime.scanner.checkpointPath,
        scannerCheckpoint: runtime.scanner.snapshot(),
        replayedBundleCount: runtime.replayState.replayedBundleCount,
        replayedEventCount: runtime.replayState.replayedEventCount,
        exportedBundleCount: exported.exportedBundleCount,
        exportedEventCount: exported.exportedEventCount,
        startupWarnings: runtime.startupWarnings,
        lastTeacherError: runtime.lastTeacherError,
        localSessionTailNoopReason: localPoll.noopReason,
        lastHandledMaterializationPackId: runtime.lastHandledMaterializationPackId,
        lastObservedDelta,
        embedInstrumentation: runtime.lastEmbedInstrumentation,
        failure,
        snapshot
    });
    const persistedScannerCheckpoint = runtime.scanner.snapshot();
    if (options.json) {
        console.log(JSON.stringify({
            timestamp: observedAt,
            replayedBundles: runtime.replayState.replayedBundleCount,
            replayedEvents: runtime.replayState.replayedEventCount,
            exportedBundles: exported.exportedBundleCount,
            exportedEvents: exported.exportedEventCount,
            selected: totalSelected,
            events: totalEvents,
            live: scanResult.live.length,
            backfill: scanResult.backfill.length,
            sessionTailSessionsTracked: localPoll.cursor.length,
            sessionTailBridgedEvents: countWatchCursorBridgedEvents(localPoll.cursor),
            scannerProcessedBundles: persistedScannerCheckpoint.processedExportDigests.length,
            scannerLiveAfter: persistedScannerCheckpoint.live.after?.exportDigest ?? null,
            materialized: materializedPackId,
            lastObservedDelta,
            diagnostics: snapshot.diagnostics ?? null,
            localSessionTailNoopReason: localPoll.noopReason
        }));
    }
    return {
        localPoll,
        exported,
        scanResult,
        snapshot,
        materializedPackId
    };
}
async function runWatchCommand(parsed) {
    const intervalMs = parsed.interval * 1000;
    const runtime = await createWatchCommandRuntime({
        activationRoot: parsed.activationRoot,
        scanRoot: parsed.scanRoot,
        pollIntervalSeconds: parsed.interval,
        log: watchLog
    });
    watchLog(`Interval: ${parsed.interval}s`);
    let stopping = false;
    const onSignal = () => {
        if (stopping) {
            process.exit(1);
        }
        stopping = true;
        watchLog("Stopping... (Ctrl+C again to force)");
    };
    process.on("SIGINT", onSignal);
    process.on("SIGTERM", onSignal);
    while (!stopping) {
        try {
            await runWatchCommandPass(runtime, {
                json: parsed.json,
                log: watchLog
            });
        }
        catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            watchLog(`Error: ${message}`);
        }
        const deadline = Date.now() + intervalMs;
        while (!stopping && Date.now() < deadline) {
            await new Promise((resolve) => {
                setTimeout(resolve, Math.min(1000, deadline - Date.now()));
            });
        }
    }
    watchLog("Watch stopped.");
    process.removeListener("SIGINT", onSignal);
    process.removeListener("SIGTERM", onSignal);
    return 0;
}
function promptSyncLine(prompt) {
    process.stdout.write(prompt);
    const buf = Buffer.alloc(256);
    let input = "";
    const fd = openSync("/dev/tty", "r");
    try {
        const bytesRead = readSync(fd, buf, 0, buf.length, null);
        input = buf.toString("utf8", 0, bytesRead).replace(/\r?\n$/, "");
    }
    finally {
        closeSync(fd);
    }
    return input;
}
function resetActivationRoot(activationRoot) {
    const resolvedRoot = path.resolve(activationRoot);
    const removedPacks = [];
    const packsDir = path.join(resolvedRoot, "packs");
    if (existsSync(packsDir)) {
        try {
            const entries = readdirSync(packsDir);
            for (const entry of entries) {
                const packPath = path.join(packsDir, entry);
                rmSync(packPath, { recursive: true, force: true });
                removedPacks.push(entry);
            }
        }
        catch {
            // packs dir may not be readable
        }
    }
    const logsDir = path.join(resolvedRoot, "logs");
    if (existsSync(logsDir)) {
        rmSync(logsDir, { recursive: true, force: true });
    }
    const seedPointers = {
        contract: "activation_pointers.v1",
        active: null,
        candidate: null,
        previous: null
    };
    const pointersPath = path.join(resolvedRoot, "activation-pointers.json");
    mkdirSync(resolvedRoot, { recursive: true });
    writeFileSync(pointersPath, JSON.stringify(seedPointers, null, 2) + "\n", "utf8");
    return { removedPacks, pointersReset: true };
}
function runResetCommand(parsed) {
    if (parsed.help) {
        console.log([
            "Usage: openclawbrain reset [--activation-root <path>|--openclaw-home <path>] [--yes] [--json]",
            "",
            "Wipes all learned state and returns the brain to seed state.",
            "",
            "Options:",
            "  --activation-root <path>  Activation root (auto-detected if omitted)",
            "  --openclaw-home <path>   Pin auto-detection to one installed OpenClaw profile",
            "  --yes, -y                 Skip confirmation prompt",
            "  --json                    Emit machine-readable JSON output",
            "  --help                    Show this help"
        ].join("\n"));
        return 0;
    }
    const activationRoot = parsed.activationRoot;
    if (!existsSync(activationRoot)) {
        const msg = `Activation root does not exist: ${activationRoot}`;
        if (parsed.json) {
            console.log(JSON.stringify({ ok: false, error: msg }, null, 2));
        }
        else {
            console.error(msg);
        }
        return 1;
    }
    if (!parsed.yes) {
        let answer;
        try {
            answer = promptSyncLine("This will delete all learned context. Type 'reset' to confirm: ");
        }
        catch {
            console.error("Cannot prompt for confirmation in non-interactive mode. Use --yes to skip.");
            return 1;
        }
        if (answer.trim() !== "reset") {
            console.log("Reset cancelled.");
            return 1;
        }
    }
    const result = resetActivationRoot(activationRoot);
    if (parsed.json) {
        console.log(JSON.stringify({
            ok: true,
            activationRoot,
            removedPacks: result.removedPacks,
            pointersReset: result.pointersReset
        }, null, 2));
    }
    else {
        console.log("RESET complete\n");
        if (result.removedPacks.length > 0) {
            console.log(`  Removed ${result.removedPacks.length} pack(s): ${result.removedPacks.join(", ")}`);
        }
        else {
            console.log("  No packs to remove.");
        }
        console.log("  Activation pointers reset to seed state.");
        console.log(`\nBrain at ${shortenPath(activationRoot)} is now in seed state.`);
        console.log(`Run \`openclawbrain status --activation-root ${quoteShellArg(activationRoot)}\` to verify.`);
    }
    return 0;
}
export function runOperatorCli(argv = process.argv.slice(2)) {
    const parsed = parseOperatorCliArgs(argv);
    if (parsed.command === "context") {
        return runContextCommand(parsed);
    }
    if (parsed.command === "reset") {
        return runResetCommand(parsed);
    }
    if (parsed.help) {
        console.log(operatorCliHelp());
        return 0;
    }
    if (parsed.command === "export") {
        const result = exportBrain({
            activationRoot: parsed.activationRoot,
            outputPath: parsed.outputPath,
        });
        if (parsed.json) {
            console.log(JSON.stringify(result, null, 2));
        }
        else if (result.ok) {
            console.log(`EXPORT ok`);
            console.log(`  Archive: ${result.outputPath}`);
            console.log(`  Source:  ${result.activationRoot}`);
        }
        else {
            console.error(`EXPORT failed: ${result.error}`);
        }
        return result.ok ? 0 : 1;
    }
    if (parsed.command === "import") {
        const result = importBrain({
            archivePath: parsed.archivePath,
            activationRoot: parsed.activationRoot,
            force: parsed.force,
        });
        if (parsed.json) {
            console.log(JSON.stringify(result, null, 2));
        }
        else if (result.ok) {
            console.log(`IMPORT ok`);
            console.log(`  Activation root: ${result.activationRoot}`);
            console.log(`  Archive:         ${result.archivePath}`);
            if (result.warning) {
                console.log(`  Warning:         ${result.warning}`);
            }
        }
        else {
            console.error(`IMPORT failed: ${result.error}`);
        }
        return result.ok ? 0 : 1;
    }
    if (parsed.command === "daemon") {
        return runDaemonCommand(parsed);
    }
    if (parsed.command === "history") {
        return runHistoryCommand(parsed);
    }
    if (parsed.command === "learn") {
        return runLearnCommand(parsed);
    }
    if (parsed.command === "watch") {
        // Watch is async — bridge to sync CLI entry by scheduling and returning 0.
        // The process stays alive due to the interval loop and exits via SIGINT or error.
        runWatchCommand(parsed).then((code) => { process.exitCode = code; }, (error) => {
            console.error("[openclawbrain] watch failed");
            console.error(error instanceof Error ? error.stack ?? error.message : String(error));
            process.exitCode = 1;
        });
        return 0;
    }
    if (parsed.command === "install") {
        return runInstallCommand(parsed);
    }
    if (parsed.command === "detach") {
        return runDetachCommand(parsed);
    }
    if (parsed.command === "uninstall") {
        return runUninstallCommand(parsed);
    }
    if (parsed.command === "attach") {
        return runAttachCommand(parsed);
    }
    if (parsed.command === "scan") {
        if (parsed.sessionPath !== null) {
            const result = scanRecordedSession({
                rootDir: parsed.rootDir,
                trace: readJsonFile(parsed.sessionPath)
            });
            if (parsed.json) {
                console.log(JSON.stringify(result, null, 2));
            }
            else {
                console.log(formatScanSessionSummary(result));
            }
            return 0;
        }
        const result = scanLiveEventExport({
            normalizedEventExport: loadCliScanLiveExport(parsed.livePath),
            workspace: readJsonFile(parsed.workspacePath),
            ...(parsed.packLabel === null ? {} : { packLabel: parsed.packLabel }),
            ...(parsed.observedAt === null ? {} : { observedAt: parsed.observedAt })
        });
        const snapshotOutPath = parsed.snapshotOutPath === null ? null : path.resolve(parsed.snapshotOutPath);
        if (snapshotOutPath !== null) {
            mkdirSync(path.dirname(snapshotOutPath), { recursive: true });
            writeFileSync(snapshotOutPath, JSON.stringify(result.snapshot, null, 2), "utf8");
        }
        if (parsed.json) {
            console.log(JSON.stringify({ ...result, snapshotOutPath }, null, 2));
        }
        else {
            console.log(formatScanLiveSummary(result, snapshotOutPath));
        }
        return 0;
    }
    // At this point only status/rollback commands remain
    const statusOrRollback = parsed;
    const activationRoot = requireActivationRoot(statusOrRollback.input, statusOrRollback.openclawHome, statusOrRollback.command);
    const targetInspection = statusOrRollback.openclawHome === null ? null : inspectOpenClawHome(statusOrRollback.openclawHome);
    if (statusOrRollback.command === "rollback") {
        const result = rollbackRuntimeAttach({
            activationRoot,
            ...(statusOrRollback.input.updatedAt === null ? {} : { updatedAt: statusOrRollback.input.updatedAt }),
            dryRun: statusOrRollback.dryRun
        });
        if (statusOrRollback.json) {
            console.log(JSON.stringify(result, null, 2));
        }
        else {
            console.log(formatOperatorRollbackReport(result));
        }
        return result.allowed ? 0 : 1;
    }
    const operatorInput = {
        ...statusOrRollback.input,
        activationRoot,
        openclawHome: statusOrRollback.openclawHome,
        ...(targetInspection?.profileId === null || targetInspection?.profileId === undefined
            ? {}
            : { profileId: targetInspection.profileId }),
        teacherSnapshotPath: resolveOperatorTeacherSnapshotPath(activationRoot, statusOrRollback.input.teacherSnapshotPath)
    };
    const status = describeCurrentProfileBrainStatus(operatorInput);
    const tracedLearning = buildTracedLearningStatusSurface(activationRoot);
    const normalizedStatusAndReport = applyAttachmentPolicyTruth(status, statusOrRollback.json ? null : buildOperatorSurfaceReport(operatorInput));
    if (statusOrRollback.json) {
        console.log(JSON.stringify({
            ...normalizedStatusAndReport.status,
            tracedLearning
        }, null, 2));
    }
    else {
        const report = normalizedStatusAndReport.report;
        const providerConfig = readOpenClawBrainProviderConfigFromSources({
            env: process.env,
            activationRoot
        });
        if (statusOrRollback.detailed) {
            console.log(formatCurrentProfileStatusSummary(normalizedStatusAndReport.status, report, targetInspection, {
                openclawHome: statusOrRollback.openclawHome,
                providerConfig,
                tracedLearning
            }));
        }
        else {
            console.log(formatHumanFriendlyStatus(normalizedStatusAndReport.status, report, targetInspection, {
                openclawHome: statusOrRollback.openclawHome,
                providerConfig,
                tracedLearning
            }));
        }
    }
    return 0;
}
if (isDirectCliRun(process.argv[1], import.meta.url)) {
    try {
        process.exitCode = runOperatorCli();
    }
    catch (error) {
        console.error("[openclawbrain] failed");
        console.error(error instanceof Error ? error.stack ?? error.message : String(error));
        process.exitCode = 1;
    }
}
//# sourceMappingURL=cli.js.map
