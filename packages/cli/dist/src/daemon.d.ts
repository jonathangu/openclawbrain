/**
 * macOS launchd daemon management for OpenClawBrain.
 *
 * Manages macOS launchd user agents that run `openclawbrain watch` in the background.
 * Service identity is derived per activation root so one profile/service boundary
 * does not collide with another.
 *
 * Commands:
 *   daemon start  — generate and load a launchd plist
 *   daemon stop   — unload the plist
 *   daemon status — show running/stopped + PID + launch command + last log lines
 *   daemon logs   — tail the daemon log file
 */
type DaemonCommandRunner = (command: string) => string;
export interface DaemonServiceIdentity {
    requestedActivationRoot: string;
    canonicalActivationRoot: string;
    activationRootHash: string;
    activationRootSlug: string;
    label: string;
    plistFilename: string;
    plistPath: string;
    logPath: string;
}
export declare function buildDaemonServiceIdentity(activationRoot: string): DaemonServiceIdentity;
export declare function setDaemonCommandRunnerForTesting(runner: DaemonCommandRunner | null): void;
export type DaemonSubcommand = "start" | "stop" | "status" | "logs";
export interface DaemonCliArgs {
    command: "daemon";
    subcommand: DaemonSubcommand;
    activationRoot: string;
    json: boolean;
    help: boolean;
}
export interface ManagedLearnerServiceInspection {
    requestedActivationRoot: string;
    canonicalActivationRoot: string;
    serviceLabel: string;
    plistPath: string;
    logPath: string;
    installed: boolean;
    running: boolean;
    pid: number | null;
    configuredActivationRoot: string | null;
    configuredProgramArguments: string[] | null;
    configuredCommand: string | null;
    configuredRuntimePath: string | null;
    configuredRuntimePackageSpec: string | null;
    configuredRuntimeLooksEphemeral: boolean | null;
    matchesRequestedActivationRoot: boolean | null;
    launchctlAvailable: boolean;
}
export interface ManagedLearnerServiceEnsureResult {
    state: "started" | "ensured" | "deferred";
    reason: "started_exact_root" | "already_running_exact_root" | "launchctl_unavailable" | "launch_command_unavailable" | "launch_failed";
    detail: string;
    inspection: ManagedLearnerServiceInspection;
}
export interface ManagedLearnerServiceRemovalResult {
    state: "removed" | "preserved" | "already_absent";
    reason: "removed_exact_root" | "not_installed" | "configured_root_mismatch" | "launchctl_unavailable" | "stop_failed";
    detail: string;
    inspection: ManagedLearnerServiceInspection;
}
export declare function inspectManagedLearnerService(activationRoot: string): ManagedLearnerServiceInspection;
export declare function ensureManagedLearnerServiceForActivationRoot(activationRoot: string): ManagedLearnerServiceEnsureResult;
export declare function removeManagedLearnerServiceForActivationRoot(activationRoot: string): ManagedLearnerServiceRemovalResult;
export declare function daemonStart(activationRoot: string, json: boolean): number;
export declare function daemonStop(activationRoot: string, json: boolean): number;
export declare function daemonStatus(activationRoot: string, json: boolean): number;
export declare function daemonLogs(activationRoot: string, json: boolean): number;
export declare function runDaemonCommand(args: DaemonCliArgs): number;
export declare function daemonHelp(): string;
export declare function parseDaemonArgs(argv: readonly string[]): DaemonCliArgs;
export {};
