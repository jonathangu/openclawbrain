import type { BrainConfig } from "../brain-core/types.js";
import type { OpenClawBrainRuntimeConfig } from "../db/config.js";
import type { BrainStore } from "../brain-store/store.js";

export type WorkerLastExit = {
  code: number | null;
  signal: NodeJS.Signals | null;
  at: number;
};

export type WorkerRuntimeState = {
  workerMode: BrainConfig["workerMode"];
  workerModeDevOnly: boolean;
  workerModeWarning: string | null;
  workerPid: number | null;
  workerStatus: string | null;
  workerLastHeartbeatAt: number | null;
  workerLastReadyAt: number | null;
  workerHealthy: boolean;
  workerLastExit: WorkerLastExit | null;
  workerRestartCount: number;
  workerLastRestartAt: number | null;
  workerStartedAt: number | null;
  workerLastReloadRequestedAt: number | null;
  workerLastReloadAckAt: number | null;
  workerLastTickAt: number | null;
  workerLastTickResultAt: number | null;
  workerLastTickOk: boolean | null;
  workerLastTickError: string | null;
  workerLastFatalError: string | null;
};

function readInt(store: BrainStore, key: string): number | null {
  const raw = store.getTrainingState(key)?.trim();
  if (!raw) {
    return null;
  }
  const value = Number.parseInt(raw, 10);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function readString(store: BrainStore, key: string): string | null {
  const raw = store.getTrainingState(key)?.trim() ?? "";
  return raw.length > 0 ? raw : null;
}

function readBoolean(store: BrainStore, key: string): boolean | null {
  const raw = readString(store, key);
  if (raw === "true") {
    return true;
  }
  if (raw === "false") {
    return false;
  }
  return null;
}

type WorkerStateConfig = Pick<BrainConfig, "workerMode" | "workerHeartbeatTimeoutMs">
  | Pick<OpenClawBrainRuntimeConfig, "workerMode" | "workerHeartbeatTimeoutMs">;

export function readWorkerRuntimeState(
  store: BrainStore,
  config: WorkerStateConfig,
): WorkerRuntimeState {
  const workerMode = config.workerMode ?? "child";
  const workerHeartbeatTimeoutMs = config.workerHeartbeatTimeoutMs ?? 90_000;
  const workerPid = readInt(store, "worker_pid");
  const workerLastHeartbeatAt = readInt(store, "worker_last_heartbeat_at");
  const workerLastReadyAt = readInt(store, "worker_last_ready_at");
  const workerLastExitAt = readInt(store, "worker_last_exit_at");
  const workerLastExitCodeRaw = store.getTrainingState("worker_last_exit_code")?.trim() ?? "";
  const workerLastExitSignal = readString(store, "worker_last_exit_signal");
  const workerLastExit = workerLastExitAt
    ? {
        code: workerLastExitCodeRaw.length > 0 ? Number.parseInt(workerLastExitCodeRaw, 10) : null,
        signal: workerLastExitSignal as NodeJS.Signals | null,
        at: workerLastExitAt,
      }
    : null;

  return {
    workerMode,
    workerModeDevOnly: workerMode === "in_process",
    workerModeWarning: workerMode === "in_process"
      ? "in_process worker mode is dev-only; use child mode for production operator truth"
      : null,
    workerPid,
    workerStatus: readString(store, "worker_status") ?? (workerMode === "child" ? "unknown" : "running"),
    workerLastHeartbeatAt,
    workerLastReadyAt,
    workerHealthy: workerMode === "child"
      ? Boolean(workerLastHeartbeatAt && (Date.now() - workerLastHeartbeatAt) < workerHeartbeatTimeoutMs)
      : true,
    workerLastExit,
    workerRestartCount: readInt(store, "worker_restart_count") ?? 0,
    workerLastRestartAt: readInt(store, "worker_last_restart_at"),
    workerStartedAt: readInt(store, "worker_started_at"),
    workerLastReloadRequestedAt: readInt(store, "worker_last_reload_requested_at"),
    workerLastReloadAckAt: readInt(store, "worker_last_reload_ack_at"),
    workerLastTickAt: readInt(store, "worker_last_tick_at"),
    workerLastTickResultAt: readInt(store, "worker_last_tick_result_at"),
    workerLastTickOk: readBoolean(store, "worker_last_tick_ok"),
    workerLastTickError: readString(store, "worker_last_tick_error"),
    workerLastFatalError: readString(store, "worker_last_fatal_error"),
  };
}
