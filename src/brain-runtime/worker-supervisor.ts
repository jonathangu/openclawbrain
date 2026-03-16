import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import type { BrainConfig } from "../brain-core/types.js";
import type { BrainStore } from "../brain-store/store.js";
import type {
  ChildToParentMessage,
  ParentTeacherCompleteResultMessage,
  ParentToChildMessage,
  WorkerTeacherCompleteRequestMessage,
} from "../brain-worker/protocol.js";

export class WorkerSupervisor {
  private child: ChildProcess | null = null;
  private shouldRun = false;
  private restartTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private params: {
      config: BrainConfig;
      store: BrainStore;
      log: { info: (msg: string) => void; warn: (msg: string) => void; error: (msg: string) => void };
      teacherModel: { provider: string; model: string } | null;
      isEnabled: () => boolean;
      onPackPromoted: () => void;
      onTeacherComplete: (
        message: WorkerTeacherCompleteRequestMessage,
        teacherModel: { provider: string; model: string } | null,
      ) => Promise<ParentTeacherCompleteResultMessage>;
    },
  ) {}

  start(): void {
    if (!this.params.isEnabled()) {
      return;
    }
    this.shouldRun = true;
    this.ensureChildWorker(false);
  }

  stop(): void {
    this.shouldRun = false;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    if (this.child) {
      this.params.store.setTrainingState("worker_status", "stopping");
      this.send({ type: "shutdown" });
      const child = this.child;
      setTimeout(() => {
        if (this.child === child) {
          this.child.kill("SIGTERM");
        }
      }, 2_000);
    }
  }

  requestGraphReload(): void {
    this.params.store.setTrainingState("worker_last_reload_requested_at", Date.now());
    this.send({ type: "reload-graph" });
  }

  private ensureChildWorker(isRestart: boolean): void {
    if (this.child || !this.params.isEnabled()) {
      return;
    }

    const status = isRestart ? "restarting" : "starting";
    this.params.store.setTrainingState("worker_mode", "child");
    this.params.store.setTrainingState("worker_status", status);
    if (isRestart) {
      const nextCount = (Number.parseInt(this.params.store.getTrainingState("worker_restart_count") ?? "0", 10) || 0) + 1;
      const restartedAt = Date.now();
      this.params.store.setTrainingState("worker_restart_count", nextCount);
      this.params.store.setTrainingState("worker_last_restart_at", restartedAt);
    }

    const child = fork(
      fileURLToPath(new URL("../brain-worker/child-runner.ts", import.meta.url)),
      [],
      {
        execArgv: ["--import", "tsx/esm"],
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        env: {
          ...process.env,
          OPENCLAWBRAIN_CHILD_CONFIG_JSON: JSON.stringify(this.params.config),
          OPENCLAWBRAIN_CHILD_TEACHER_MODEL_JSON: this.params.teacherModel
            ? JSON.stringify(this.params.teacherModel)
            : "",
        },
      },
    );

    this.child = child;

    child.stdout?.on("data", (chunk) => {
      const text = String(chunk).trim();
      if (text) {
        this.params.log.info(text);
      }
    });
    child.stderr?.on("data", (chunk) => {
      const text = String(chunk).trim();
      if (text) {
        this.params.log.warn(text);
      }
    });
    child.on("message", (message) => {
      void this.handleMessage(message as ChildToParentMessage, child);
    });
    child.on("exit", (code, signal) => {
      const exitedAt = Date.now();
      this.params.store.setTrainingState("worker_pid", "");
      this.params.store.setTrainingState("worker_last_exit_at", exitedAt);
      this.params.store.setTrainingState("worker_last_exit_code", code === null ? "" : String(code));
      this.params.store.setTrainingState("worker_last_exit_signal", signal ?? "");
      if (this.child === child) {
        this.child = null;
      }
      const nextStatus = this.shouldRun && this.params.isEnabled() ? "restarting" : "stopped";
      this.params.store.setTrainingState("worker_status", nextStatus);
      if (this.shouldRun && this.params.isEnabled()) {
        this.restartTimer = setTimeout(() => {
          this.restartTimer = null;
          this.ensureChildWorker(true);
        }, this.params.config.workerRestartDelayMs);
      }
    });
  }

  private send(message: ParentToChildMessage): void {
    this.child?.send(message);
  }

  private async handleMessage(message: ChildToParentMessage, child: ChildProcess): Promise<void> {
    switch (message.type) {
      case "ready": {
        this.params.store.setTrainingState("worker_last_ready_at", message.at);
        this.params.store.setTrainingState("worker_last_fatal_error", "");
        return;
      }
      case "heartbeat": {
        return;
      }
      case "reload-graph-ack": {
        return;
      }
      case "tick-result": {
        return;
      }
      case "pack-promoted": {
        this.params.onPackPromoted();
        return;
      }
      case "teacher-complete": {
        const result = await this.params.onTeacherComplete(message, this.params.teacherModel);
        child.send?.(result);
        return;
      }
      case "fatal-error": {
        this.params.store.setTrainingState("worker_last_fatal_error", message.error);
        this.params.log.error(`[brain] child worker fatal error: ${message.error}`);
        return;
      }
      default:
        return;
    }
  }
}
