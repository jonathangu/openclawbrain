import process from "node:process";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { BrainGraph } from "../brain-core/graph.js";
import { BrainMutator } from "../brain-core/mutator.js";
import { PackManager } from "../brain-core/pack.js";
import { BrainTeacher, type BrainTeacherCompletion } from "../brain-core/teacher.js";
import type { BrainConfig } from "../brain-core/types.js";
import type { LcmDependencies } from "../types.js";
import { runBrainMigrations } from "../brain-store/migrations.js";
import { BrainStore } from "../brain-store/store.js";
import { promoteGraphSnapshot, reloadGraphFromStore } from "../brain-runtime/graph-io.js";
import type { ChildToParentMessage, ParentToChildMessage } from "./protocol.js";
import { BrainWorker } from "./worker.js";

function send(message: ChildToParentMessage): void {
  process.send?.(message);
}

function parseConfig(): BrainConfig {
  const raw = process.env.OPENCLAWBRAIN_CHILD_CONFIG_JSON;
  if (!raw) {
    throw new Error("OPENCLAWBRAIN_CHILD_CONFIG_JSON is required");
  }
  return JSON.parse(raw) as BrainConfig;
}

function parseResolvedTeacherModel(): { provider: string; model: string } | null {
  const raw = process.env.OPENCLAWBRAIN_CHILD_TEACHER_MODEL_JSON?.trim();
  if (!raw) {
    return null;
  }
  return JSON.parse(raw) as { provider: string; model: string };
}

function isPidAlive(pid: number): boolean {
  if (!Number.isFinite(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

class LeaseManager {
  private readonly leasePath: string;
  private readonly startedAt = Date.now();

  constructor(
    private config: BrainConfig,
    private store: BrainStore,
  ) {
    this.leasePath = join(config.root, "worker-lease.json");
  }

  acquire(): void {
    const existing = this.readLease();
    if (
      existing
      && existing.pid !== process.pid
      && isPidAlive(existing.pid)
      && (Date.now() - existing.heartbeatAt) < this.config.workerHeartbeatTimeoutMs
    ) {
      const heldPid = existing.pid;
      throw new Error(`worker lease already held by pid ${heldPid}`);
    }
    this.refresh("running");
    this.store.setTrainingState("worker_started_at", this.startedAt);
    this.store.setTrainingState("worker_mode", "child");
  }

  refresh(status = "running"): void {
    const now = Date.now();
    writeFileSync(this.leasePath, JSON.stringify({
      pid: process.pid,
      startedAt: this.startedAt,
      heartbeatAt: now,
      status,
    }, null, 2), "utf8");
    this.store.setTrainingState("worker_pid", process.pid);
    this.store.setTrainingState("worker_status", status);
    this.store.setTrainingState("worker_last_heartbeat_at", now);
    send({ type: "heartbeat", pid: process.pid, at: now, status });
  }

  release(status = "stopped"): void {
    this.store.setTrainingState("worker_status", status);
    this.store.setTrainingState("worker_last_heartbeat_at", Date.now());
    const existing = this.readLease();
    if (existing?.pid === process.pid && existsSync(this.leasePath)) {
      rmSync(this.leasePath, { force: true });
    }
  }

  private readLease(): { pid: number; heartbeatAt: number } | null {
    if (!existsSync(this.leasePath)) {
      return null;
    }
    try {
      const parsed = JSON.parse(readFileSync(this.leasePath, "utf8")) as { pid?: number; heartbeatAt?: number };
      return {
        pid: Number(parsed.pid ?? 0),
        heartbeatAt: Number(parsed.heartbeatAt ?? 0),
      };
    } catch {
      return null;
    }
  }
}

class IpcCompletionBridge {
  private pending = new Map<string, {
    resolve: (value: { content?: Array<{ text?: string }> }) => void;
    reject: (error: Error) => void;
  }>();

  handleMessage(message: ParentToChildMessage): boolean {
    if (message.type !== "teacher-complete-result") {
      return false;
    }
    const pending = this.pending.get(message.requestId);
    if (!pending) {
      return true;
    }
    this.pending.delete(message.requestId);
    if (!message.ok) {
      pending.reject(new Error(message.error));
      return true;
    }
    pending.resolve({
      content: (message.content ?? []).map((block) => ({ text: block.text })),
    });
    return true;
  }

  readonly complete: BrainTeacherCompletion = async (params) => {
    const requestId = `btc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    return await new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      send({
        type: "teacher-complete",
        requestId,
        provider: params.provider,
        model: params.model,
        messages: params.messages,
        system: params.system,
        maxTokens: params.maxTokens,
        temperature: params.temperature,
      });
    });
  };
}

async function main(): Promise<void> {
  const config = parseConfig();
  const resolvedTeacherModel = parseResolvedTeacherModel();

  const db = new DatabaseSync(join(config.root, "state.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA busy_timeout = 5000");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);

  const store = new BrainStore(db, { brainRoot: config.root });
  const graph = new BrainGraph();
  reloadGraphFromStore(store, graph);

  const log: LcmDependencies["log"] = {
    info: (msg) => process.stdout.write(`${msg}\n`),
    warn: (msg) => process.stderr.write(`${msg}\n`),
    error: (msg) => process.stderr.write(`${msg}\n`),
    debug: (msg) => process.stdout.write(`${msg}\n`),
  };

  const lease = new LeaseManager(config, store);
  lease.acquire();

  const bridge = new IpcCompletionBridge();
  const teacher = config.teacherEnabled && resolvedTeacherModel
    ? new BrainTeacher(
        bridge.complete,
        () => resolvedTeacherModel,
        async () => undefined,
        graph,
        log,
      )
    : null;

  const mutator = new BrainMutator(
    {
      insertNode: (node) => store.insertNode(node),
      insertEdge: (edge) => store.insertEdge(edge),
      deleteNode: (id) => store.deleteNode(id),
      deleteEdge: (source, target, kind) => store.deleteEdge(source, target, kind as never),
      resolveMutation: (id, status) => store.resolveMutation(id, status),
    },
    graph,
    log,
  );
  const packManager = new PackManager(
    {
      insertPack: (pack) => store.insertPack(pack),
      promotePack: (version) => store.promotePack(version),
      rollbackPack: (version) => store.rollbackPack(version),
    },
    graph,
    log,
  );

  const worker = new BrainWorker(
    store,
    graph,
    teacher,
    mutator,
    packManager,
    config,
    log,
    {
      isEnabled: () => !existsSync(join(config.root, "DISABLED")),
      onPromotionReady: async ({ healthJson }) => {
        const version = promoteGraphSnapshot({
          store,
          graph,
          packManager,
          config,
          reason: "worker",
          metadata: {
            healthJson,
            workerPid: process.pid,
          },
        });
        lease.refresh("running");
        send({ type: "pack-promoted", pid: process.pid, version });
      },
      onTickResult: ({ ok, at, error }) => {
        store.setTrainingState("worker_last_tick_result_at", at);
        store.setTrainingState("worker_last_tick_ok", ok ? "true" : "false");
        store.setTrainingState("worker_last_tick_error", error ?? "");
        send({ type: "tick-result", pid: process.pid, at, ok, error });
      },
    },
  );

  let shuttingDown = false;
  const heartbeatInterval = setInterval(() => {
    lease.refresh(shuttingDown ? "stopping" : "running");
  }, Math.max(1_000, Math.min(15_000, Math.floor(config.trainerIntervalMs / 2))));

  const cleanup = (status: string) => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    clearInterval(heartbeatInterval);
    worker.stop();
    lease.release(status);
  };

  process.on("message", async (message: ParentToChildMessage) => {
    if (bridge.handleMessage(message)) {
      return;
    }
    if (message.type === "reload-graph") {
      reloadGraphFromStore(store, graph);
      const reloadedAt = Date.now();
      store.setTrainingState("worker_last_reload_ack_at", reloadedAt);
      lease.refresh("running");
      send({
        type: "reload-graph-ack",
        pid: process.pid,
        at: reloadedAt,
        nodeCount: graph.nodeCount(),
        edgeCount: graph.edgeCount(),
      });
      return;
    }
    if (message.type === "shutdown") {
      cleanup("stopped");
      process.exit(0);
    }
  });

  process.on("disconnect", () => {
    cleanup("parent_disconnected");
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    cleanup("sigterm");
    process.exit(0);
  });
  process.on("SIGINT", () => {
    cleanup("sigint");
    process.exit(0);
  });
  process.on("uncaughtException", (error) => {
    send({ type: "fatal-error", pid: process.pid, error: error.message });
    cleanup("crashed");
    process.exit(1);
  });
  process.on("unhandledRejection", (error) => {
    const message = error instanceof Error ? error.message : String(error);
    send({ type: "fatal-error", pid: process.pid, error: message });
    cleanup("crashed");
    process.exit(1);
  });

  worker.start();
  const readyAt = Date.now();
  store.setTrainingState("worker_last_ready_at", readyAt);
  send({ type: "ready", pid: process.pid, at: readyAt });
}

void main().catch((error) => {
  const message = (error as Error).message;
  send({ type: "fatal-error", pid: process.pid, error: message });
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
