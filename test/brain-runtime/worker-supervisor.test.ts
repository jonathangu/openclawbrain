import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { fork } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_BRAIN_CONFIG } from "../../src/brain-core/types.js";
import { resolveChildWorkerExecArgv } from "../../src/brain-runtime/worker-supervisor.js";

const tempDirs: string[] = [];
const childRunnerPath = fileURLToPath(new URL("../../src/brain-worker/child-runner.ts", import.meta.url));

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  while (tempDirs.length) {
    const dir = tempDirs.pop();
    if (dir) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("resolveChildWorkerExecArgv", () => {
  it("resolves tsx to an absolute loader path instead of relying on cwd", () => {
    const execArgv = resolveChildWorkerExecArgv();
    expect(execArgv[0]).toBe("--import");
    expect(execArgv[1]).not.toBe("tsx/esm");
    expect(execArgv[1]).toMatch(/^file:\/\//);
  });

  it("lets the child worker boot from cwd=/ without tsx resolution failures", async () => {
    const brainRoot = makeTempDir("ocb-worker-supervisor-");
    mkdirSync(brainRoot, { recursive: true });

    const childConfig = {
      ...DEFAULT_BRAIN_CONFIG,
      root: brainRoot,
      trainerIntervalMs: 60_000,
      workerHeartbeatTimeoutMs: 5_000,
      teacherEnabled: false,
        persistRawSurfaces: false,
      embeddingProvider: "ollama",
      embeddingModel: "bge-large:latest",
    };

    const result = await new Promise<{ ready: boolean; stderr: string; exitCode: number | null }>((resolve) => {
      let stderr = "";
      let resolved = false;
      const child = fork(childRunnerPath, [], {
        cwd: "/",
        execArgv: resolveChildWorkerExecArgv(),
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        env: {
          ...process.env,
          OPENCLAWBRAIN_CHILD_CONFIG_JSON: JSON.stringify(childConfig),
          OPENCLAWBRAIN_CHILD_TEACHER_MODEL_JSON: "",
        },
      });

      const finish = (payload: { ready: boolean; stderr: string; exitCode: number | null }) => {
        if (resolved) return;
        resolved = true;
        resolve(payload);
      };

      child.stderr?.on("data", (chunk) => {
        stderr += String(chunk);
      });

      child.on("message", (message: { type?: string }) => {
        if (message?.type === "ready") {
          child.send?.({ type: "shutdown" });
          finish({ ready: true, stderr, exitCode: null });
        }
      });

      child.on("exit", (code) => {
        finish({ ready: false, stderr, exitCode: code });
      });

      setTimeout(() => {
        child.kill("SIGTERM");
        finish({ ready: false, stderr, exitCode: null });
      }, 5_000);
    });

    expect(result.ready).toBe(true);
    expect(result.stderr).not.toContain("Cannot find package 'tsx'");
  });
});
