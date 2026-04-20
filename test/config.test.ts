import { describe, it, expect } from "vitest";
import manifest from "../openclaw.plugin.json" with { type: "json" };
import packageManifest from "../packages/openclaw/openclaw.plugin.json" with { type: "json" };
import { resolveLcmConfig } from "../src/db/config.js";

describe("resolveLcmConfig", () => {
  it("uses hardcoded defaults when no env or plugin config", () => {
    const config = resolveLcmConfig({}, {});
    expect(config.enabled).toBe(true);
    expect(config.contextThreshold).toBe(0.75);
    expect(config.freshTailCount).toBe(32);
    expect(config.incrementalMaxDepth).toBe(0);
    expect(config.leafMinFanout).toBe(8);
    expect(config.condensedMinFanout).toBe(4);
    expect(config.condensedMinFanoutHard).toBe(2);
    expect(config.autocompactDisabled).toBe(false);
    expect(config.pruneHeartbeatOk).toBe(false);
    expect(config.brain?.persistRawSurfaces).toBe(false);
  });

  it("reads values from plugin config", () => {
    const config = resolveLcmConfig({}, {
      contextThreshold: 0.5,
      freshTailCount: 16,
      incrementalMaxDepth: -1,
      brainEmbeddingBaseUrl: "https://embeddings.example.invalid/v1",
      brainMaxCompileMs: 75,
      brainBudgetFraction: 0.45,
      brainMaxHops: 9,
      brainMaxFanoutPerNode: 7,
      brainMaxFrontierSize: 19,
      brainPersistRawSurfaces: true,
      brainDirectAnswerNoFire: true,
      brainSuppressSyntheticWorkspaceSentinel: true,
      brainMaxSeeds: 13,
      brainSemanticThreshold: 0.82,
      brainShadowMode: true,
      brainWorkerMode: "in_process",
      brainWorkerHeartbeatTimeoutMs: 12_000,
      brainWorkerRestartDelayMs: 600,
      leafMinFanout: 4,
      condensedMinFanout: 2,
      autocompactDisabled: true,
      pruneHeartbeatOk: true,
      enabled: false,
    });
    expect(config.enabled).toBe(false);
    expect(config.contextThreshold).toBe(0.5);
    expect(config.freshTailCount).toBe(16);
    expect(config.incrementalMaxDepth).toBe(-1);
    expect(config.brain?.embeddingBaseUrl).toBe("https://embeddings.example.invalid/v1");
    expect(config.brain?.maxCompileMs).toBe(75);
    expect(config.brain?.budgetFraction).toBe(0.45);
    expect(config.brain?.maxHops).toBe(9);
    expect(config.brain?.maxFanoutPerNode).toBe(7);
    expect(config.brain?.maxFrontierSize).toBe(19);
    expect(config.brain?.persistRawSurfaces).toBe(true);
    expect(config.brain?.directAnswerNoFire).toBe(true);
    expect(config.brain?.suppressSyntheticWorkspaceSentinel).toBe(true);
    expect(config.brain?.maxSeeds).toBe(13);
    expect(config.brain?.semanticThreshold).toBe(0.82);
    expect(config.brain?.shadowMode).toBe(true);
    expect(config.brain?.workerMode).toBe("in_process");
    expect(config.brain?.workerHeartbeatTimeoutMs).toBe(12_000);
    expect(config.brain?.workerRestartDelayMs).toBe(600);
    expect(config.leafMinFanout).toBe(4);
    expect(config.condensedMinFanout).toBe(2);
    expect(config.autocompactDisabled).toBe(true);
    expect(config.pruneHeartbeatOk).toBe(true);
  });

  it("env vars override plugin config", () => {
    const env = {
      LCM_CONTEXT_THRESHOLD: "0.9",
      LCM_FRESH_TAIL_COUNT: "64",
      LCM_INCREMENTAL_MAX_DEPTH: "3",
      LCM_ENABLED: "false",
      LCM_AUTOCOMPACT_DISABLED: "true",
      OPENCLAWBRAIN_PERSIST_RAW_SURFACES: "true",
    } as NodeJS.ProcessEnv;
    const pluginConfig = {
      contextThreshold: 0.5,
      freshTailCount: 16,
      incrementalMaxDepth: -1,
      enabled: true,
      autocompactDisabled: false,
    };
    const config = resolveLcmConfig(env, pluginConfig);
    expect(config.enabled).toBe(false); // env wins
    expect(config.contextThreshold).toBe(0.9); // env wins
    expect(config.freshTailCount).toBe(64); // env wins
    expect(config.incrementalMaxDepth).toBe(3); // env wins
    expect(config.autocompactDisabled).toBe(true); // env wins
    expect(config.brain?.persistRawSurfaces).toBe(true); // env wins
  });

  it("plugin config fills gaps when env vars are absent", () => {
    const env = {
      LCM_CONTEXT_THRESHOLD: "0.9",
    } as NodeJS.ProcessEnv;
    const pluginConfig = {
      contextThreshold: 0.5, // should be overridden by env
      freshTailCount: 16, // should be used (no env)
      incrementalMaxDepth: -1, // should be used (no env)
    };
    const config = resolveLcmConfig(env, pluginConfig);
    expect(config.contextThreshold).toBe(0.9); // env wins
    expect(config.freshTailCount).toBe(16); // plugin config
    expect(config.incrementalMaxDepth).toBe(-1); // plugin config
    expect(config.leafMinFanout).toBe(8); // hardcoded default
  });

  it("handles string values in plugin config (from JSON)", () => {
    const config = resolveLcmConfig({}, {
      contextThreshold: "0.6",
      freshTailCount: "24",
    });
    expect(config.contextThreshold).toBe(0.6);
    expect(config.freshTailCount).toBe(24);
  });

  it("ignores invalid plugin config values", () => {
    const config = resolveLcmConfig({}, {
      contextThreshold: "not-a-number",
      freshTailCount: null,
      enabled: "maybe",
    });
    expect(config.contextThreshold).toBe(0.75); // falls through to default
    expect(config.freshTailCount).toBe(32); // falls through to default
    expect(config.enabled).toBe(true); // falls through to default
  });

  it("handles databasePath from plugin config", () => {
    const config = resolveLcmConfig({}, {
      databasePath: "/custom/path/lcm.db",
    });
    expect(config.databasePath).toBe("/custom/path/lcm.db");
  });

  it("accepts manifest dbPath from plugin config", () => {
    const config = resolveLcmConfig({}, {
      dbPath: "/manifest/path/lcm.db",
    });
    expect(config.databasePath).toBe("/manifest/path/lcm.db");
  });

  it("env databasePath overrides plugin config", () => {
    const config = resolveLcmConfig(
      { LCM_DATABASE_PATH: "/env/path/lcm.db" } as NodeJS.ProcessEnv,
      { databasePath: "/plugin/path/lcm.db" },
    );
    expect(config.databasePath).toBe("/env/path/lcm.db");
  });

  it("accepts manifest largeFileThresholdTokens from plugin config", () => {
    const config = resolveLcmConfig({}, {
      largeFileThresholdTokens: 12345,
    });
    expect(config.largeFileTokenThreshold).toBe(12345);
  });

  it("ships a manifest that accepts unlimited incremental depth", () => {
    expect(manifest.configSchema.properties.incrementalMaxDepth.minimum).toBe(-1);
  });

  it("clamps budgetFraction and falls back for invalid traversal caps", () => {
    const config = resolveLcmConfig({}, {
      brainBudgetFraction: 4,
      brainMaxFanoutPerNode: 0,
      brainMaxFrontierSize: -3,
    });

    expect(config.brain?.budgetFraction).toBe(1);
    expect(config.brain?.maxFanoutPerNode).toBe(4);
    expect(config.brain?.maxFrontierSize).toBe(32);
  });

  it("keeps root and published manifests aligned for live runtime controls", () => {
    const keys = [
      "brainEmbeddingBaseUrl",
      "brainMaxCompileMs",
      "brainBudgetFraction",
      "brainMaxHops",
      "brainMaxFanoutPerNode",
      "brainMaxFrontierSize",
      "brainPersistRawSurfaces",
      "brainDirectAnswerNoFire",
      "brainSuppressSyntheticWorkspaceSentinel",
      "brainMaxSeeds",
      "brainSemanticThreshold",
      "brainShadowMode",
      "brainWorkerMode",
      "brainWorkerHeartbeatTimeoutMs",
      "brainWorkerRestartDelayMs",
    ] as const;

    for (const key of keys) {
      expect(packageManifest.configSchema.properties[key]).toEqual(manifest.configSchema.properties[key]);
      expect(packageManifest.uiHints[key]).toEqual(manifest.uiHints[key]);
    }
  });
});
