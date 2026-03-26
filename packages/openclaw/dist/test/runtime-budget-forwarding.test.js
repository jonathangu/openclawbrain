import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createBeforePromptBuildHandler } from "../extension/runtime-guard.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test("before_prompt_build forwards explicit maxContextChars to compileRuntimeContext", async () => {
  let compileInput = null;
  const handler = createBeforePromptBuildHandler({
    activationRoot: "/tmp/openclawbrain-activation",
    compileRuntimeContext: (input) => {
      compileInput = input;
      return { ok: true, brainContext: "[BRAIN_CONTEXT v1]\n[/BRAIN_CONTEXT]\n" };
    },
    reportDiagnostic: async () => undefined,
  });

  const result = await handler(
    {
      maxContextChars: 4096,
      messages: [{ role: "user", content: "latest prompt" }],
    },
    {},
  );

  assert.deepEqual(result, {
    appendSystemContext: "[BRAIN_CONTEXT v1]\n[/BRAIN_CONTEXT]\n",
  });
  assert.equal(compileInput?.maxContextChars, 4096);
});

test("before_prompt_build reports shaped diagnostics for malformed runtime envelopes", async () => {
  const diagnostics = [];
  const handler = createBeforePromptBuildHandler({
    activationRoot: "/tmp/openclawbrain-activation",
    compileRuntimeContext: () => ({ ok: true, brainContext: "" }),
    reportDiagnostic: async (diagnostic) => {
      diagnostics.push(diagnostic);
    },
  });

  const result = await handler(
    {
      messages: {
        content: "not-an-array",
      },
    },
    {},
  );

  assert.deepEqual(result, {});
  assert.equal(diagnostics.length, 1);
  assert.equal(diagnostics[0]?.key, "runtime-messages-not-array");
  assert.equal(diagnostics[0]?.severity, "degraded");
  assert.equal(diagnostics[0]?.actionability, "inspect_host_event_shape");
  assert.match(diagnostics[0]?.summary ?? "", /partial or malformed/);
  assert.match(diagnostics[0]?.action ?? "", /fail-opened safely/);
});

test("serve-time operator audit code records explicit maxContextChars", () => {
  const runtimeCoreSource = readFileSync(path.join(__dirname, "..", "src", "runtime-core.js"), "utf8");
  const learningSpineSource = readFileSync(path.join(__dirname, "..", "src", "learning-spine.js"), "utf8");

  assert.match(runtimeCoreSource, /syntheticTurn\.maxContextChars = input\.compileInput\.maxContextChars;/);
  assert.match(learningSpineSource, /maxContextChars: input\.turn\.maxContextChars \?\? null/);
  assert.match(learningSpineSource, /activePackGraphChecksum: activePack\?\.manifest\.payloadChecksums\.graph \?\? null/);
  assert.match(learningSpineSource, /selectionDigest: input\.compileResult\.ok \? input\.compileResult\.compileResponse\.diagnostics\.selectionDigest : null/);
  assert.match(
    learningSpineSource,
    /structuralSignals: compactStructuralSignals\(input\.compileResult\.ok \? input\.compileResult\.compileResponse\.structuralSignals : null\)/,
  );
  assert.doesNotMatch(learningSpineSource, /structuralSignals:\s*null/);
});

test("serve-time operator audit code keeps compact structural stop-truth signals", () => {
  const learningSpineSource = readFileSync(path.join(__dirname, "..", "src", "learning-spine.js"), "utf8");
  const helperStart = learningSpineSource.indexOf("function isPlainObject");
  const helperEnd = learningSpineSource.indexOf("function isStableKernelContextBlock");

  assert.notEqual(helperStart, -1);
  assert.notEqual(helperEnd, -1);
  assert.ok(helperEnd > helperStart);

  const helperSource = learningSpineSource.slice(helperStart, helperEnd);
  const compactStructuralSignals = new Function(
    `const roundNumber = (value) => Math.round(value * 10_000) / 10_000; ${helperSource}; return compactStructuralSignals;`,
  )();

  const compacted = compactStructuralSignals({
    chosenStopCount: 0,
    forcedStopCount: 2,
    droppedProposalCount: 1,
    droppedProposalReasons: {
      missing_target_node: 1,
    },
    graphWalkPathNodeIds: ["node-a", "node-b"],
    ignored: new Map([["x", 1]]),
  });

  assert.deepEqual(compacted, {
    chosenStopCount: 0,
    forcedStopCount: 2,
    droppedProposalCount: 1,
    droppedProposalReasons: {
      missing_target_node: 1,
    },
    graphWalkPathNodeIds: ["node-a", "node-b"],
  });
});

test("CLI and OpenClaw keep Wave 1 serve-time provenance forwarding aligned", () => {
  const cliLearningSpineSource = readFileSync(
    path.join(__dirname, "..", "..", "..", "cli", "dist", "src", "learning-spine.js"),
    "utf8",
  );
  const openclawLearningSpineSource = readFileSync(path.join(__dirname, "..", "src", "learning-spine.js"), "utf8");

  const parityPatterns = [
    /maxContextChars: input\.turn\.maxContextChars \?\? null/,
    /activePackGraphChecksum: activePack\?\.manifest\.payloadChecksums\.graph \?\? null/,
    /selectionDigest: input\.compileResult\.ok \? input\.compileResult\.compileResponse\.diagnostics\.selectionDigest : null/,
    /structuralSignals: compactStructuralSignals\(input\.compileResult\.ok \? input\.compileResult\.compileResponse\.structuralSignals : null\)/,
  ];

  for (const pattern of parityPatterns) {
    assert.match(cliLearningSpineSource, pattern);
    assert.match(openclawLearningSpineSource, pattern);
  }

  assert.doesNotMatch(cliLearningSpineSource, /structuralSignals:\s*null/);
});
