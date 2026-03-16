import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");

type ScenarioTurn = {
  to: string;
  message: string;
  turnShape: "first_turn" | "primed_second_turn";
};

type Scenario = {
  name: string;
  description: string;
  turns: ScenarioTurn[];
};

type ValidationRecord = {
  at?: number;
  sessionId?: string | null;
  conversationId?: number | null;
  queryText?: string | null;
  mode?: string | null;
  footer?: string | null;
  traceId?: string | null;
  episodeId?: string | null;
  tokenBudget?: number | null;
};

type AgentResult = {
  payloads?: Array<{ text?: string | null }>;
  meta?: {
    aborted?: boolean;
    durationMs?: number;
    agentMeta?: { sessionId?: string | null };
    systemPromptReport?: {
      sessionKey?: string | null;
      sessionId?: string | null;
      provider?: string | null;
      model?: string | null;
      workspaceDir?: string | null;
    };
  };
};

type TurnResult = {
  to: string;
  message: string;
  turnShape: ScenarioTurn["turnShape"];
  validationRecord: ValidationRecord | null;
  finalReportedHostDecision: string | null;
  responseText: string;
  responseSource: "static_lookup" | "brain_pack" | "mixed" | "unknown";
  sessionKey: string | null;
  sessionId: string | null;
  conversationId: number | null;
  aborted: boolean;
  durationMs: number | null;
};

type ScenarioResult = {
  name: string;
  description: string;
  validationRoot: string;
  stateDir: string;
  configPath: string;
  validationRecordFile: string;
  turns: TurnResult[];
  scenarioResponseSource: "static_lookup" | "brain_pack" | "mixed" | "unknown";
};

type CommandCapture = {
  command: string;
  args: string[];
  status: number | null;
  stdout: string;
  stderr: string;
  error: string | null;
};

type HostPreflight = {
  validationRoot: string;
  configPath: string;
  blockedByStaleHostSeam: boolean;
  reasons: string[];
  setupOnly: CommandCapture;
  doctor: CommandCapture;
  sdkProbe: CommandCapture;
};

function parseArgs(argv: string[]) {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key?.startsWith("--") && value) {
      args.set(key.slice(2), value);
      index += 1;
    }
  }

  const home = process.env.HOME ?? homedir();
  const gitSha = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  const artifactDate = new Date().toISOString().slice(0, 10);
  return {
    validationRoot: resolve(
      args.get("state-dir")
        ?? process.env.OPENCLAWBRAIN_VALIDATION_ROOT
        ?? join(home, ".openclaw-ocbphase1-short-static"),
    ),
    workspaceRoot: resolve(
      args.get("workspace")
        ?? process.env.OPENCLAWBRAIN_VALIDATION_WORKSPACE
        ?? join(home, ".openclaw", "workspace-ocbphase1"),
    ),
    validationModel:
      args.get("model")
      ?? process.env.OPENCLAWBRAIN_VALIDATION_MODEL
      ?? "ollama/qwen2.5:7b-instruct",
    embeddingProvider:
      args.get("embedding-provider")
      ?? process.env.OPENCLAWBRAIN_VALIDATION_EMBEDDING_PROVIDER
      ?? "ollama",
    embeddingModel:
      args.get("embedding-model")
      ?? process.env.OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL
      ?? "bge-large:latest",
    agentTimeoutSeconds: Number.parseInt(args.get("agent-timeout") ?? "120", 10),
    artifactDir: resolve(
      args.get("artifact-dir")
        ?? process.env.OPENCLAWBRAIN_VALIDATION_ARTIFACT_DIR
        ?? join(repoRoot, "docs", "evidence", artifactDate, gitSha, "short-static-classification"),
    ),
    gitSha,
  };
}

const options = parseArgs(process.argv.slice(2));
mkdirSync(options.artifactDir, { recursive: true });

const scenarios: Scenario[] = [
  {
    name: "first-turn-short-static",
    description: "Fresh local host session; first turn is `open PLAYBOOK.md`.",
    turns: [
      { to: "+15550010101", message: "open PLAYBOOK.md", turnShape: "first_turn" },
    ],
  },
  {
    name: "primed-second-turn-after-short-static",
    description: "Fresh state; primer is `open PLAYBOOK.md`, then the host is asked to answer the previous question directly.",
    turns: [
      { to: "+15550010201", message: "open PLAYBOOK.md", turnShape: "first_turn" },
      { to: "+15550010201", message: "Please answer my previous question directly.", turnShape: "primed_second_turn" },
    ],
  },
  {
    name: "primed-second-turn-after-routed-ask",
    description: "Fresh state; primer is a routed recurrent ask, then the host is asked to answer the previous question directly.",
    turns: [
      { to: "+15550010301", message: "How do I open a pull request again?", turnShape: "first_turn" },
      { to: "+15550010301", message: "Please answer my previous question directly.", turnShape: "primed_second_turn" },
    ],
  },
  {
    name: "local-to-isolation-probe",
    description: "Fresh state; two different `--to` values are used to check whether `openclaw agent --local` isolates local sessions by recipient.",
    turns: [
      { to: "+15550010401", message: "How do I open a pull request again?", turnShape: "first_turn" },
      { to: "+15550010402", message: "How do I open a pull request again?", turnShape: "first_turn" },
    ],
  },
];

function extractJson(text: string): any {
  try {
    return JSON.parse(text);
  } catch {}

  for (let index = text.lastIndexOf("{"); index >= 0; index = text.lastIndexOf("{", index - 1)) {
    const candidate = text.slice(index).trim();
    try {
      return JSON.parse(candidate);
    } catch {}
  }

  throw new Error(`Unable to parse JSON from command output:\n${text}`);
}

function runCapture(command: string, args: string[], env: NodeJS.ProcessEnv, timeoutMs?: number) {
  return spawnSync(command, args, {
    cwd: repoRoot,
    env,
    encoding: "utf8",
    ...(typeof timeoutMs === "number" ? { timeout: timeoutMs } : {}),
  });
}

function captureCommand(command: string, args: string[], env: NodeJS.ProcessEnv, timeoutMs?: number): CommandCapture {
  const result = runCapture(command, args, env, timeoutMs);
  return {
    command,
    args,
    status: result.status ?? null,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error ? String(result.error) : null,
  };
}

function runJson(command: string, args: string[], env: NodeJS.ProcessEnv, timeoutMs?: number): any {
  const result = runCapture(command, args, env, timeoutMs);
  if (result.error) {
    throw new Error(`${command} ${args.join(" ")} failed: ${String(result.error)}`);
  }
  if ((result.status ?? 1) !== 0) {
    throw new Error(`${command} ${args.join(" ")} exited with code ${result.status ?? 1}:\n${result.stderr || result.stdout}`);
  }
  return extractJson(result.stdout ?? "");
}

function textContainsAny(text: string, needles: string[]): boolean {
  const haystack = text.toLowerCase();
  return needles.some((needle) => haystack.includes(needle.toLowerCase()));
}

function normalizeText(value: unknown): string | null {
  return typeof value === "string" ? value.trim().toLowerCase() : null;
}

function buildScenarioEnv(root: string, port: number): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env };
  for (const key of Object.keys(env)) {
    if (key.startsWith("OPENCLAW_")) {
      delete env[key];
    }
  }

  env.OPENCLAWBRAIN_VALIDATION_ROOT = root;
  env.OPENCLAWBRAIN_VALIDATION_STATE_DIR = root;
  env.OPENCLAWBRAIN_VALIDATION_WORKSPACE = options.workspaceRoot;
  env.OPENCLAWBRAIN_VALIDATION_GATEWAY_PORT = String(port);
  env.OPENCLAWBRAIN_VALIDATION_MODEL = options.validationModel;
  env.OPENCLAWBRAIN_VALIDATION_EMBEDDING_PROVIDER = options.embeddingProvider;
  env.OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL = options.embeddingModel;

  const configPath = join(root, "openclaw.json");
  const stateDir = root;
  env.OPENCLAW_CONFIG_PATH = configPath;
  env.OPENCLAW_STATE_DIR = stateDir;
  env.LCM_DATABASE_PATH = join(stateDir, "lcm.db");
  env.OPENCLAWBRAIN_ROOT = join(stateDir, "openclawbrain");
  env.OPENCLAWBRAIN_EMBEDDING_PROVIDER = options.embeddingProvider;
  env.OPENCLAWBRAIN_EMBEDDING_MODEL = options.embeddingModel;
  env.OPENCLAWBRAIN_VALIDATION_RECORD_FILE = join(stateDir, "validation-records", "validation-assemble.jsonl");
  return env;
}

function readValidationRecords(path: string): ValidationRecord[] {
  if (!existsSync(path)) {
    return [];
  }
  const text = readFileSync(path, "utf8").trim();
  if (!text) {
    return [];
  }
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ValidationRecord);
}

function findRecordForTurn(records: ValidationRecord[], startIndex: number, message: string): ValidationRecord | null {
  const normalizedMessage = normalizeText(message);
  const newRecords = records.slice(startIndex);
  if (normalizedMessage) {
    const exact = [...newRecords].reverse().find((record) => normalizeText(record.queryText) === normalizedMessage);
    if (exact) {
      return exact;
    }
  }
  return newRecords.at(-1) ?? null;
}

function collectResponseText(payload: AgentResult): string {
  return (payload.payloads ?? [])
    .map((entry) => entry.text ?? "")
    .join("\n")
    .trim();
}

function classifyTurnResponseSource(record: ValidationRecord | null, responseText: string): TurnResult["responseSource"] {
  const mode = record?.mode ?? null;
  const normalized = responseText.toLowerCase();
  const containsFixtureFact = normalized.includes("gh pr create") || normalized.includes("pull request workflows");
  if (mode === "skip_short_static_lookup") {
    return containsFixtureFact ? "static_lookup" : "unknown";
  }
  if (mode === "use_brain" || mode === "shadow") {
    return containsFixtureFact ? "brain_pack" : "unknown";
  }
  if (containsFixtureFact) {
    return "mixed";
  }
  return "unknown";
}

function scenarioResponseSource(turns: TurnResult[]): ScenarioResult["scenarioResponseSource"] {
  const unique = new Set(turns.map((turn) => turn.responseSource).filter((value) => value !== "unknown"));
  if (unique.size === 0) {
    return "unknown";
  }
  if (unique.size > 1) {
    return "mixed";
  }
  return [...unique][0] as ScenarioResult["scenarioResponseSource"];
}

function ensureFixtureWorkspace() {
  mkdirSync(options.workspaceRoot, { recursive: true });
  writeFileSync(
    join(options.workspaceRoot, "PLAYBOOK.md"),
    [
      "# Pull Requests",
      "",
      "Use `gh pr create` for pull request workflows.",
      "If the branch is not pushed yet, push it first and then open the PR.",
      "",
      "# Deployments",
      "",
      "Check CI logs before retrying a deployment.",
    ].join("\n"),
    "utf8",
  );
}

function runHostPreflight(): HostPreflight {
  const preflightRoot = join(options.validationRoot, "preflight");
  rmSync(preflightRoot, { recursive: true, force: true });
  mkdirSync(preflightRoot, { recursive: true });
  const env = buildScenarioEnv(preflightRoot, 19130);
  const setupOnly = captureCommand(
    "node",
    ["scripts/validate-openclaw-install.mjs", "--sterile-lane", "--setup-only"],
    env,
    300_000,
  );
  const doctor = captureCommand("openclaw", ["doctor"], env, 120_000);
  const sdkProbe = captureCommand(
    "node",
    [
      "-e",
      [
        'const sdk = require("openclaw/plugin-sdk");',
        'console.log(JSON.stringify({ hasRegisterContextEngine: typeof sdk.registerContextEngine === "function", keys: Object.keys(sdk).sort().filter((key) => key.toLowerCase().includes("context") || key.toLowerCase().includes("memory")).slice(0, 50) }));',
      ].join(" "),
    ],
    env,
    30_000,
  );

  const combinedOutput = [setupOnly.stdout, setupOnly.stderr, doctor.stdout, doctor.stderr, sdkProbe.stdout, sdkProbe.stderr]
    .filter(Boolean)
    .join("\n");
  const reasons: string[] = [];
  if (textContainsAny(combinedOutput, ['plugins.slots.contextEngine', 'unrecognized key: "contextEngine"'])) {
    reasons.push("OpenClaw config schema no longer accepts `plugins.slots.contextEngine` in the sterile host lane.");
  }
  if (textContainsAny(combinedOutput, ['openclawbrain failed during register', 'api.registerContextEngine is not a function'])) {
    reasons.push("OpenClawBrain still failed during plugin register on the current host seam, so turn-level host probes are not trustworthy yet.");
  }

  return {
    validationRoot: preflightRoot,
    configPath: join(preflightRoot, "openclaw.json"),
    blockedByStaleHostSeam: reasons.length > 0,
    reasons,
    setupOnly,
    doctor,
    sdkProbe,
  };
}

function runScenario(scenario: Scenario, index: number): ScenarioResult {
  const scenarioRoot = join(options.validationRoot, scenario.name);
  rmSync(scenarioRoot, { recursive: true, force: true });
  mkdirSync(scenarioRoot, { recursive: true });
  const env = buildScenarioEnv(scenarioRoot, 19131 + index);

  runCapture(
    "node",
    ["scripts/validate-openclaw-install.mjs", "--sterile-lane", "--setup-only"],
    env,
    300_000,
  );
  runJson("node", ["bin/openclawbrain.js", "init", options.workspaceRoot], env, 300_000);

  const validationRecordFile = env.OPENCLAWBRAIN_VALIDATION_RECORD_FILE as string;
  const turns: TurnResult[] = [];
  let recordCount = readValidationRecords(validationRecordFile).length;

  for (const turn of scenario.turns) {
    const payload = runJson(
      "openclaw",
      [
        "agent",
        "--local",
        "--to",
        turn.to,
        "--message",
        turn.message,
        "--json",
        "--timeout",
        String(options.agentTimeoutSeconds),
      ],
      env,
      (options.agentTimeoutSeconds + 30) * 1000,
    ) as AgentResult;
    const status = runJson("node", ["bin/openclawbrain.js", "status"], env, 60_000);
    const records = readValidationRecords(validationRecordFile);
    const record = findRecordForTurn(records, recordCount, turn.message);
    recordCount = records.length;
    const responseText = collectResponseText(payload);

    turns.push({
      to: turn.to,
      message: turn.message,
      turnShape: turn.turnShape,
      validationRecord: record,
      finalReportedHostDecision:
        typeof (status.lastAssemblyDecision as { mode?: unknown } | undefined)?.mode === "string"
          ? ((status.lastAssemblyDecision as { mode: string }).mode)
          : null,
      responseText,
      responseSource: classifyTurnResponseSource(record, responseText),
      sessionKey:
        typeof payload.meta?.systemPromptReport?.sessionKey === "string"
          ? payload.meta.systemPromptReport.sessionKey
          : null,
      sessionId:
        typeof payload.meta?.agentMeta?.sessionId === "string"
          ? payload.meta.agentMeta.sessionId
          : typeof payload.meta?.systemPromptReport?.sessionId === "string"
            ? payload.meta.systemPromptReport.sessionId
            : null,
      conversationId: typeof record?.conversationId === "number" ? record.conversationId : null,
      aborted: payload.meta?.aborted === true,
      durationMs: typeof payload.meta?.durationMs === "number" ? payload.meta.durationMs : null,
    });
  }

  return {
    name: scenario.name,
    description: scenario.description,
    validationRoot: scenarioRoot,
    stateDir: scenarioRoot,
    configPath: join(scenarioRoot, "openclaw.json"),
    validationRecordFile,
    turns,
    scenarioResponseSource: scenarioResponseSource(turns),
  };
}

function classifyResult(preflight: HostPreflight, results: ScenarioResult[]) {
  const byName = new Map(results.map((result) => [result.name, result]));
  const firstTurnShort = byName.get("first-turn-short-static")?.turns[0] ?? null;
  const secondTurnAfterShort = byName.get("primed-second-turn-after-short-static")?.turns[1] ?? null;
  const secondTurnAfterRouted = byName.get("primed-second-turn-after-routed-ask")?.turns[1] ?? null;
  const isolationProbe = byName.get("local-to-isolation-probe")?.turns ?? [];

  const sameSessionKeyAcrossDifferentTo =
    isolationProbe.length >= 2
    && Boolean(isolationProbe[0]?.sessionKey)
    && isolationProbe.every((turn) => turn.sessionKey === isolationProbe[0]?.sessionKey);
  const sameSessionIdAcrossDifferentTo =
    isolationProbe.length >= 2
    && Boolean(isolationProbe[0]?.sessionId)
    && isolationProbe.every((turn) => turn.sessionId === isolationProbe[0]?.sessionId);

  const bucket = preflight.blockedByStaleHostSeam
    ? "upstream host-agent/profile interaction"
    : sameSessionKeyAcrossDifferentTo
      ? "upstream host-agent/profile interaction"
      : firstTurnShort?.validationRecord?.mode === "skip_short_static_lookup"
        && (secondTurnAfterShort?.validationRecord?.mode === "use_brain" || secondTurnAfterRouted?.validationRecord?.mode === "use_brain")
        ? "true behavioral drift by turn shape"
        : firstTurnShort?.validationRecord?.mode === firstTurnShort?.finalReportedHostDecision
          ? "reporting drift only"
          : "true behavioral drift by turn shape";

  const rationale = preflight.blockedByStaleHostSeam
    ? [
        ...preflight.reasons,
        "That means the current raw host validation lane cannot honestly reach the short-static semantic question yet; the host/plugin integration boundary moved underneath the Phase 1 harness.",
        "Freeze this as upstream host-agent/profile interaction for now, then adapt the plugin + config seam before claiming host-path short-static behavior is classified.",
      ]
    : bucket === "upstream host-agent/profile interaction"
      ? [
          "Fresh first-turn `open PLAYBOOK.md` still bypasses as `skip_short_static_lookup`, so the assembler-level bypass remains real.",
          "But `openclaw agent --local` reuses the same local session key across different `--to` values (`agent:main:main` in observed runs), so recipient changes do not create isolated local sessions for host-surface probes.",
          "That means any apparent primed short-static drift on the raw host surface is dominated by local-agent session reuse / host interaction rather than evidence that the short-static classifier itself is wrong.",
        ]
      : bucket === "true behavioral drift by turn shape"
        ? [
            "Fresh first-turn `open PLAYBOOK.md` bypasses as `skip_short_static_lookup`.",
            "But the primed second-turn prompt (`Please answer my previous question directly.`) is not itself a short static lookup, so the host surface routes it differently.",
            "The observed semantic difference is therefore a real turn-shape difference, not just a logging mismatch.",
          ]
        : [
            "The host-surface output and final reported decision disagree while the underlying behavior stays aligned.",
            "This is best classified as reporting drift only.",
          ];

  return {
    bucket,
    blockedByStaleHostSeam: preflight.blockedByStaleHostSeam,
    preflightReasons: preflight.reasons,
    sameSessionKeyAcrossDifferentTo,
    sameSessionIdAcrossDifferentTo,
    firstTurnShortMode: firstTurnShort?.validationRecord?.mode ?? null,
    secondTurnAfterShortMode: secondTurnAfterShort?.validationRecord?.mode ?? null,
    secondTurnAfterRoutedMode: secondTurnAfterRouted?.validationRecord?.mode ?? null,
    rationale,
  };
}

function buildSummary(preflight: HostPreflight, results: ScenarioResult[], classification: ReturnType<typeof classifyResult>) {
  const lines = [
    "# Short-static host-path classification summary",
    "",
    `- commit: \`${options.gitSha}\``,
    `- workspace: \`${options.workspaceRoot}\``,
    `- validation root: \`${options.validationRoot}\``,
    `- model: \`${options.validationModel}\``,
    `- embedding: \`${options.embeddingProvider}/${options.embeddingModel}\``,
    `- classification bucket: **${classification.bucket}**`,
    `- blocked by stale host seam: ${classification.blockedByStaleHostSeam}`,
    `- same local session key across different --to values: ${classification.sameSessionKeyAcrossDifferentTo}`,
    `- same local session id across different --to values: ${classification.sameSessionIdAcrossDifferentTo}`,
    "",
    "## Host preflight",
    `- preflight root: \`${preflight.validationRoot}\``,
    `- preflight config: \`${preflight.configPath}\``,
    `- setup-only exit: ${preflight.setupOnly.status}`,
    `- doctor exit: ${preflight.doctor.status}`,
    `- sdk probe exit: ${preflight.sdkProbe.status}`,
    "",
    "## Why this bucket",
  ];

  for (const item of classification.rationale) {
    lines.push(`- ${item}`);
  }

  lines.push("", "## Scenario matrix");
  if (results.length === 0) {
    lines.push("- Skipped turn-level host probes because the host/plugin seam is stale before the agent path becomes meaningful.");
  }
  for (const scenario of results) {
    lines.push("", `### ${scenario.name}`);
    lines.push(`- description: ${scenario.description}`);
    lines.push(`- response source: ${scenario.scenarioResponseSource}`);
    lines.push(`- validation record file: \`${scenario.validationRecordFile}\``);
    for (const turn of scenario.turns) {
      lines.push(`- turn (${turn.turnShape}, to=${turn.to}):`);
      lines.push(`  - message: ${JSON.stringify(turn.message)}`);
      lines.push(`  - validation mode: ${JSON.stringify(turn.validationRecord?.mode ?? null)}`);
      lines.push(`  - validation query text: ${JSON.stringify(turn.validationRecord?.queryText ?? null)}`);
      lines.push(`  - final reported host decision: ${JSON.stringify(turn.finalReportedHostDecision)}`);
      lines.push(`  - session key: ${JSON.stringify(turn.sessionKey)}`);
      lines.push(`  - session id: ${JSON.stringify(turn.sessionId)}`);
      lines.push(`  - response source: ${turn.responseSource}`);
    }
  }

  lines.push(
    "",
    "## Honest release implication",
    classification.blockedByStaleHostSeam
      ? "- Do not treat the current raw host lane as a valid short-static proof surface. First adapt the plugin/config seam to the current OpenClaw host, then rerun turn-level classification on top of that repaired boundary."
      : classification.bucket === "upstream host-agent/profile interaction"
        ? "- Treat raw `openclaw agent --local` primed short-static probes as non-isolated host interaction, not as the canonical proof boundary for short-static semantics. First-turn assembler bypass remains real; the raw host path should be truth-frozen accordingly."
        : classification.bucket === "true behavioral drift by turn shape"
          ? "- The first-turn static bypass remains real, but multi-turn host semantics differ by turn shape and must be described that way until or unless behavior is intentionally changed."
          : "- The underlying behavior is aligned, but the host-surface reporting needs wording or instrumentation cleanup before the surface is described as fully resolved.",
  );

  return `${lines.join("\n")}\n`;
}

ensureFixtureWorkspace();
const preflight = runHostPreflight();
const results = preflight.blockedByStaleHostSeam ? [] : scenarios.map(runScenario);
const classification = classifyResult(preflight, results);
const report = {
  ok: true,
  gitSha: options.gitSha,
  artifactDir: options.artifactDir,
  workspace: options.workspaceRoot,
  validationRoot: options.validationRoot,
  model: options.validationModel,
  embeddingProvider: options.embeddingProvider,
  embeddingModel: options.embeddingModel,
  preflight,
  classification,
  scenarios: results,
};

writeFileSync(join(options.artifactDir, "preflight-setup-only.json"), `${JSON.stringify(preflight.setupOnly, null, 2)}\n`, "utf8");
writeFileSync(join(options.artifactDir, "preflight-doctor.json"), `${JSON.stringify(preflight.doctor, null, 2)}\n`, "utf8");
writeFileSync(join(options.artifactDir, "preflight-sdk-probe.json"), `${JSON.stringify(preflight.sdkProbe, null, 2)}\n`, "utf8");
writeFileSync(join(options.artifactDir, "validation-report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
writeFileSync(join(options.artifactDir, "summary.md"), buildSummary(preflight, results, classification), "utf8");
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
