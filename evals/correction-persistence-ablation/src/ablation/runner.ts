import type { AgentClient } from "../agent/client.js";
import { runCase } from "../eval/correction-recurrence.js";
import type { Ledger } from "../ledger/ledger.js";
import type { AblationResult, MemoryBackend, TaskCase } from "../types.js";
import type { MemoryBackendImpl } from "./backends.js";

export interface AblationArgs {
  run_id: string;
  cases: TaskCase[];
  backends: MemoryBackendImpl[];
  agent: AgentClient;
  ledger: Ledger;
  system_prompt?: string;
  onProgress?: (evt: {
    case_id: string;
    backend: MemoryBackend;
    passed: boolean;
    i: number;
    total: number;
  }) => void;
}

export async function runAblation(args: AblationArgs): Promise<{
  summary: AblationResult[];
  matrix: Array<{ case_id: string; results: Partial<Record<MemoryBackend, boolean>> }>;
}> {
  const { run_id, cases, backends, agent, ledger, system_prompt, onProgress } = args;

  const matrix: Array<{ case_id: string; results: Partial<Record<MemoryBackend, boolean>> }> = [];
  const total = cases.length * backends.length;
  let i = 0;

  for (const taskCase of cases) {
    const results: Partial<Record<MemoryBackend, boolean>> = {};
    for (const backend of backends) {
      const { passed } = await runCase({
        run_id,
        case: taskCase,
        backend,
        agent,
        ledger,
        system_prompt,
      });
      results[backend.name] = passed;
      i++;
      onProgress?.({ case_id: taskCase.case_id, backend: backend.name, passed, i, total });
    }
    matrix.push({ case_id: taskCase.case_id, results });
  }

  const summary: AblationResult[] = [];
  const slices: Array<"all" | TaskCase["slice"]> = ["all", ...new Set(cases.map((c) => c.slice))];
  for (const backend of backends) {
    for (const slice of slices) {
      summary.push(ledger.aggregate({ run_id, backend: backend.name, slice }));
    }
  }

  return { summary, matrix };
}
