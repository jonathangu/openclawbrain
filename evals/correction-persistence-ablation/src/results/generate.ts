import type { Ledger } from "../ledger/ledger.js";
import type { AblationResult, MemoryBackend, TurnSlice } from "../types.js";

export interface ResultsPageArgs {
  run_id: string;
  backends: MemoryBackend[];
  slices: Array<TurnSlice | "all">;
  title?: string;
  notes?: string;
}

export function generateResults(ledger: Ledger, args: ResultsPageArgs): { html: string; json: unknown } {
  const rows: AblationResult[] = [];
  for (const backend of args.backends) {
    for (const slice of args.slices) {
      rows.push(ledger.aggregate({ run_id: args.run_id, backend, slice }));
    }
  }

  const json = {
    run_id: args.run_id,
    generated_at: new Date().toISOString(),
    results: rows,
  };

  return {
    html: renderHtml(rows, args),
    json,
  };
}

function renderHtml(rows: AblationResult[], args: ResultsPageArgs): string {
  const slices = [...new Set(rows.map((row) => row.slice))];
  const backends = [...new Set(rows.map((row) => row.backend))];

  const cell = (row: AblationResult) => {
    if (row.total_cases === 0) return `<td class="empty">—</td>`;
    const pct = (row.pass_rate * 100).toFixed(1);
    return `<td>
      <div class="pass">${pct}%</div>
      <div class="meta">${Math.round(row.pass_rate * row.total_cases)}/${row.total_cases}</div>
      <div class="meta">fires ${row.total_fires}/${row.total_cases}</div>
      <div class="meta">regret ${row.abstention_regret_count} · harm ${row.false_fire_harm_count}</div>
      <div class="meta">tok/pass ${Number.isFinite(row.tokens_per_pass) ? row.tokens_per_pass.toFixed(0) : "∞"}</div>
    </td>`;
  };

  const sliceRows = slices
    .map((slice) => {
      const cells = backends
        .map((backend) => cell(rows.find((row) => row.backend === backend && row.slice === slice)!))
        .join("");
      return `<tr><th>${slice}</th>${cells}</tr>`;
    })
    .join("");

  const header = backends.map((backend) => `<th>${backend}</th>`).join("");

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${args.title ?? "OpenClawBrain — Results"}</title>
<style>
  body { font: 15px/1.45 system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #111; }
  h1 { margin-bottom: 0.25rem; }
  .sub { color: #555; margin-top: 0; }
  table { border-collapse: collapse; width: 100%; margin-top: 1.5rem; }
  th, td { border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; vertical-align: top; }
  th { background: #fafafa; }
  td.empty { color: #aaa; }
  .pass { font-size: 1.2rem; font-weight: 600; }
  .meta { font-size: 0.75rem; color: #666; }
  .note { background: #fff8e6; border: 1px solid #f0d16f; padding: 0.75rem 1rem; margin: 1rem 0; border-radius: 4px; }
</style>
</head>
<body>
<h1>${args.title ?? "Results"}</h1>
<p class="sub">Run <code>${args.run_id}</code> · generated ${new Date().toUTCString()}</p>

<div class="note">
These numbers come directly from the decision and outcome ledger. No blended averages.
Each cell shows pass rate, pass count / total cases, fire count, abstention regret and
false-fire harm counts versus the <code>none</code> baseline, and mean tokens per pass.
</div>

${args.notes ? `<p>${args.notes}</p>` : ""}

<table>
<thead><tr><th>slice</th>${header}</tr></thead>
<tbody>${sliceRows}</tbody>
</table>

<p class="meta" style="margin-top: 2rem;">
Abstention regret = cases where the <code>none</code> baseline passed, this backend did not fire, and it failed.<br>
False-fire harm = cases where the <code>none</code> baseline passed, this backend fired, and it failed.<br>
Tokens per pass = (total cases × mean total tokens) / total passes. Lower is better. ∞ means zero passes.
</p>
</body>
</html>`;
}
