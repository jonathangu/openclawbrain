import type { ChatTurn } from "../src/agent/client.js";
import {
  CorrectionOnlyBackend,
  CorrectionPlusHeuristicsBackend,
  NoneBackend,
} from "../src/ablation/backends.js";
import { grade } from "../src/eval/correction-recurrence.js";

let failed = 0;

function assert(condition: boolean, label: string): void {
  if (!condition) {
    console.error("FAIL:", label);
    failed++;
  } else {
    console.log("ok  :", label);
  }
}

assert(
  grade("Sure, here's TypeScript: const x: string = 'hi';", {
    type: "contains_any",
    positive_signals: [": string", "interface"],
    negative_signals: ["```javascript"],
  }),
  "grader contains_any positive match, no negatives",
);

assert(
  !grade("Here's the JS:\n```javascript\nconst x = 1;\n```", {
    type: "contains_any",
    positive_signals: [": string"],
    negative_signals: ["```javascript"],
  }),
  "grader contains_any rejects when negative present",
);

assert(
  grade("Cheers, Jonathan", {
    type: "contains_all",
    positive_signals: ["jonathan"],
  }),
  "grader contains_all positive",
);

assert(
  grade("def reverse(s):\n    return s[::-1]", {
    type: "regex",
    pattern: "def\\s+\\w+\\s*\\(",
    must_not_match: "^\\s*#|^\\s*\"\"\"|^\\s*'''",
  }),
  "grader regex pass without comments",
);

assert(
  !grade("def reverse(s):\n    return s[::-1]\n# trailing explanation", {
    type: "regex",
    pattern: "def\\s+\\w+\\s*\\(",
    must_not_match: "^\\s*#|^\\s*\"\"\"|^\\s*'''",
  }),
  "grader regex must_not_match triggers on later comment lines too",
);

const history: ChatTurn[] = [
  { role: "user", content: "Rule: don't use JavaScript. I want TypeScript, not JavaScript." },
  { role: "assistant", content: "Understood." },
  { role: "user", content: "What's the weather like?" },
  { role: "assistant", content: "Varies by region." },
];

const none = await new NoneBackend().decide(history, "Write me a URL parser.");
assert(!none.fire && none.injected_text === "", "NoneBackend never fires");

const correctionOnly = await new CorrectionOnlyBackend().decide(history, "Write me a URL parser in javascript.");
assert(correctionOnly.fire, "CorrectionOnlyBackend fires when query terms overlap a correction");
assert(correctionOnly.injected_text.toLowerCase().includes("typescript"), "CorrectionOnlyBackend injects the prior correction");

const unrelated = await new CorrectionOnlyBackend().decide(history, "What's the capital of France?");
assert(!unrelated.fire, "CorrectionOnlyBackend stays quiet when query is unrelated");

const heuristic = await new CorrectionPlusHeuristicsBackend().decide(history, "Tell me about the weather in Paris.");
assert(heuristic.fire, "CorrectionPlusHeuristics fires on keyword overlap alone");

const deduped = await new CorrectionPlusHeuristicsBackend().decide(history, "Use javascript for the parser.");
assert(
  deduped.retrieved.filter((item) => item.content.toLowerCase().includes("typescript")).length === 1,
  "CorrectionPlusHeuristics does not duplicate a deterministic correction retrieval",
);

console.log(failed === 0 ? "\nall checks passed" : `\n${failed} failures`);
process.exit(failed === 0 ? 0 : 1);
