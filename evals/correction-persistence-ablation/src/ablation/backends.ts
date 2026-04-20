import type { ChatTurn } from "../agent/client.js";
import type { RetrievedItem } from "../types.js";

export interface MemoryDecision {
  fire: boolean;
  retrieved: RetrievedItem[];
  injected_text: string;
  gate_score: number | null;
  gate_threshold: number | null;
}

export interface MemoryBackendImpl {
  readonly name:
    | "none"
    | "correction-only"
    | "correction-plus-heuristics"
    | "full-ocb";
  decide(history: ChatTurn[], query: string): Promise<MemoryDecision>;
}

export class NoneBackend implements MemoryBackendImpl {
  readonly name = "none" as const;

  async decide(): Promise<MemoryDecision> {
    return {
      fire: false,
      retrieved: [],
      injected_text: "",
      gate_score: null,
      gate_threshold: null,
    };
  }
}

export class CorrectionOnlyBackend implements MemoryBackendImpl {
  readonly name = "correction-only" as const;

  async decide(history: ChatTurn[], query: string): Promise<MemoryDecision> {
    const corrections = extractCorrections(history);
    const queryTerms = contentWords(query);
    const relevant = corrections.filter((c) => {
      const words = contentWords(c.content);
      for (const word of words) {
        if (queryTerms.has(word)) return true;
      }
      return false;
    });

    if (relevant.length === 0) {
      return {
        fire: false,
        retrieved: [],
        injected_text: "",
        gate_score: 0,
        gate_threshold: 0.5,
      };
    }

    const retrieved: RetrievedItem[] = relevant.map((c) => ({
      source_id: `correction-${c.turn_index}`,
      content: c.content,
      score: 1,
      age_seconds: c.age_seconds,
    }));

    return {
      fire: true,
      retrieved,
      injected_text:
        "The user previously corrected or stated a preference:\n" +
        relevant.map((c) => `- ${c.content}`).join("\n"),
      gate_score: 1,
      gate_threshold: 0.5,
    };
  }
}

export class CorrectionPlusHeuristicsBackend implements MemoryBackendImpl {
  readonly name = "correction-plus-heuristics" as const;
  private correction = new CorrectionOnlyBackend();

  async decide(history: ChatTurn[], query: string): Promise<MemoryDecision> {
    const base = await this.correction.decide(history, query);
    const queryTerms = contentWords(query);
    const alreadyUsedTurnIndexes = new Set(
      base.retrieved
        .map((item) => item.source_id.match(/^correction-(\d+)$/)?.[1])
        .filter((value): value is string => value !== undefined)
        .map((value) => Number.parseInt(value, 10)),
    );
    const seenContent = new Set(base.retrieved.map((item) => normalizeText(item.content)));

    const heuristicRetrieved: RetrievedItem[] = history
      .map((turn, index) => ({ turn, index }))
      .filter(({ turn, index }) => turn.role === "user" && !alreadyUsedTurnIndexes.has(index))
      .map(({ turn, index }) => {
        const words = contentWords(turn.content);
        let overlap = 0;
        for (const word of words) {
          if (queryTerms.has(word)) overlap++;
        }
        return { turn, index, score: overlap / Math.max(words.size, 1) };
      })
      .filter((candidate) => candidate.score >= 0.2)
      .sort((a, b) => b.score - a.score)
      .filter((candidate) => {
        const normalized = normalizeText(candidate.turn.content);
        if (seenContent.has(normalized)) return false;
        seenContent.add(normalized);
        return true;
      })
      .slice(0, 3)
      .map((candidate) => ({
        source_id: `heuristic-${candidate.index}`,
        content: candidate.turn.content,
        score: candidate.score,
        age_seconds: (history.length - candidate.index) * 60,
      }));

    const retrieved = [...base.retrieved, ...heuristicRetrieved];
    const heuristicBlock =
      heuristicRetrieved.length > 0
        ? "\n\nPossibly relevant earlier context:\n" + heuristicRetrieved.map((r) => `- ${r.content}`).join("\n")
        : "";

    return {
      fire: retrieved.length > 0,
      retrieved,
      injected_text: base.injected_text + heuristicBlock,
      gate_score: retrieved.length > 0 ? Math.max(...retrieved.map((item) => item.score)) : 0,
      gate_threshold: 0.2,
    };
  }
}

export class FullOcbBackend implements MemoryBackendImpl {
  readonly name = "full-ocb" as const;

  constructor(
    private ocb: {
      route(
        history: ChatTurn[],
        query: string,
      ): Promise<{
        fire: boolean;
        retrieved: RetrievedItem[];
        injected_text: string;
        gate_score: number;
        gate_threshold: number;
      }>;
    },
  ) {}

  async decide(history: ChatTurn[], query: string): Promise<MemoryDecision> {
    const result = await this.ocb.route(history, query);
    return {
      fire: result.fire,
      retrieved: result.retrieved,
      injected_text: result.injected_text,
      gate_score: result.gate_score,
      gate_threshold: result.gate_threshold,
    };
  }
}

interface Correction {
  turn_index: number;
  content: string;
  age_seconds: number;
}

const CORRECTION_CUES = [
  /\bnot\b/i,
  /\bdon'?t\b/i,
  /\bprefer(red)?\b/i,
  /\binstead\b/i,
  /\bactually\b/i,
  /\bi (said|told you|meant)\b/i,
  /\bstop\b/i,
  /\bno,? /i,
  /\buse .* not\b/i,
];

function extractCorrections(history: ChatTurn[]): Correction[] {
  const out: Correction[] = [];
  for (let i = 0; i < history.length; i++) {
    const turn = history[i];
    if (!turn || turn.role !== "user") continue;
    if (CORRECTION_CUES.some((re) => re.test(turn.content))) {
      out.push({
        turn_index: i,
        content: turn.content,
        age_seconds: (history.length - i) * 60,
      });
    }
  }
  return out;
}

const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "if", "then", "to", "of", "in",
  "on", "for", "with", "is", "are", "was", "were", "be", "been", "being",
  "do", "does", "did", "have", "has", "had", "i", "you", "he", "she", "it",
  "we", "they", "this", "that", "these", "those", "can", "could", "would",
  "should", "will", "shall", "may", "might", "me", "my", "your", "our",
  "their", "as", "at", "by", "so", "not", "no",
]);

function contentWords(text: string): Set<string> {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2 && !STOPWORDS.has(word));
  return new Set(words);
}

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}
