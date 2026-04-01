import type {
  BrainContextUsefulnessEvaluationV1,
  BrainContextUsefulnessFollowUpClass,
  BrainContextUsefulnessRouteIntegrityClass,
  BrainContextUsefulnessSignal,
  BrainContextUsefulnessTeacherAlignmentClass,
  BrainContextUsefulnessToolOutcomeClass,
  BrainContextUsefulnessVerdict,
  BrainObservation,
  BrainObservationTeacherEvaluation,
  BrainObservationToolResult,
} from "./types.js";

const FOLLOW_UP_WEIGHT = 0.45;
const TOOL_OUTCOME_WEIGHT = 0.35;
const ROUTE_INTEGRITY_WEIGHT = 0.15;
const TEACHER_ALIGNMENT_WEIGHT = 0.05;

const HELPFUL_MIN = 0.35;
const HARMFUL_MAX = -0.35;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toTrimmedText(value: string | null | undefined): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeLower(value: string | null | undefined): string {
  return toTrimmedText(value).toLowerCase();
}

function snippet(value: string | null | undefined, maxLength = 96): string | null {
  const text = toTrimmedText(value);
  if (!text) {
    return null;
  }
  return text.length <= maxLength ? text : `${text.slice(0, maxLength - 1)}…`;
}

function signal(
  className: string,
  score: number,
  evidence: string[],
  detail: string | null,
): BrainContextUsefulnessSignal {
  return {
    class: className,
    score,
    evidence,
    detail,
  };
}

function containsAny(text: string, phrases: string[]): string | null {
  for (const phrase of phrases) {
    if (text.includes(phrase)) {
      return phrase;
    }
  }
  return null;
}

function classifyFollowUpText(followUpText: string | null | undefined): BrainContextUsefulnessSignal & {
  class: BrainContextUsefulnessFollowUpClass;
} {
  const normalized = normalizeLower(followUpText);
  if (!normalized) {
    return signal("missing", 0, [], "no follow-up text") as BrainContextUsefulnessSignal & {
      class: BrainContextUsefulnessFollowUpClass;
    };
  }

  const correctionPhrase = containsAny(normalized, [
    "no,",
    "no.",
    "no ",
    "wrong",
    "incorrect",
    "actually",
    "not right",
    "doesn't work",
    "didn't work",
    "did not work",
    "that didn't work",
    "that does not work",
    "that doesn't work",
    "still broken",
    "broken",
    "failed",
  ]);
  if (correctionPhrase) {
    return signal(
      "correction",
      -0.8,
      [snippet(followUpText) ?? correctionPhrase],
      `follow-up matched correction phrase: ${correctionPhrase}`,
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessFollowUpClass };
  }

  const contradictionPhrase = containsAny(normalized, [
    "that's not",
    "that is not",
    "that isn't",
    "you said",
    "you told me",
    "contrary",
    "actually,",
    "actually ",
  ]);
  if (contradictionPhrase) {
    return signal(
      "contradiction",
      -1.0,
      [snippet(followUpText) ?? contradictionPhrase],
      `follow-up matched contradiction phrase: ${contradictionPhrase}`,
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessFollowUpClass };
  }

  const reaskPhrase = containsAny(normalized, [
    "can you",
    "could you",
    "try again",
    "another way",
    "what about",
    "how do i",
    "how can i",
    "please",
    "still need",
    "need you to",
    "can we",
  ]);
  if (reaskPhrase || normalized.includes("?")) {
    return signal(
      "reask",
      -0.3,
      [snippet(followUpText) ?? reaskPhrase ?? "question mark detected"],
      reaskPhrase ? `follow-up matched re-ask phrase: ${reaskPhrase}` : "follow-up looks like a re-ask",
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessFollowUpClass };
  }

  const confirmationPhrase = containsAny(normalized, [
    "exactly right",
    "that's right",
    "that worked",
    "worked",
    "fixed it",
    "solved",
    "perfect",
    "great",
    "nice",
    "thanks",
    "thank you",
    "good",
    "awesome",
  ]);
  if (confirmationPhrase) {
    const score = containsAny(normalized, ["thanks", "thank you", "ok", "okay"]) ? 0.2 : 0.8;
    return signal(
      "confirmation",
      score,
      [snippet(followUpText) ?? confirmationPhrase],
      `follow-up matched confirmation phrase: ${confirmationPhrase}`,
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessFollowUpClass };
  }

  return signal(
    "neutral_ack",
    0.2,
    [snippet(followUpText) ?? normalized],
    "follow-up is acknowledged but not strongly positive or negative",
  ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessFollowUpClass };
}

function classifyToolOutcome(toolResults: BrainObservationToolResult[] | null | undefined): BrainContextUsefulnessSignal & {
  class: BrainContextUsefulnessToolOutcomeClass;
} {
  const results = (toolResults ?? []).filter((result) => result.sourceRole === "tool");
  if (results.length === 0) {
    return signal("missing", 0, [], "no tool results") as BrainContextUsefulnessSignal & {
      class: BrainContextUsefulnessToolOutcomeClass;
    };
  }

  const excerpts = results
    .map((result) => result.excerpt ?? result.output ?? result.input)
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .map((value) => snippet(value) ?? value);

  const normalizedHaystack = results
    .flatMap((result) => [result.output, result.excerpt, result.input])
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .map((value) => value.toLowerCase())
    .join("\n");

  const failurePhrase = containsAny(normalizedHaystack, [
    "error",
    "failed",
    "failure",
    "exception",
    "enoent",
    "eacces",
    "eperm",
    "not found",
    "exitcode\":1",
    "exitcode\":2",
    "exit code 1",
    "exit code 2",
    "status 1",
    "status 2",
    "traceback",
  ]);
  if (results.some((result) => result.isError) || failurePhrase) {
    return signal(
      failurePhrase ? "error" : "failure",
      -0.7,
      excerpts.slice(0, 3),
      failurePhrase ? `tool output matched error phrase: ${failurePhrase}` : "tool result marked as error",
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessToolOutcomeClass };
  }

  const successPhrase = containsAny(normalizedHaystack, [
    "exitcode\":0",
    "exit code 0",
    "\"ok\":true",
    "\"success\":true",
    "success",
    "succeeded",
    "done",
    "passed",
    "created",
    "updated",
    "wrote",
    "installed",
  ]);
  if (successPhrase) {
    return signal(
      "success",
      0.7,
      excerpts.slice(0, 3),
      `tool output matched success phrase: ${successPhrase}`,
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessToolOutcomeClass };
  }

  return signal(
    "partial",
    0.2,
    excerpts.slice(0, 3),
    "tool results were present but only weakly positive",
  ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessToolOutcomeClass };
}

function hasClippingSignal(selectionMetadata: Record<string, unknown> | null | undefined): boolean {
  if (!selectionMetadata) {
    return false;
  }
  return Boolean(
    selectionMetadata.contextClipped
    || selectionMetadata.compileDeadlineHit
    || selectionMetadata.servedPartial
    || selectionMetadata.queryInterrupted
    || selectionMetadata.interruption
    || selectionMetadata.interruptionStage
    || selectionMetadata.interruptionReason
  );
}

function classifyRouteIntegrity(observation: BrainObservation): BrainContextUsefulnessSignal & {
  class: BrainContextUsefulnessRouteIntegrityClass;
} {
  const selectionMetadata = observation.routeMetadata.selectionMetadata as Record<string, unknown> | null | undefined;
  const bindingMode = observation.routeMetadata.bindingMode;
  if (bindingMode === "unbound") {
    return signal(
      "unbound",
      -0.5,
      [bindingMode],
      "observation is unbound from the served route",
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessRouteIntegrityClass };
  }

  if (hasClippingSignal(selectionMetadata)) {
    const evidence = [
      bindingMode ?? "unknown-binding",
      String(selectionMetadata?.servedPartial ?? false),
      String(selectionMetadata?.contextClipped ?? false),
      String(selectionMetadata?.compileDeadlineHit ?? false),
    ];
    return signal(
      "clipped_or_fail_open",
      -0.35,
      evidence,
      "served route clipped, interrupted, or fail-open adjacent",
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessRouteIntegrityClass };
  }

  if (
    bindingMode === "exact_decision_id"
    || bindingMode === "exact_selection_digest"
    || bindingMode === "turn_compile_event_id"
  ) {
    return signal(
      "exact_full_serve",
      0.25,
      [bindingMode],
      "route bound exactly with no clipping signal",
    ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessRouteIntegrityClass };
  }

  return signal(
    "fallback",
    0,
    [bindingMode ?? "unknown-binding"],
    "route binding fell back to a weaker attachment",
  ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessRouteIntegrityClass };
}

function classifyTeacherAlignment(
  teacherEvaluation: BrainObservationTeacherEvaluation | null | undefined,
): (BrainContextUsefulnessSignal & { class: BrainContextUsefulnessTeacherAlignmentClass }) | null {
  if (!teacherEvaluation) {
    return null;
  }

  const score = clamp(teacherEvaluation.finalScore, -1, 1);
  return signal(
    "calibration",
    score,
    [
      `teacher_final_score=${teacherEvaluation.finalScore}`,
      `teacher_confidence=${teacherEvaluation.confidence}`,
    ],
    teacherEvaluation.reason || "teacher evaluation calibration",
  ) as BrainContextUsefulnessSignal & { class: BrainContextUsefulnessTeacherAlignmentClass };
}

function buildReason(parts: string[], authorityGateBlocked: boolean): string {
  const text = parts.filter((part) => part.trim().length > 0).join(", ");
  return authorityGateBlocked ? `${text}; human follow-up blocks promotion` : text;
}

function computeConfidence(params: {
  followUpClass: BrainContextUsefulnessFollowUpClass;
  toolOutcomeClass: BrainContextUsefulnessToolOutcomeClass;
  routeIntegrityClass: BrainContextUsefulnessRouteIntegrityClass;
  teacherAlignment: BrainContextUsefulnessSignal | null;
}): number {
  let confidence = 0.15;
  if (params.followUpClass !== "missing") {
    confidence += 0.35;
  }
  if (params.toolOutcomeClass !== "missing") {
    confidence += 0.25;
  }
  if (params.routeIntegrityClass === "exact_full_serve") {
    confidence += 0.1;
  } else if (params.routeIntegrityClass === "clipped_or_fail_open") {
    confidence -= 0.05;
  } else if (params.routeIntegrityClass === "unbound") {
    confidence -= 0.1;
  }
  if (params.teacherAlignment) {
    confidence += 0.1;
  }
  if (params.followUpClass === "confirmation" && params.toolOutcomeClass === "success") {
    confidence += 0.1;
  }
  if (params.followUpClass === "correction" || params.followUpClass === "contradiction") {
    confidence += 0.05;
  }
  return clamp(confidence, 0.05, 1);
}

export function evaluateContextUsefulness(observation: BrainObservation): BrainContextUsefulnessEvaluationV1 {
  const followUp = classifyFollowUpText(observation.followUpText);
  const toolOutcome = classifyToolOutcome(observation.toolResults);
  const routeIntegrity = classifyRouteIntegrity(observation);
  const teacherAlignment = classifyTeacherAlignment(observation.teacherEvaluation);

  const weightedRaw = (followUp.score * FOLLOW_UP_WEIGHT)
    + (toolOutcome.score * TOOL_OUTCOME_WEIGHT)
    + (routeIntegrity.score * ROUTE_INTEGRITY_WEIGHT)
    + ((teacherAlignment?.score ?? 0) * TEACHER_ALIGNMENT_WEIGHT);

  let finalScore = clamp(weightedRaw, -1, 1);
  const authorityGateBlocked = followUp.class === "correction" || followUp.class === "contradiction";
  if (authorityGateBlocked && finalScore > HARMFUL_MAX) {
    finalScore = HARMFUL_MAX;
  }
  if (
    !authorityGateBlocked
    && followUp.class === "confirmation"
    && toolOutcome.class === "success"
    && routeIntegrity.class === "exact_full_serve"
    && finalScore < HELPFUL_MIN
  ) {
    finalScore = HELPFUL_MIN;
  }

  const confidence = computeConfidence({
    followUpClass: followUp.class,
    toolOutcomeClass: toolOutcome.class,
    routeIntegrityClass: routeIntegrity.class,
    teacherAlignment,
  });

  const verdict: BrainContextUsefulnessVerdict = finalScore >= HELPFUL_MIN
    ? "helpful"
    : finalScore <= HARMFUL_MAX
      ? "harmful"
      : "irrelevant";

  const reason = buildReason([
    `follow-up=${followUp.class}`,
    `tool=${toolOutcome.class}`,
    `route=${routeIntegrity.class}`,
    teacherAlignment ? `teacher=${teacherAlignment.score.toFixed(2)}` : "teacher=none",
    `score=${finalScore.toFixed(2)}`,
    `confidence=${confidence.toFixed(2)}`,
  ], authorityGateBlocked);

  return {
    version: 1,
    observationId: observation.id,
    episodeId: observation.episodeId,
    traceId: observation.traceId,
    conversationId: observation.conversationId,
    bindingMode: observation.routeMetadata.bindingMode,
    signals: {
      followUp,
      toolOutcome,
      routeIntegrity,
      teacherAlignment,
      authorityGate: {
        blocked: authorityGateBlocked,
        reason: authorityGateBlocked ? "human follow-up correction/contradiction dominates shadow score" : null,
      },
    },
    finalScore,
    confidence,
    verdict,
    reason,
    computedAt: Date.now(),
  };
}

export function verdictFromScore(score: number): BrainContextUsefulnessVerdict {
  if (score >= HELPFUL_MIN) {
    return "helpful";
  }
  if (score <= HARMFUL_MAX) {
    return "harmful";
  }
  return "irrelevant";
}

export function usefulnessThresholds(): { helpfulMin: number; harmfulMax: number } {
  return {
    helpfulMin: HELPFUL_MIN,
    harmfulMax: HARMFUL_MAX,
  };
}
