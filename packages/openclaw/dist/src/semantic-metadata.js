const QUESTIONNAIRE_KEYWORD_PATTERN = /\b(?:questionnaire|question list|list of questions|discovery questions|intake questions)\b/u;
const QUESTION_ROUTING_PATTERNS = [
    /\b(?:send|give|pass|forward|route)\b.{0,40}\b(?:these|the following|this)\b.{0,24}\b(?:questions?|questionnaire|question list|checklist)\b/u,
    /\b(?:ask|walk|take|run)\b.{0,32}\b(?:through|over)\b.{0,24}\b(?:these|the following|this)\b.{0,24}\b(?:questions?|questionnaire|checklist)\b/u,
    /\b(?:ask|send)\b.{0,16}\beagle\b.{0,32}\b(?:questions?|questionnaire)\b/u,
    /\bone question at a time\b/u
];
const NUMBERED_QUESTION_PATTERN = /(?:^|\n|\s)(?:\d+[.)]|[-*])\s*(?:what|when|why|how|which|who|can|could|should|would|is|are|do|does|did)\b/gmu;
const QUESTION_SCAFFOLDING_CONTEXT_PATTERN = /\b(?:question|questions|ask|before answering|before you answer|intake|discovery)\b/u;
function normalizeSemanticContent(value) {
    return (value ?? "").replace(/\s+/gu, " ").trim().toLowerCase();
}
function interactionSemanticType(kind) {
    switch (kind) {
        case "memory_compiled":
            return "observability_residue";
        case "message_delivered":
            return "delivery_residue";
        case "operator_override":
            return "control_signal";
    }
}
function interactionDiagnosticIntent(kind) {
    switch (kind) {
        case "memory_compiled":
            return "compile_observability";
        case "message_delivered":
            return "delivery_observability";
        case "operator_override":
            return undefined;
    }
}
export function buildInteractionSemanticMetadata(sourceKind, kind) {
    const diagnosticIntent = interactionDiagnosticIntent(kind);
    return {
        semanticType: interactionSemanticType(kind),
        sourceKind,
        ...(diagnosticIntent === undefined ? {} : { diagnosticIntent })
    };
}
export function buildAssistantMessageSemanticMetadata() {
    return {
        semanticType: "memory_candidate",
        sourceKind: "session_store"
    };
}
export function isInstructionalScaffoldingContent(kind, content) {
    if (kind !== "teaching") {
        return false;
    }
    const normalized = normalizeSemanticContent(content);
    if (normalized.length === 0) {
        return false;
    }
    if (QUESTIONNAIRE_KEYWORD_PATTERN.test(normalized)) {
        return true;
    }
    if (QUESTION_ROUTING_PATTERNS.some((pattern) => pattern.test(normalized))) {
        return true;
    }
    const numberedQuestions = [...normalized.matchAll(NUMBERED_QUESTION_PATTERN)].length;
    return numberedQuestions >= 2 && QUESTION_SCAFFOLDING_CONTEXT_PATTERN.test(normalized);
}
export function buildFeedbackSemanticMetadata(sourceKind, kind, content) {
    return {
        semanticType: isInstructionalScaffoldingContent(kind, content) ? "instructional_scaffolding" : "teacher_signal",
        sourceKind
    };
}
