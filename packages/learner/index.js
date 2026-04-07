export function describeSparseFeedbackEventDispositions(feedbackEvents = [], observedAt = null, sparseFeedback = undefined) {
  return (Array.isArray(feedbackEvents) ? feedbackEvents : []).map((event) => ({
    eventId: event?.eventId ?? null,
    kind: event?.kind ?? null,
    observedAt,
    sparseFeedback: sparseFeedback ?? null,
    disposition: "retained",
  }));
}
