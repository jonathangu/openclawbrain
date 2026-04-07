export function scanSession(rawSession, options = {}) {
  const extractTextContent = (content) => {
    if (typeof content === "string") {
      const trimmed = content.trim();
      return trimmed.length > 0 ? trimmed : null;
    }
    if (Array.isArray(content)) {
      const text = content
        .map((part) => {
          if (part && typeof part === "object") {
            if (typeof part.text === "string") {
              return part.text;
            }
            if (typeof part.content === "string") {
              return part.content;
            }
          }
          return null;
        })
        .filter((piece) => typeof piece === "string" && piece.trim().length > 0)
        .join(" ")
        .trim();
      return text.length > 0 ? text : null;
    }
    return null;
  };

  const records = Array.isArray(rawSession?.records)
    ? rawSession.records
    : Array.isArray(rawSession?.events)
      ? rawSession.events
      : Array.isArray(rawSession?.messages)
        ? rawSession.messages
        : [];
  const session = rawSession?.session ?? {
    agentId: rawSession?.agentId ?? options.defaultAgentId ?? "main",
    sessionId: rawSession?.sessionId ?? rawSession?.id ?? "unknown-session",
    channel: rawSession?.channel ?? options.defaultChannel ?? "telegram",
    source: rawSession?.source ?? options.defaultSourceStream ?? "unknown",
  };
  const events = records.map((record, index) => {
    const message = record?.message ?? record;
    const role = message?.role ?? record?.role ?? record?.actorRole;
    const content = extractTextContent(message?.content ?? record?.content);
    const createdAt = record?.timestamp ?? message?.timestamp ?? options.observedAt ?? null;
    if (role === "assistant" || role === "system") {
      return {
        eventId: record?.id ?? record?.recordId ?? `assistant-${index}`,
        actor: role === "system" ? "system" : "assistant",
        kind: "assistant_turn",
        contentFormat: typeof content === "string" ? "text" : "unknown",
        content,
        sequence: index,
        createdAt,
        messageId: record?.id ?? record?.messageId ?? null,
      };
    }
    if (role === "user") {
      return {
        eventId: record?.id ?? record?.recordId ?? `user-${index}`,
        actor: "user",
        kind: "user_turn",
        contentFormat: typeof content === "string" ? "text" : "unknown",
        content,
        sequence: index,
        createdAt,
        messageId: record?.id ?? record?.messageId ?? null,
      };
    }
    return null;
  }).filter(Boolean);
  return { session, events, warnings: [] };
}
