export function isActivationRootPlaceholder(activationRoot) {
    return activationRoot === "__ACTIVATION_" + "ROOT__" || activationRoot.trim().length === 0;
}
export function validateExtensionRegistrationApi(api) {
    if (!isRecord(api) || typeof api.on !== "function") {
        return {
            ok: false,
            diagnostic: {
                key: "registration-api-invalid",
                once: true,
                message: `[openclawbrain] extension inactive: host registration API is missing api.on(event, handler, options) ` +
                    `(received=${describeValue(api)})`
            }
        };
    }
    return {
        ok: true,
        api: api
    };
}
export function normalizePromptBuildEvent(event) {
    if (!isRecord(event)) {
        return {
            ok: false,
            diagnostic: failOpenDiagnostic("runtime-event-not-object", "before_prompt_build event is not an object", `event=${describeValue(event)}`)
        };
    }
    const messages = event.messages;
    if (!Array.isArray(messages)) {
        return {
            ok: false,
            diagnostic: failOpenDiagnostic("runtime-messages-not-array", "before_prompt_build event.messages is not an array", `event=${describeValue(event)} messages=${describeValue(messages)}`)
        };
    }
    const warnings = [];
    const sessionId = normalizeOptionalScalarField(event.sessionId, "sessionId", warnings);
    const channel = normalizeOptionalScalarField(event.channel, "channel", warnings);
    const promptFallback = extractTextContent(event.prompt);
    let extractedMessage = promptFallback ?? "";
    if (messages.length === 0) {
        if (extractedMessage.length === 0) {
            warnings.push(failOpenDiagnostic("runtime-messages-empty", "before_prompt_build event.messages is empty", `event=${describeValue(event)}`));
        }
    }
    else {
        const lastMessage = messages.at(-1);
        extractedMessage = extractPromptMessage(lastMessage) ?? promptFallback ?? "";
        if (extractedMessage.length === 0) {
            warnings.push(failOpenDiagnostic("runtime-last-message-invalid", "before_prompt_build last message has no usable text content", `lastMessage=${describeValue(lastMessage)}`));
        }
    }
    return {
        ok: true,
        event: {
            message: extractedMessage,
            ...(sessionId !== undefined ? { sessionId } : {}),
            ...(channel !== undefined ? { channel } : {}),
            warnings
        }
    };
}
export function createBeforePromptBuildHandler(input) {
    return async (event, _ctx) => {
        if (isActivationRootPlaceholder(input.activationRoot)) {
            await input.reportDiagnostic({
                key: "activation-root-placeholder",
                once: true,
                message: "[openclawbrain] BRAIN NOT YET LOADED: ACTIVATION_ROOT is still a placeholder. Run: openclawbrain install --openclaw-home <path>"
            });
            return {};
        }
        const normalized = normalizePromptBuildEvent(event);
        if (!normalized.ok) {
            await input.reportDiagnostic(normalized.diagnostic);
            return {};
        }
        for (const warning of normalized.event.warnings) {
            await input.reportDiagnostic(warning);
        }
        if (normalized.event.message.length === 0) {
            // A partial before_prompt_build envelope has nothing usable to compile, so
            // fail open here instead of spending hot-path time in compile/log I/O.
            return {};
        }
        try {
            const result = input.compileRuntimeContext({
                activationRoot: input.activationRoot,
                message: normalized.event.message,
                ...(normalized.event.sessionId !== undefined ? { sessionId: normalized.event.sessionId } : {}),
                ...(normalized.event.channel !== undefined ? { channel: normalized.event.channel } : {}),
                ...(input.extensionEntryPath === undefined
                    ? {}
                    : {
                        _serveRouteBreadcrumbs: {
                            invocationSurface: "installed_extension_before_prompt_build",
                            hostEvent: "before_prompt_build",
                            installedEntryPath: input.extensionEntryPath
                        }
                    })
            });
            if (!result.ok) {
                const mode = result.hardRequirementViolated ? "hard-fail" : "fail-open";
                await input.reportDiagnostic({
                    key: `compile-${mode}`,
                    message: `[openclawbrain] ${mode}: ${result.error} ` +
                        `(activationRoot=${input.activationRoot}, sessionId=${normalized.event.sessionId ?? "unknown"}, channel=${normalized.event.channel ?? "unknown"})`
                });
                return {};
            }
            if (result.brainContext.length > 0) {
                input.debug?.(`[openclawbrain] compiled context, chars: ${result.brainContext.length}`);
                return {
                    appendSystemContext: result.brainContext
                };
            }
        }
        catch (error) {
            const detail = error instanceof Error ? error.stack ?? error.message : String(error);
            await input.reportDiagnostic({
                key: "compile-threw",
                message: `[openclawbrain] compile threw: ${detail} ` +
                    `(activationRoot=${input.activationRoot}, sessionId=${normalized.event.sessionId ?? "unknown"}, channel=${normalized.event.channel ?? "unknown"})`
            });
        }
        return {};
    };
}
function failOpenDiagnostic(key, reason, detail) {
    return {
        key,
        message: `[openclawbrain] fail-open: ${reason} (${detail})`
    };
}
function normalizeOptionalScalarField(value, fieldName, warnings) {
    if (value === undefined || value === null) {
        return undefined;
    }
    if (typeof value === "string") {
        const trimmed = value.trim();
        return trimmed.length > 0 ? trimmed : undefined;
    }
    if (typeof value === "number" || typeof value === "bigint" || typeof value === "boolean") {
        return String(value);
    }
    warnings.push({
        key: `runtime-${fieldName}-ignored`,
        message: `[openclawbrain] fail-open: ignored unsupported before_prompt_build ${fieldName} ` +
            `(${fieldName}=${describeValue(value)})`
    });
    return undefined;
}
function extractPromptMessage(message) {
    if (typeof message === "string") {
        return normalizeText(message);
    }
    if (!isRecord(message)) {
        return undefined;
    }
    return extractTextContent(message.content);
}
function extractTextContent(content) {
    if (typeof content === "string") {
        return normalizeText(content);
    }
    if (Array.isArray(content)) {
        const parts = content
            .map((part) => extractTextPart(part))
            .filter((part) => part !== undefined);
        return parts.length > 0 ? parts.join("\n") : undefined;
    }
    if (isRecord(content)) {
        if (typeof content.text === "string") {
            return normalizeText(content.text);
        }
        if (typeof content.content === "string") {
            return normalizeText(content.content);
        }
    }
    return undefined;
}
function extractTextPart(part) {
    if (typeof part === "string") {
        return normalizeText(part);
    }
    if (!isRecord(part)) {
        return undefined;
    }
    if ("text" in part && typeof part.text === "string") {
        return normalizeText(part.text);
    }
    if ("content" in part && typeof part.content === "string") {
        return normalizeText(part.content);
    }
    return undefined;
}
function normalizeText(value) {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
}
function describeValue(value) {
    if (value === null) {
        return "null";
    }
    if (Array.isArray(value)) {
        const itemKinds = Array.from(new Set(value.slice(0, 4).map((entry) => {
            if (entry === null) {
                return "null";
            }
            if (Array.isArray(entry)) {
                return "array";
            }
            return typeof entry;
        })));
        const suffix = value.length > 4 ? ",..." : "";
        return `array(len=${value.length}, itemKinds=${itemKinds.join("|") || "none"}${suffix})`;
    }
    if (typeof value === "string") {
        const preview = value.replace(/\s+/g, " ").trim().slice(0, 48);
        const suffix = value.trim().length > 48 ? "..." : "";
        return `string(len=${value.length}, preview=${JSON.stringify(preview + suffix)})`;
    }
    if (typeof value === "object") {
        const keys = Object.keys(value).slice(0, 6);
        const suffix = Object.keys(value).length > 6 ? ",..." : "";
        return `object(keys=${keys.join(",") || "none"}${suffix})`;
    }
    if (typeof value === "function") {
        return "function";
    }
    if (typeof value === "symbol") {
        return `symbol(${String(value.description ?? "")})`;
    }
    return `${typeof value}(${String(value)})`;
}
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
//# sourceMappingURL=runtime-guard.js.map