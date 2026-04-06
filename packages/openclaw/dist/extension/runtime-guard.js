import { existsSync } from "node:fs";
import path from "node:path";

export function isActivationRootPlaceholder(activationRoot) {
    return activationRoot === "__ACTIVATION_" + "ROOT__" || activationRoot.trim().length === 0;
}
function resolveOpenClawHomeFromExtensionEntryPath(extensionEntryPath) {
    let currentPath = path.resolve(extensionEntryPath);
    while (true) {
        const parentPath = path.dirname(currentPath);
        if (parentPath === currentPath) {
            break;
        }
        if (path.basename(parentPath) === "extensions") {
            return path.dirname(parentPath);
        }
        currentPath = parentPath;
    }
    return null;
}
function resolveFallbackActivationRootFromExtensionEntryPath(extensionEntryPath) {
    const openclawHome = resolveOpenClawHomeFromExtensionEntryPath(extensionEntryPath);
    if (openclawHome === null) {
        return null;
    }
    const candidate = path.resolve(path.join(path.dirname(openclawHome), ".openclawbrain", "activation"));
    return existsSync(candidate) ? candidate : null;
}
export function resolveInstalledActivationRoot(input) {
    const rawActivationRoot = input.activationRoot;
    if (!isActivationRootPlaceholder(rawActivationRoot)) {
        return {
            activationRoot: path.resolve(rawActivationRoot),
            recoveredFromPlaceholder: false
        };
    }
    if (typeof input.extensionEntryPath === "string" && input.extensionEntryPath.trim().length > 0) {
        const fallbackActivationRoot = resolveFallbackActivationRootFromExtensionEntryPath(input.extensionEntryPath);
        if (fallbackActivationRoot !== null) {
            return {
                activationRoot: fallbackActivationRoot,
                recoveredFromPlaceholder: true
            };
        }
    }
    return {
        activationRoot: rawActivationRoot,
        recoveredFromPlaceholder: false
    };
}
export function validateExtensionRegistrationApi(api) {
    if (!isRecord(api) || typeof api.on !== "function") {
        return {
            ok: false,
            diagnostic: shapeDiagnostic({
                key: "registration-api-invalid",
                once: true,
                message: `[openclawbrain] extension inactive: host registration API is missing api.on(event, handler, options) ` +
                    `(received=${describeValue(api)})`
            })
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
    const maxContextChars = normalizeOptionalNonNegativeIntegerField(event.maxContextChars, "maxContextChars", warnings);
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
            ...(maxContextChars !== undefined ? { maxContextChars } : {}),
            ...(sessionId !== undefined ? { sessionId } : {}),
            ...(channel !== undefined ? { channel } : {}),
            warnings
        }
    };
}
export function createBeforePromptBuildHandler(input) {
    const resolvedActivationRoot = resolveInstalledActivationRoot({
        activationRoot: input.activationRoot,
        extensionEntryPath: input.extensionEntryPath
    });
    return async (event, _ctx) => {
        if (isActivationRootPlaceholder(resolvedActivationRoot.activationRoot)) {
            await input.reportDiagnostic(shapeDiagnostic({
                key: "activation-root-placeholder",
                once: true,
                message: "[openclawbrain] BRAIN NOT YET LOADED: ACTIVATION_ROOT is still a placeholder. Install OpenClawBrain, then run: openclawbrain install --openclaw-home <path>"
            }));
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
                activationRoot: resolvedActivationRoot.activationRoot,
                message: normalized.event.message,
                ...(normalized.event.maxContextChars !== undefined ? { maxContextChars: normalized.event.maxContextChars } : {}),
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
                await input.reportDiagnostic(shapeDiagnostic({
                    key: `compile-${mode}`,
                    message: `[openclawbrain] ${mode}: ${result.error} ` +
                        `(activationRoot=${resolvedActivationRoot.activationRoot}, sessionId=${normalized.event.sessionId ?? "unknown"}, channel=${normalized.event.channel ?? "unknown"})`
                }));
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
            await input.reportDiagnostic(shapeDiagnostic({
                key: "compile-threw",
                message: `[openclawbrain] compile threw: ${detail} ` +
                    `(activationRoot=${resolvedActivationRoot.activationRoot}, sessionId=${normalized.event.sessionId ?? "unknown"}, channel=${normalized.event.channel ?? "unknown"})`
            }));
        }
        return {};
    };
}
function failOpenDiagnostic(key, reason, detail) {
    return shapeDiagnostic({
        key,
        message: `[openclawbrain] fail-open: ${reason} (${detail})`
    });
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
    warnings.push(shapeDiagnostic({
        key: `runtime-${fieldName}-ignored`,
        message: `[openclawbrain] fail-open: ignored unsupported before_prompt_build ${fieldName} ` +
            `(${fieldName}=${describeValue(value)})`
    }));
    return undefined;
}
function normalizeOptionalNonNegativeIntegerField(value, fieldName, warnings) {
    if (value === undefined || value === null) {
        return undefined;
    }
    if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) {
        return value;
    }
    if (typeof value === "bigint" && value >= 0n && value <= BigInt(Number.MAX_SAFE_INTEGER)) {
        return Number(value);
    }
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (trimmed.length === 0) {
            return undefined;
        }
        if (/^\d+$/.test(trimmed)) {
            const parsed = Number(trimmed);
            if (Number.isSafeInteger(parsed)) {
                return parsed;
            }
        }
    }
    warnings.push(shapeDiagnostic({
        key: `runtime-${fieldName}-ignored`,
        message: `[openclawbrain] fail-open: ignored unsupported before_prompt_build ${fieldName} ` +
            `(${fieldName}=${describeValue(value)})`
    }));
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
function shapeDiagnostic(diagnostic) {
    if (diagnostic.severity !== undefined &&
        diagnostic.actionability !== undefined &&
        diagnostic.summary !== undefined &&
        diagnostic.action !== undefined) {
        return diagnostic;
    }
    if (diagnostic.key === "activation-root-placeholder") {
        return {
            ...diagnostic,
            severity: "blocking",
            actionability: "rerun_install",
            summary: "extension hook is installed but ACTIVATION_ROOT is still unpinned",
            action: "Run openclawbrain install --openclaw-home <path> to pin the runtime hook."
        };
    }
    if (diagnostic.key === "registration-api-invalid") {
        return {
            ...diagnostic,
            severity: "blocking",
            actionability: "inspect_host_registration_api",
            summary: "extension host registration API is missing or incompatible",
            action: "Repair or upgrade the host extension API so api.on(event, handler, options) is available."
        };
    }
    if (diagnostic.key === "compile-hard-fail") {
        return {
            ...diagnostic,
            severity: "blocking",
            actionability: "inspect_runtime_compile",
            summary: "brain context compile hit a hard requirement",
            action: "Inspect the activation root and compile error; rerun install if the pinned hook may be stale."
        };
    }
    if (diagnostic.key === "compile-fail-open" || diagnostic.key === "compile-threw") {
        return {
            ...diagnostic,
            severity: "degraded",
            actionability: "inspect_runtime_compile",
            summary: diagnostic.key === "compile-threw"
                ? "brain context compile threw during before_prompt_build"
                : "brain context compile failed open during before_prompt_build",
            action: "Inspect the activation root and compile error if brain context is unexpectedly empty."
        };
    }
    if (diagnostic.key.startsWith("runtime-")) {
        return {
            ...diagnostic,
            severity: "degraded",
            actionability: "inspect_host_event_shape",
            summary: "before_prompt_build payload was partial or malformed",
            action: "Inspect the host before_prompt_build event shape; OpenClawBrain fail-opened safely."
        };
    }
    return diagnostic;
}
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
//# sourceMappingURL=runtime-guard.js.map
