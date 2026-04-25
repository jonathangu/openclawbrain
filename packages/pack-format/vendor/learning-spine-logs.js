import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { checksumJsonPayload } from "@openclawbrain/contracts";
export const LEARNING_SPINE_LOG_LAYOUT = {
    dir: "logs/learning-spine",
    serveTimeRouteDecisions: "logs/learning-spine/serve-time-route-decisions.jsonl",
    supervisionLabelBindings: "logs/learning-spine/supervision-label-bindings.jsonl",
    pgRouteUpdates: "logs/learning-spine/pg-route-updates.jsonl",
    promotionActivationEvents: "logs/learning-spine/promotion-activation-events.jsonl",
    trajectories: "logs/learning-spine/trajectories.jsonl"
};
const LEARNING_SPINE_LOG_STREAM_PATHS = {
    serveTimeRouteDecisions: LEARNING_SPINE_LOG_LAYOUT.serveTimeRouteDecisions,
    supervisionLabelBindings: LEARNING_SPINE_LOG_LAYOUT.supervisionLabelBindings,
    pgRouteUpdates: LEARNING_SPINE_LOG_LAYOUT.pgRouteUpdates,
    promotionActivationEvents: LEARNING_SPINE_LOG_LAYOUT.promotionActivationEvents,
    trajectories: LEARNING_SPINE_LOG_LAYOUT.trajectories
};
export function resolveLearningSpineLogPath(rootDir, stream) {
    return path.join(path.resolve(rootDir), LEARNING_SPINE_LOG_STREAM_PATHS[stream]);
}
export function buildLearningSpineLogId(prefix, payload) {
    return `${prefix}-${checksumJsonPayload(payload)}`;
}
export function appendLearningSpineLogEntry(rootDir, stream, entry) {
    const logPath = resolveLearningSpineLogPath(rootDir, stream);
    mkdirSync(path.dirname(logPath), { recursive: true });
    appendFileSync(logPath, `${JSON.stringify(entry)}\n`, "utf8");
    return logPath;
}
export function readLearningSpineLogEntries(rootDir, stream) {
    const logPath = resolveLearningSpineLogPath(rootDir, stream);
    if (!existsSync(logPath)) {
        return [];
    }
    return readFileSync(logPath, "utf8")
        .split(/\r?\n/u)
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line));
}
