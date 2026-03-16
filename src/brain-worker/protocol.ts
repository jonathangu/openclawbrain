import type { CompletionContentBlock } from "../types.js";

export type WorkerReadyMessage = {
  type: "ready";
  pid: number;
  at: number;
};

export type WorkerHeartbeatMessage = {
  type: "heartbeat";
  pid: number;
  at: number;
  status?: string;
};

export type WorkerReloadGraphAckMessage = {
  type: "reload-graph-ack";
  pid: number;
  at: number;
  nodeCount: number;
  edgeCount: number;
};

export type WorkerTickResultMessage = {
  type: "tick-result";
  pid: number;
  at: number;
  ok: boolean;
  error?: string;
};

export type WorkerTeacherCompleteRequestMessage = {
  type: "teacher-complete";
  requestId: string;
  provider?: string;
  model: string;
  messages: Array<{ role: string; content: unknown }>;
  system?: string;
  maxTokens: number;
  temperature?: number;
};

export type WorkerPackPromotedMessage = {
  type: "pack-promoted";
  pid: number;
  version: number | null;
};

export type WorkerFatalErrorMessage = {
  type: "fatal-error";
  pid: number;
  error: string;
};

export type ChildToParentMessage =
  | WorkerReadyMessage
  | WorkerHeartbeatMessage
  | WorkerReloadGraphAckMessage
  | WorkerTickResultMessage
  | WorkerTeacherCompleteRequestMessage
  | WorkerPackPromotedMessage
  | WorkerFatalErrorMessage;

export type ParentTeacherCompleteResultMessage =
  | { type: "teacher-complete-result"; requestId: string; ok: true; content: CompletionContentBlock[] }
  | { type: "teacher-complete-result"; requestId: string; ok: false; error: string };

export type ParentReloadGraphMessage = {
  type: "reload-graph";
};

export type ParentShutdownMessage = {
  type: "shutdown";
};

export type ParentToChildMessage =
  | ParentTeacherCompleteResultMessage
  | ParentReloadGraphMessage
  | ParentShutdownMessage;
