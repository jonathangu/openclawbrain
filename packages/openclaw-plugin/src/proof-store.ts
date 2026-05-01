import { chmod, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { activationRootForAgent } from './config.js';
import { MemoryStore } from './memory-store.js';
import { sanitizeForProof } from './redact.js';

export async function appendProofEvent(event: any, options: any) {
  const root = await proofRoot(options, event.agentId || options.agentId || 'main');
  const sanitized = sanitizeProofEvent(event);

  // Canonical v0.2 path: SQLite
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: event.agentId || options.agentId || 'main' });
    const retention = Number.isFinite(Number(options.proofRetentionEvents))
      ? Math.min(50000, Math.max(50, Math.trunc(Number(options.proofRetentionEvents))))
      : 1000;
    store.insertProofEvent({
      agentId: sanitized.agentId || event.agentId || options.agentId || 'main',
      kind: sanitized.kind || sanitized.decisionKind || 'proof_event',
      sourceHook: sanitized.hookPhase,
      turnId: sanitized.turnId,
      sessionId: sanitized.sessionId,
      runId: sanitized.runId,
      memoryId: sanitized.memoryId,
      injectionId: sanitized.injectionId,
      routeDecisionId: sanitized.routeDecisionId,
      distillationRunId: sanitized.distillationRunId,
      rawTranscriptStored: false,
      payload: sanitized,
    });
    store.pruneProofEvents(sanitized.agentId || event.agentId || options.agentId || 'main', retention);
    store.close();
  } catch {
    // keep legacy path as compatibility fallback
  }

  // Legacy v0.1 compatibility mirror: JSONL file
  const file = path.join(root, 'proof-events.jsonl');
  const previous = await readJsonl(file);
  previous.push(sanitized);
  const configuredRetention = Number(options.proofRetentionEvents);
  const retention = Number.isFinite(configuredRetention) ? Math.min(50000, Math.max(50, Math.trunc(configuredRetention))) : 1000;
  const retained = previous.slice(-retention);
  await atomicWrite(file, retained.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  return sanitized;
}

export async function readProofEvents(options: any = {}) {
  const agentId = options.agentId || 'main';
  const root = await proofRoot(options, agentId);
  const limit = Math.min(100, Math.max(1, Number(options.limit || 20)));

  try {
    const store = new MemoryStore({ activationRoot: root, agentId });
    const events = store.getProofEvents(agentId, limit).map((event) => sanitizeProofEvent({
      ...(event.payload ?? {}),
      kind: event.kind,
      agentId: event.agentId,
      createdAt: event.createdAt,
      sourceHook: event.sourceHook,
      turnId: event.turnId,
      sessionId: event.sessionId,
      runId: event.runId,
      memoryId: event.memoryId,
      injectionId: event.injectionId,
      routeDecisionId: event.routeDecisionId,
      distillationRunId: event.distillationRunId,
      rawTranscriptStored: event.rawTranscriptStored,
    })).reverse();
    store.close();
    if (events.length > 0) return events;
  } catch {
  }

  const file = path.join(root, 'proof-events.jsonl');
  const events: any[] = await readJsonl(file);
  return events.slice(-limit).map(sanitizeProofEvent);
}

export async function writeStatus(status: any, options: any) {
  const agentId = status.agentId || options.agentId || 'main';
  const root = await proofRoot(options, agentId);
  const sanitized = sanitizeForProof(status);

  try {
    const store = new MemoryStore({ activationRoot: root, agentId });
    store.writeStatusSnapshot(agentId, sanitized);
    store.close();
  } catch {
  }

  const file = path.join(root, 'status.json');
  await atomicWrite(file, `${JSON.stringify(sanitized, null, 2)}\n`);
  return status;
}

export async function readStatus(options: any = {}) {
  const agentId = options.agentId || 'main';
  const root = await proofRoot(options, agentId);

  try {
    const store = new MemoryStore({ activationRoot: root, agentId });
    const status = store.readStatusSnapshot(agentId);
    store.close();
    if (status) return status;
  } catch {
  }

  const file = path.join(root, 'status.json');
  try {
    return JSON.parse(await readFile(file, 'utf8'));
  } catch (error: any) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

export function sanitizeProofEvent(event: any = {}) {
  return sanitizeForProof({
    ...event,
    rawTranscriptStored: false,
    rawUserTextStored: false,
    redactionApplied: true,
    hashesOnlyForUserText: true,
  });
}

async function proofRoot(options: any = {}, agentId = 'main') {
  const root = options.activationRoot
    ? activationRootForAgent({ activationRoot: options.activationRoot }, agentId)
    : activationRootForAgent(options.config || {}, agentId);
  await mkdir(root, { recursive: true, mode: 0o700 });
  try {
    await chmod(root, 0o700);
  } catch {
  }
  return root;
}

async function readJsonl(file: string) {
  try {
    const text = await readFile(file, 'utf8');
    return text.split('\n').filter(Boolean).map((line) => JSON.parse(line)).map(sanitizeProofEvent);
  } catch (error: any) {
    if (error?.code === 'ENOENT') return [];
    throw error;
  }
}

async function atomicWrite(file: string, content: string) {
  const temporary = `${file}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary, content, { mode: 0o600 });
  await rename(temporary, file);
}
