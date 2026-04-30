import { chmod, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { activationRootForAgent } from './config.js';
import { sanitizeForProof } from './redact.js';
export async function appendProofEvent(event, options) {
    const root = await proofRoot(options, event.agentId || options.agentId || 'main');
    const file = path.join(root, 'proof-events.jsonl');
    const sanitized = sanitizeProofEvent(event);
    const previous = await readJsonl(file);
    previous.push(sanitized);
    const configuredRetention = Number(options.proofRetentionEvents);
    const retention = Number.isFinite(configuredRetention) ? Math.min(50000, Math.max(50, Math.trunc(configuredRetention))) : 1000;
    const retained = previous.slice(-retention);
    await atomicWrite(file, retained.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
    return sanitized;
}
export async function readProofEvents(options = {}) {
    const root = await proofRoot(options, options.agentId || 'main');
    const file = path.join(root, 'proof-events.jsonl');
    const events = await readJsonl(file);
    const limit = Math.min(100, Math.max(1, Number(options.limit || 20)));
    return events.slice(-limit).map(sanitizeProofEvent);
}
export async function writeStatus(status, options) {
    const root = await proofRoot(options, status.agentId || options.agentId || 'main');
    const file = path.join(root, 'status.json');
    await atomicWrite(file, `${JSON.stringify(sanitizeForProof(status), null, 2)}\n`);
    return status;
}
export async function readStatus(options = {}) {
    const root = await proofRoot(options, options.agentId || 'main');
    const file = path.join(root, 'status.json');
    try {
        return JSON.parse(await readFile(file, 'utf8'));
    }
    catch (error) {
        if (error?.code === 'ENOENT')
            return null;
        throw error;
    }
}
export function sanitizeProofEvent(event = {}) {
    return sanitizeForProof({
        ...event,
        rawTranscriptStored: false,
        rawUserTextStored: false,
        redactionApplied: true,
        hashesOnlyForUserText: true
    });
}
async function proofRoot(options = {}, agentId = 'main') {
    const root = options.activationRoot
        ? activationRootForAgent({ activationRoot: options.activationRoot }, agentId)
        : activationRootForAgent(options.config || {}, agentId);
    await mkdir(root, { recursive: true, mode: 0o700 });
    try {
        await chmod(root, 0o700);
    }
    catch {
    }
    return root;
}
async function readJsonl(file) {
    try {
        const text = await readFile(file, 'utf8');
        return text.split('\n').filter(Boolean).map((line) => JSON.parse(line)).map(sanitizeProofEvent);
    }
    catch (error) {
        if (error?.code === 'ENOENT')
            return [];
        throw error;
    }
}
async function atomicWrite(file, content) {
    const temporary = `${file}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(temporary, content, { mode: 0o600 });
    await rename(temporary, file);
}
