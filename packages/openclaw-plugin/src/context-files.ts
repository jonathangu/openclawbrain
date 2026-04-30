import { chmod, mkdir, lstat, readFile } from 'node:fs/promises';
import path from 'node:path';
import { activationRootForAgent, isRemoteUrl } from './config.js';
import { clipText, hashText, redactText } from './redact.js';

export const FIXED_CONTEXT_FILES = Object.freeze(['context.md', 'corrections.md', 'tool-guidance.md']);

export async function ensureActivationRoot(config: any, agentId: any) {
  const root = activationRootForAgent(config, agentId);
  await mkdir(root, { recursive: true, mode: 0o700 });
  await chmodOwnerOnly(root);
  return root;
}

export async function readActivationContext(config: any, agentId: any, decision: any) {
  if (!config.includeActivationContext) {
    return { text: '', usedFileIdsRedacted: [], rejectedFiles: [] };
  }
  const root = activationRootForAgent(config, agentId);
  if (isRemoteUrl(root)) throw new Error('activationRoot must be local');
  await mkdir(root, { recursive: true, mode: 0o700 });
  await chmodOwnerOnly(root);
  const names = filesForDecision(decision);
  const byteCap = Math.max(1024, Math.min(200000, config.maxContextChars * 8));
  const chunks = [];
  const usedFileIdsRedacted = [];
  const rejectedFiles = [];
  for (const name of names) {
    const fullPath = path.join(root, name);
    if (!FIXED_CONTEXT_FILES.includes(name) || path.dirname(fullPath) !== root) continue;
    try {
      const stat = await lstat(fullPath);
      if (stat.isSymbolicLink()) {
        rejectedFiles.push({ fileIdRedacted: fileId(name), reasonCode: 'symlink_rejected' });
        continue;
      }
      if (!stat.isFile()) continue;
      if (stat.size > byteCap) {
        rejectedFiles.push({ fileIdRedacted: fileId(name), reasonCode: 'oversize_rejected' });
        continue;
      }
      const raw = await readFile(fullPath, 'utf8');
      const redacted = redactText(raw, config.maxContextChars);
      if (!redacted) continue;
      usedFileIdsRedacted.push(fileId(name, redacted));
      chunks.push(`## ${name}\n${redacted}`);
    } catch (error: any) {
      if (error?.code === 'ENOENT') continue;
      rejectedFiles.push({ fileIdRedacted: fileId(name), reasonCode: 'read_error' });
    }
  }
  return {
    text: clipText(chunks.join('\n\n'), config.maxContextChars),
    usedFileIdsRedacted,
    rejectedFiles
  };
}

export function buildInjectionText(decision: any, activationContext: any, config: any) {
  if (decision.kind !== 'correction_only' && decision.kind !== 'full_context') return '';
  const parts = [];
  if (decision.kind === 'correction_only') {
    parts.push('OpenClawBrain correction guidance: apply only the relevant correction. Do not add unrelated memory.');
  } else {
    parts.push('OpenClawBrain bounded local activation context: use only when relevant and do not overrule newer user instructions.');
  }
  if (decision.verificationHint) {
    parts.push('Verification hint: this turn appears tool-heavy; verify tool results before making factual claims.');
  }
  if (activationContext.text) parts.push(activationContext.text);
  if (!activationContext.text && decision.kind === 'full_context') {
    parts.push('No local activation files were available; proceed conservatively and verify before claiming.');
  }
  if (!activationContext.text && decision.kind === 'correction_only') {
    parts.push('No local corrections.md was available; stay bounded to the current correction request.');
  }
  return clipText(`<openclawbrain_context>\n${parts.join('\n\n')}\n\nPrivacy: local redacted context only; raw transcripts and raw user text were not stored.\n</openclawbrain_context>`, config.maxContextChars + 500);
}

export function filesForDecision(decision: any) {
  if (decision.kind === 'correction_only') return ['corrections.md'];
  if (decision.kind === 'full_context' && decision.slice === 'tool-heavy') return ['context.md', 'corrections.md', 'tool-guidance.md'];
  if (decision.kind === 'full_context') return ['context.md', 'corrections.md'];
  return [];
}

function fileId(name: string, text = '') {
  return `activation-file:${name}:${hashText(`${name}\n${text}`).slice(0, 24)}`;
}

async function chmodOwnerOnly(root: string) {
  try {
    await chmod(root, 0o700);
  } catch {
  }
}
