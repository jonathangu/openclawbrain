import type { MemoryNode } from './memory-types.js';
import { MemoryStore } from './memory-store.js';
import { safeString } from './redact.js';

export function buildMemoryPromptSupplement() {
  return () => [
    'OpenClawBrain local memory supplement is available for scoped corrections, preferences, workflows, and context when prior decisions matter.',
  ];
}

export function buildMemoryCorpusSupplement(config: any) {
  return {
    search: async ({ query, maxResults }: { query: string; maxResults?: number; agentSessionKey?: string }) => {
      const agentId = defaultAgentId(config);
      const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
      try {
        return store.searchMemories(query, agentId, { limit: maxResults ?? 10 }).map((memory) => searchResultFromMemory(memory));
      } finally {
        store.close();
      }
    },
    get: async ({ lookup }: { lookup: string; fromLine?: number; lineCount?: number; agentSessionKey?: string }) => {
      const agentId = defaultAgentId(config);
      const memoryId = extractMemoryId(lookup);
      const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
      try {
        const memory = store.getMemory(memoryId);
        if (!memory) return null;
        return {
          corpus: 'openclawbrain',
          path: memoryPath(memory),
          title: `${memory.type}: ${memory.normalizedKey}`,
          kind: memory.type,
          content: renderMemory(memory),
          fromLine: 1,
          lineCount: renderMemory(memory).split(/\n/).length,
          id: memory.id,
          provenanceLabel: 'OpenClawBrain local memory graph',
          updatedAt: memory.updatedAt,
        };
      } finally {
        store.close();
      }
    },
  };
}

export function searchPayload(config: any, agentId: string, query: string, limit = 10) {
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  try {
    const memories = store.searchMemories(query, agentId, { limit });
    return {
      ok: true,
      agentId,
      query,
      limit,
      results: memories.map((memory) => ({
        id: memory.id,
        type: memory.type,
        normalizedKey: memory.normalizedKey,
        score: Number((memory.importance * memory.confidence).toFixed(3)),
        content: memory.content,
        tags: memory.tags,
        updatedAt: memory.updatedAt,
      })),
    };
  } finally {
    store.close();
  }
}

export function graphPayload(config: any, agentId: string, limit = 20) {
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  try {
    const nodes = store.listMemories(agentId, { limit });
    const nodeIds = new Set(nodes.map((node) => node.id));
    const edges = dedupeEdges(nodes.flatMap((node) => store.getEdges(node.id))).filter((edge) => nodeIds.has(edge.fromId) || nodeIds.has(edge.toId));
    return {
      ok: true,
      agentId,
      nodes: nodes.map((node) => ({
        id: node.id,
        type: node.type,
        normalizedKey: node.normalizedKey,
        content: node.content,
        importance: node.importance,
        confidence: node.confidence,
        supersededBy: node.supersededBy || null,
      })),
      edges,
    };
  } finally {
    store.close();
  }
}

export function learnPayload(config: any, agentId: string, limit = 20) {
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  try {
    return {
      ok: true,
      agentId,
      activePolicySnapshot: store.getActivePolicySnapshot(agentId),
      examples: store.getRouteExamples(agentId, limit),
      policySnapshots: store.listPolicySnapshots(agentId, limit),
    };
  } finally {
    store.close();
  }
}

function defaultAgentId(config: any) {
  return safeString(config?.scopes?.agents?.[0] ?? 'main') || 'main';
}

function extractMemoryId(lookup: string) {
  const value = safeString(lookup);
  const match = value.match(/([0-9a-f]{8}-[0-9a-f-]{27,})/i);
  return match ? match[1] : value.replace(/^memory\//, '').replace(/\.md$/, '');
}

function searchResultFromMemory(memory: MemoryNode) {
  return {
    corpus: 'openclawbrain',
    path: memoryPath(memory),
    title: `${memory.type}: ${memory.normalizedKey}`,
    kind: memory.type,
    score: Number((memory.importance * memory.confidence).toFixed(3)),
    snippet: memory.content,
    id: memory.id,
    startLine: 1,
    endLine: renderMemory(memory).split(/\n/).length,
    citation: `${memoryPath(memory)}#L1-L${renderMemory(memory).split(/\n/).length}`,
    source: 'openclawbrain',
    provenanceLabel: 'OpenClawBrain local memory graph',
    updatedAt: memory.updatedAt,
  };
}

function renderMemory(memory: MemoryNode) {
  return [
    `# ${memory.type}: ${memory.normalizedKey}`,
    '',
    memory.content,
    '',
    `- scope: ${memory.scopeKind}${memory.scopeKey ? `:${memory.scopeKey}` : ''}`,
    `- tags: ${memory.tags.join(', ')}`,
    `- confidence: ${memory.confidence}`,
    `- importance: ${memory.importance}`,
  ].join('\n');
}

function memoryPath(memory: MemoryNode) {
  return `memory/${memory.id}.md`;
}

function dedupeEdges(edges: any[]) {
  const seen = new Set<string>();
  return edges.filter((edge) => {
    const key = `${edge.id}:${edge.fromId}:${edge.toId}:${edge.relation}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
