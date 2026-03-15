/**
 * Source discovery, chunking, and seed graph building for brain init.
 *
 * Brain init is a first-class product moment:
 * 1. Discover sources (markdown, code, sessions, LCM summaries, OpenClaw memory)
 * 2. Structure-aware chunking (heading boundaries, code blocks intact)
 * 3. Create nodes with embeddings
 * 4. Create cold-start edges (sibling + semantic)
 * 5. Build pack v0
 */

import { randomUUID } from "node:crypto";
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join, extname, relative } from "node:path";
import type { BrainNode, BrainEdge, NodeKind } from "../brain-core/types.js";
import { cosineSimilarity } from "../brain-core/graph.js";

// ─── Source Discovery ───

export interface Source {
  uri: string;
  content: string;
  type: "markdown" | "code" | "config" | "memory";
}

export function discoverSources(workspaceRoot: string): Source[] {
  const sources: Source[] = [];
  const maxFileSize = 100_000; // 100KB

  function walk(dir: string, depth = 0): void {
    if (depth > 4) return; // Don't recurse too deep
    let entries: string[];
    try {
      entries = readdirSync(dir);
    } catch {
      return;
    }

    for (const entry of entries) {
      if (entry.startsWith(".") || entry === "node_modules" || entry === "dist") continue;
      const fullPath = join(dir, entry);
      let stat;
      try {
        stat = statSync(fullPath);
      } catch {
        continue;
      }

      if (stat.isDirectory()) {
        walk(fullPath, depth + 1);
        continue;
      }

      if (stat.size > maxFileSize) continue;

      const ext = extname(entry).toLowerCase();
      const relPath = relative(workspaceRoot, fullPath);

      if (ext === ".md") {
        sources.push({ uri: relPath, content: readFileSync(fullPath, "utf-8"), type: "markdown" });
      } else if ([".ts", ".js", ".py", ".go", ".rs"].includes(ext)) {
        sources.push({ uri: relPath, content: readFileSync(fullPath, "utf-8"), type: "code" });
      } else if (["package.json", "tsconfig.json", "Dockerfile"].includes(entry) || ext === ".toml" || ext === ".yaml" || ext === ".yml") {
        sources.push({ uri: relPath, content: readFileSync(fullPath, "utf-8"), type: "config" });
      }
    }
  }

  walk(workspaceRoot);

  // Also check for OpenClaw memory files
  const memoryDirs = [
    join(workspaceRoot, ".claude", "memory"),
    join(workspaceRoot, "memory"),
  ];
  for (const memDir of memoryDirs) {
    if (existsSync(memDir)) {
      try {
        for (const entry of readdirSync(memDir)) {
          if (entry.endsWith(".md")) {
            const fullPath = join(memDir, entry);
            sources.push({ uri: relative(workspaceRoot, fullPath), content: readFileSync(fullPath, "utf-8"), type: "memory" });
          }
        }
      } catch { /* ignore */ }
    }
  }

  // Check for MEMORY.md
  const memoryMd = join(workspaceRoot, "MEMORY.md");
  if (existsSync(memoryMd)) {
    sources.push({ uri: "MEMORY.md", content: readFileSync(memoryMd, "utf-8"), type: "memory" });
  }

  return sources;
}

// ─── Chunking ───

export interface Chunk {
  content: string;
  sourceUri: string;
  kind: NodeKind;
  ordinal: number;
}

/**
 * Structure-aware chunking.
 * - Markdown: split on ## headings, keep code blocks intact
 * - Code: split on function/class boundaries
 * - Config: one chunk per file
 */
export function chunkSources(sources: Source[]): Chunk[] {
  const chunks: Chunk[] = [];

  for (const source of sources) {
    if (source.type === "markdown" || source.type === "memory") {
      chunks.push(...chunkMarkdown(source.content, source.uri));
    } else if (source.type === "code") {
      chunks.push(...chunkCode(source.content, source.uri));
    } else {
      // Config: one chunk per file
      if (source.content.trim()) {
        chunks.push({ content: source.content.trim(), sourceUri: source.uri, kind: "chunk", ordinal: 0 });
      }
    }
  }

  return chunks;
}

function chunkMarkdown(content: string, uri: string): Chunk[] {
  const chunks: Chunk[] = [];
  const lines = content.split("\n");
  let current: string[] = [];
  let ordinal = 0;

  function flush(): void {
    const text = current.join("\n").trim();
    if (text.length > 20) {
      // Detect workflows (numbered steps)
      const kind: NodeKind = /^\s*\d+\.\s/m.test(text) ? "workflow" : "chunk";
      chunks.push({ content: text, sourceUri: uri, kind, ordinal: ordinal++ });
    }
    current = [];
  }

  for (const line of lines) {
    if (/^#{1,3}\s/.test(line) && current.length > 0) {
      flush();
    }
    current.push(line);
  }
  flush();

  return chunks;
}

function chunkCode(content: string, uri: string): Chunk[] {
  const chunks: Chunk[] = [];
  const lines = content.split("\n");
  let current: string[] = [];
  let ordinal = 0;

  function flush(): void {
    const text = current.join("\n").trim();
    if (text.length > 30) {
      chunks.push({ content: text, sourceUri: uri, kind: "chunk", ordinal: ordinal++ });
    }
    current = [];
  }

  for (const line of lines) {
    if (/^(export\s+)?(function|class|const\s+\w+\s*=\s*(\(|async\s*\())/.test(line) && current.length > 3) {
      flush();
    }
    current.push(line);
  }
  flush();

  return chunks;
}

// ─── Node Creation ───

export function createNodesFromChunks(chunks: Chunk[]): BrainNode[] {
  const now = Date.now();
  return chunks.map((chunk) => ({
    id: `bn_${randomUUID().slice(0, 12)}`,
    kind: chunk.kind,
    content: chunk.content,
    embedding: null, // Computed separately
    sourceUri: chunk.sourceUri,
    trust: "scanner" as const,
    tags: [],
    tokenCount: Math.ceil(chunk.content.length / 4),
    metadata: { ordinal: chunk.ordinal },
    createdAt: now,
    updatedAt: now,
  }));
}

// ─── Edge Creation ───

/**
 * Sibling edges: same-document adjacent chunks.
 */
export function createSiblingEdges(nodes: BrainNode[]): BrainEdge[] {
  const edges: BrainEdge[] = [];
  const now = Date.now();

  // Group by sourceUri
  const bySource = new Map<string, BrainNode[]>();
  for (const node of nodes) {
    if (!node.sourceUri) continue;
    const group = bySource.get(node.sourceUri) ?? [];
    group.push(node);
    bySource.set(node.sourceUri, group);
  }

  for (const group of bySource.values()) {
    // Sort by ordinal
    group.sort((a, b) => ((a.metadata as any).ordinal ?? 0) - ((b.metadata as any).ordinal ?? 0));
    for (let i = 0; i < group.length - 1; i++) {
      edges.push({
        source: group[i].id,
        target: group[i + 1].id,
        kind: "sibling",
        weight: 0.8,
        prior: 1.0,
        metadata: {},
        decayedAt: now,
        createdAt: now,
      });
      // Bidirectional
      edges.push({
        source: group[i + 1].id,
        target: group[i].id,
        kind: "sibling",
        weight: 0.8,
        prior: 1.0,
        metadata: {},
        decayedAt: now,
        createdAt: now,
      });
    }
  }

  return edges;
}

/**
 * Semantic edges: embedding cosine similarity above threshold.
 * Top-3 most similar for each node.
 */
export function createSemanticEdges(nodes: BrainNode[], threshold: number): BrainEdge[] {
  const edges: BrainEdge[] = [];
  const now = Date.now();
  const withEmbeddings = nodes.filter((n) => n.embedding);

  for (const node of withEmbeddings) {
    const scored: Array<{ targetId: string; score: number }> = [];
    for (const other of withEmbeddings) {
      if (other.id === node.id) continue;
      const score = cosineSimilarity(node.embedding!, other.embedding!);
      if (score >= threshold) {
        scored.push({ targetId: other.id, score });
      }
    }
    scored.sort((a, b) => b.score - a.score);

    for (const { targetId, score } of scored.slice(0, 3)) {
      edges.push({
        source: node.id,
        target: targetId,
        kind: "semantic",
        weight: score,
        prior: score,
        metadata: {},
        decayedAt: now,
        createdAt: now,
      });
    }
  }

  return edges;
}

// ─── Full Init Pipeline ───

export interface InitResult {
  nodeCount: number;
  edgeCount: number;
  nodesByKind: Record<string, number>;
  summary: string;
}

export async function initBrain(params: {
  workspaceRoot: string;
  embedFn: (text: string) => Promise<Float32Array>;
  semanticThreshold: number;
  log: { info: (msg: string) => void; warn: (msg: string) => void };
}): Promise<{ nodes: BrainNode[]; edges: BrainEdge[]; summary: string }> {
  const { workspaceRoot, embedFn, semanticThreshold, log } = params;

  // Step 1: Discover
  log.info("[brain] Discovering sources...");
  const sources = discoverSources(workspaceRoot);
  log.info(`[brain] Found ${sources.length} sources`);

  // Step 2: Chunk
  const chunks = chunkSources(sources);
  log.info(`[brain] Created ${chunks.length} chunks`);

  // Step 3: Create nodes
  const nodes = createNodesFromChunks(chunks);

  // Step 4: Compute embeddings
  log.info("[brain] Computing embeddings...");
  for (const node of nodes) {
    try {
      node.embedding = await embedFn(node.content.slice(0, 512));
    } catch (err) {
      log.warn(`[brain] Embedding failed for ${node.id}: ${(err as Error).message}`);
    }
  }

  const embeddedCount = nodes.filter((n) => n.embedding).length;
  log.info(`[brain] Embedded ${embeddedCount}/${nodes.length} nodes`);

  // Step 5: Create edges
  const siblingEdges = createSiblingEdges(nodes);
  const semanticEdges = createSemanticEdges(nodes, semanticThreshold);
  const allEdges = [...siblingEdges, ...semanticEdges];
  log.info(`[brain] Created ${allEdges.length} edges (${siblingEdges.length} sibling, ${semanticEdges.length} semantic)`);

  // Step 6: Summary
  const kindCounts: Record<string, number> = {};
  for (const node of nodes) {
    kindCounts[node.kind] = (kindCounts[node.kind] ?? 0) + 1;
  }

  const summary = `Brain initialized: ${nodes.length} nodes (${Object.entries(kindCounts).map(([k, v]) => `${v} ${k}`).join(", ")}), ${allEdges.length} edges`;
  log.info(`[brain] ${summary}`);

  return { nodes, edges: allEdges, summary };
}
