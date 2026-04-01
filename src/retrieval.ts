import type {
  ConversationStore,
  MessageRecord,
  MessageSearchResult,
} from "./store/conversation-store.js";
import type {
  SummaryStore,
  SummaryRecord,
  SummarySearchResult,
  LargeFileRecord,
  MarbleRecord,
  MarbleSearchResult,
  MarbleSourceRecord,
  SummaryLineageRecord,
  BranchSnapshotRecord,
} from "./store/summary-store.js";

// ── Public interfaces ────────────────────────────────────────────────────────

export interface DescribeResult {
  id: string;
  type: "summary" | "file" | "marble";
  /** Summary-specific fields */
  summary?: {
    conversationId: number;
    kind: "leaf" | "condensed";
    content: string;
    depth: number;
    tokenCount: number;
    descendantCount: number;
    descendantTokenCount: number;
    sourceMessageTokenCount: number;
    fileIds: string[];
    parentIds: string[];
    childIds: string[];
    messageIds: number[];
    earliestAt: Date | null;
    latestAt: Date | null;
    subtree: Array<{
      summaryId: string;
      parentSummaryId: string | null;
      depthFromRoot: number;
      kind: "leaf" | "condensed";
      depth: number;
      tokenCount: number;
      descendantCount: number;
      descendantTokenCount: number;
      sourceMessageTokenCount: number;
      earliestAt: Date | null;
      latestAt: Date | null;
      childCount: number;
      path: string;
    }>;
    lineage?: SummaryLineageRecord | null;
    snapshot?: BranchSnapshotRecord | null;
    createdAt: Date;
  };
  /** File-specific fields */
  file?: {
    conversationId: number;
    fileName: string | null;
    mimeType: string | null;
    byteSize: number | null;
    storageUri: string;
    explorationSummary: string | null;
    createdAt: Date;
  };
  /** Marble-specific fields */
  marble?: MarbleRecord & {
    sourceRefs: string[];
    sources: MarbleSourceRecord[];
  };
}

export interface GrepInput {
  query: string;
  mode: "regex" | "full_text";
  scope: "messages" | "summaries" | "marbles" | "both";
  conversationId?: number;
  since?: Date;
  before?: Date;
  limit?: number;
}

export interface GrepResult {
  messages: MessageSearchResult[];
  summaries: SummarySearchResult[];
  marbles: MarbleSearchResult[];
  totalMatches: number;
}

export interface ExpandInput {
  summaryId: string;
  /** Max traversal depth (default 1) */
  depth?: number;
  /** Include raw source messages at leaf level */
  includeMessages?: boolean;
  /** Max tokens to return before truncating */
  tokenCap?: number;
}

export interface ExpandResult {
  /** Child summaries found */
  children: Array<{
    summaryId: string;
    kind: "leaf" | "condensed";
    content: string;
    tokenCount: number;
  }>;
  /** Source messages (only if includeMessages=true and hitting leaf summaries) */
  messages: Array<{
    messageId: number;
    role: string;
    content: string;
    tokenCount: number;
  }>;
  /** Total estimated tokens in result */
  estimatedTokens: number;
  /** Whether result was truncated due to tokenCap */
  truncated: boolean;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Rough token estimate: ~4 chars per token. */
function estimateTokens(content: string): number {
  return Math.ceil(content.length / 4);
}

function formatMarbleSourceRef(source: MarbleSourceRecord): string {
  const subId = typeof source.sourceSubId === "string" && source.sourceSubId.length > 0
    ? `#${source.sourceSubId}`
    : "";
  return `${source.sourceKind}:${source.sourceId}${subId}`;
}

// ── RetrievalEngine ──────────────────────────────────────────────────────────

export class RetrievalEngine {
  constructor(
    private conversationStore: ConversationStore,
    private summaryStore: SummaryStore,
  ) {}

  // ── describe ─────────────────────────────────────────────────────────────

  /**
   * Describe an LCM item by ID.
   *
   * - IDs starting with "sum_" are looked up as summaries (with lineage).
   * - IDs starting with "file_" are looked up as large files.
   * - Returns null if the item is not found.
   */
  async describe(id: string): Promise<DescribeResult | null> {
    if (id.startsWith("sum_")) {
      return this.describeSummary(id);
    }
    if (id.startsWith("mar_")) {
      return this.describeMarble(id);
    }
    if (id.startsWith("file_")) {
      return this.describeFile(id);
    }
    return null;
  }

  private async describeSummary(id: string): Promise<DescribeResult | null> {
    const summary = await this.summaryStore.getSummary(id);
    if (!summary) {
      return null;
    }

    // Fetch lineage in parallel
    const [parents, children, messageIds, subtree, lineage] = await Promise.all([
      this.summaryStore.getSummaryParents(id),
      this.summaryStore.getSummaryChildren(id),
      this.summaryStore.getSummaryMessages(id),
      this.summaryStore.getSummarySubtree(id),
      this.maybeGetSummaryLineage(id),
    ]);

    const snapshot = lineage?.snapshotId ? await this.maybeGetBranchSnapshot(lineage.snapshotId) : null;

    return {
      id,
      type: "summary",
      summary: {
        conversationId: summary.conversationId,
        kind: summary.kind,
        content: summary.content,
        depth: summary.depth,
        tokenCount: summary.tokenCount,
        descendantCount: summary.descendantCount,
        descendantTokenCount: summary.descendantTokenCount,
        sourceMessageTokenCount: summary.sourceMessageTokenCount,
        fileIds: summary.fileIds,
        parentIds: parents.map((p) => p.summaryId),
        childIds: children.map((c) => c.summaryId),
        messageIds,
        earliestAt: summary.earliestAt,
        latestAt: summary.latestAt,
        subtree: subtree.map((node) => ({
          summaryId: node.summaryId,
          parentSummaryId: node.parentSummaryId,
          depthFromRoot: node.depthFromRoot,
          kind: node.kind,
          depth: node.depth,
          tokenCount: node.tokenCount,
          descendantCount: node.descendantCount,
          descendantTokenCount: node.descendantTokenCount,
          sourceMessageTokenCount: node.sourceMessageTokenCount,
          earliestAt: node.earliestAt,
          latestAt: node.latestAt,
          childCount: node.childCount,
          path: node.path,
        })),
        lineage,
        snapshot,
        createdAt: summary.createdAt,
      },
    };
  }

  private async maybeGetSummaryLineage(summaryId: string): Promise<SummaryLineageRecord | null> {
    const store = this.summaryStore as unknown as {
      getSummaryLineage?: (summaryId: string) => Promise<SummaryLineageRecord | null>;
    };
    if (typeof store.getSummaryLineage !== "function") {
      return null;
    }
    return store.getSummaryLineage(summaryId);
  }

  private async maybeGetBranchSnapshot(snapshotId: string): Promise<BranchSnapshotRecord | null> {
    const store = this.summaryStore as unknown as {
      getBranchSnapshot?: (snapshotId: string) => Promise<BranchSnapshotRecord | null>;
    };
    if (typeof store.getBranchSnapshot !== "function") {
      return null;
    }
    return store.getBranchSnapshot(snapshotId);
  }

  private async describeFile(id: string): Promise<DescribeResult | null> {
    const file = await this.summaryStore.getLargeFile(id);
    if (!file) {
      return null;
    }

    return {
      id,
      type: "file",
      file: {
        conversationId: file.conversationId,
        fileName: file.fileName,
        mimeType: file.mimeType,
        byteSize: file.byteSize,
        storageUri: file.storageUri,
        explorationSummary: file.explorationSummary,
        createdAt: file.createdAt,
      },
    };
  }

  private async describeMarble(id: string): Promise<DescribeResult | null> {
    const marble = await this.summaryStore.getMarble(id);
    if (!marble) {
      return null;
    }

    const sources = await this.summaryStore.getMarbleSources(id);
    return {
      id,
      type: "marble",
      marble: {
        ...marble,
        sourceRefs: sources.map(formatMarbleSourceRef),
        sources,
      },
    };
  }

  // ── grep ─────────────────────────────────────────────────────────────────

  /**
   * Search compacted history using regex or full-text search.
   *
   * Depending on `scope`, searches messages, summaries, or both (in parallel).
   */
  async grep(input: GrepInput): Promise<GrepResult> {
    const { query, mode, scope, conversationId, since, before, limit } = input;

    const searchInput = { query, mode, conversationId, since, before, limit };

    let messages: MessageSearchResult[] = [];
    let summaries: SummarySearchResult[] = [];
    let marbles: MarbleSearchResult[] = [];
    const maybeSearchMarbles = async (): Promise<MarbleSearchResult[]> => {
      const store = this.summaryStore as unknown as {
        searchMarbles?: (input: typeof searchInput) => Promise<MarbleSearchResult[]>;
      };
      if (typeof store.searchMarbles !== "function") {
        return [];
      }
      return store.searchMarbles(searchInput);
    };

    if (scope === "messages") {
      messages = await this.conversationStore.searchMessages(searchInput);
    } else if (scope === "summaries") {
      summaries = await this.summaryStore.searchSummaries(searchInput);
    } else if (scope === "marbles") {
      marbles = await maybeSearchMarbles();
    } else {
      // scope === "both" — run in parallel across all artifact rails.
      [messages, summaries, marbles] = await Promise.all([
        this.conversationStore.searchMessages(searchInput),
        this.summaryStore.searchSummaries(searchInput),
        maybeSearchMarbles(),
      ]);
    }

    messages.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
    summaries.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
    marbles.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

    return {
      messages,
      summaries,
      marbles,
      totalMatches: messages.length + summaries.length + marbles.length,
    };
  }

  // ── expand ───────────────────────────────────────────────────────────────

  /**
   * Expand a summary to its children and/or source messages.
   *
   * - Condensed summaries: returns child summaries, recursing up to `depth`.
   * - Leaf summaries with `includeMessages`: fetches the source messages.
   * - Respects `tokenCap` and sets `truncated` when the cap is exceeded.
   */
  async expand(input: ExpandInput): Promise<ExpandResult> {
    const depth = input.depth ?? 1;
    const includeMessages = input.includeMessages ?? false;
    const tokenCap = input.tokenCap ?? Infinity;

    const result: ExpandResult = {
      children: [],
      messages: [],
      estimatedTokens: 0,
      truncated: false,
    };

    await this.expandRecursive(input.summaryId, depth, includeMessages, tokenCap, result);

    return result;
  }

  private async expandRecursive(
    summaryId: string,
    depth: number,
    includeMessages: boolean,
    tokenCap: number,
    result: ExpandResult,
  ): Promise<void> {
    if (depth <= 0) {
      return;
    }
    if (result.truncated) {
      return;
    }

    const summary = await this.summaryStore.getSummary(summaryId);
    if (!summary) {
      return;
    }

    if (summary.kind === "condensed") {
      const children = await this.summaryStore.getSummaryChildren(summaryId);

      for (const child of children) {
        if (result.truncated) {
          break;
        }

        // Check if adding this child would exceed the token cap
        if (result.estimatedTokens + child.tokenCount > tokenCap) {
          result.truncated = true;
          break;
        }

        result.children.push({
          summaryId: child.summaryId,
          kind: child.kind,
          content: child.content,
          tokenCount: child.tokenCount,
        });
        result.estimatedTokens += child.tokenCount;

        // Recurse into children if depth allows
        if (depth > 1) {
          await this.expandRecursive(child.summaryId, depth - 1, includeMessages, tokenCap, result);
        }
      }
    } else if (summary.kind === "leaf" && includeMessages) {
      // Leaf summary — fetch source messages
      const messageIds = await this.summaryStore.getSummaryMessages(summaryId);

      for (const msgId of messageIds) {
        if (result.truncated) {
          break;
        }

        const msg = await this.conversationStore.getMessageById(msgId);
        if (!msg) {
          continue;
        }

        const tokenCount = msg.tokenCount || estimateTokens(msg.content);

        if (result.estimatedTokens + tokenCount > tokenCap) {
          result.truncated = true;
          break;
        }

        result.messages.push({
          messageId: msg.messageId,
          role: msg.role,
          content: msg.content,
          tokenCount,
        });
        result.estimatedTokens += tokenCount;
      }
    }
  }
}
