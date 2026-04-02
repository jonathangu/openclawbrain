import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { toAttributionTruthId, toAttributionUpdateId } from "../../src/brain-core/trace.js";
import { closeLcmConnection, getLcmConnection } from "../../src/db/connection.js";
import { runLcmMigrations } from "../../src/db/migration.js";
import { ConversationStore } from "../../src/store/conversation-store.js";
import { SummaryStore } from "../../src/store/summary-store.js";

const tempDirs: string[] = [];

afterEach(() => {
  closeLcmConnection();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("SummaryStore attribution truth", () => {
  it("round-trips delayed, matched, and unmatched attribution truth records", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "lossless-claw-attribution-truth-"));
    tempDirs.push(tempDir);
    const db = getLcmConnection(join(tempDir, "attribution-truth.db"));
    runLcmMigrations(db);

    const conversationStore = new ConversationStore(db);
    const conversation = await conversationStore.createConversation({
      sessionId: "attribution-truth-session",
      title: "Attribution truth store test",
    });

    const store = new SummaryStore(db);
    const updateId = toAttributionUpdateId({
      episodeId: "ep_attr_001",
      observationIds: ["bo_2", "bo_1"],
      supervisionIds: ["ts_2", "ts_1"],
      traceIds: ["bt_2", "bt_1"],
    });
    const attributionTruthId = toAttributionTruthId({
      observationId: "bo_1",
      supervisionId: "ts_1",
      updateId,
      episodeId: "ep_attr_001",
    });
    const firstCreatedAt = new Date("2026-04-02T19:00:00.000Z");
    const firstUpdatedAt = new Date("2026-04-02T19:00:00.000Z");
    const secondUpdatedAt = new Date("2026-04-02T19:05:00.000Z");

    const delayed = await store.insertAttributionTruth({
      attributionTruthId,
      conversationId: conversation.conversationId,
      episodeId: "ep_attr_001",
      state: "delayed",
      observation: {
        observationId: "bo_1",
        episodeId: "ep_attr_001",
        conversationId: conversation.conversationId,
        traceId: "bt_1",
        bindingMode: "exact_decision_id",
        requestDigest: "digest_1",
        serveDecisionRecordId: "decision_1",
        selectionDigest: "selection_1",
        turnCompileEventId: "compile_1",
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "delayed",
          basis: "pending_observation",
          confidence: null,
          detail: "waiting for supervision to arrive",
          candidateIds: [],
        },
        supervisionToUpdate: {
          state: "delayed",
          basis: "pending_update",
          confidence: null,
          detail: "no learner update emitted yet",
          candidateIds: [],
        },
      },
      createdAt: firstCreatedAt,
      updatedAt: firstUpdatedAt,
    });

    expect(delayed.attributionTruthId).toBe(attributionTruthId);
    expect(delayed.state).toBe("delayed");
    expect(delayed.createdAt.toISOString()).toBe(firstCreatedAt.toISOString());
    expect(delayed.linkage.observationToSupervision.state).toBe("delayed");

    const matched = await store.insertAttributionTruth({
      attributionTruthId,
      conversationId: conversation.conversationId,
      episodeId: "ep_attr_001",
      state: "matched",
      observation: {
        observationId: "bo_1",
        episodeId: "ep_attr_001",
        conversationId: conversation.conversationId,
        traceId: "bt_1",
        bindingMode: "exact_decision_id",
        requestDigest: "digest_1",
        serveDecisionRecordId: "decision_1",
        selectionDigest: "selection_1",
        turnCompileEventId: "compile_1",
        provenanceRef: null,
      },
      supervision: {
        supervisionId: "ts_1",
        episodeId: "ep_attr_001",
        conversationId: conversation.conversationId,
        source: "teacher",
        kind: "teacher_review",
        observationId: "bo_1",
        traceId: "bt_1",
        teacherTraceId: "bt_1",
        serveDecisionRecordId: "decision_1",
        selectionDigest: "selection_1",
        turnCompileEventId: "compile_1",
        bindingMode: "exact_decision_id",
        attributionQuality: "exact",
        feedbackRichness: "followup_and_tool",
        traceRequestDigest: "digest_1",
        provenanceRef: null,
      },
      update: {
        updateId,
        episodeId: "ep_attr_001",
        observationIds: ["bo_2", "bo_1", "bo_1"],
        supervisionIds: ["ts_2", "ts_1"],
        traceIds: ["bt_2", "bt_1"],
        rewardSource: "teacher",
        attributionQuality: "mixed",
        feedbackRichness: "mixed",
        routeUpdateCount: 4,
        seedUpdateCount: 1,
        stopLocalUpdateCount: 1,
        edgeUpdateCount: 2,
        updateReason: "teacher attribution matched and updated 4 route weight(s)",
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "matched",
          basis: "decision_record_id",
          confidence: 1,
          detail: "decision id bound observation and supervision exactly",
          candidateIds: ["bo_1"],
        },
        supervisionToUpdate: {
          state: "matched",
          basis: "manual",
          confidence: 1,
          detail: "the learner update consumed the supervision directly",
          candidateIds: [updateId],
        },
      },
      createdAt: firstCreatedAt,
      updatedAt: secondUpdatedAt,
    });

    expect(matched.state).toBe("matched");
    expect(matched.update?.observationIds).toEqual(["bo_1", "bo_2"]);
    expect(matched.update?.supervisionIds).toEqual(["ts_1", "ts_2"]);
    expect(matched.update?.traceIds).toEqual(["bt_1", "bt_2"]);
    expect(matched.createdAt.toISOString()).toBe(firstCreatedAt.toISOString());
    expect(matched.updatedAt.toISOString()).toBe(secondUpdatedAt.toISOString());
    expect(matched.contentHash).toMatch(/^hash_/);
    expect(matched.lineageHash).toMatch(/^lineage_/);
    expect(matched.provenanceRef).toMatch(/^prov_/);

    const unmatched = await store.insertAttributionTruth({
      conversationId: conversation.conversationId,
      episodeId: "ep_attr_002",
      state: "unmatched",
      supervision: {
        supervisionId: "ts_unmatched",
        episodeId: "ep_attr_002",
        conversationId: conversation.conversationId,
        source: "human",
        kind: "teach_correction",
        observationId: null,
        traceId: null,
        teacherTraceId: null,
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        bindingMode: "unbound",
        attributionQuality: "unbound",
        feedbackRichness: "sparse",
        traceRequestDigest: null,
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "unmatched",
          basis: "missing",
          confidence: null,
          detail: "no qualifying observation was available",
          candidateIds: [],
        },
        supervisionToUpdate: {
          state: "delayed",
          basis: "pending_update",
          confidence: null,
          detail: "teacher correction has not produced an update yet",
          candidateIds: [],
        },
      },
      createdAt: new Date("2026-04-02T19:10:00.000Z"),
    });

    expect(unmatched.state).toBe("unmatched");
    expect(unmatched.supervision?.supervisionId).toBe("ts_unmatched");

    const reloaded = await store.getAttributionTruth(attributionTruthId);
    expect(reloaded).not.toBeNull();
    expect(reloaded?.state).toBe("matched");
    expect(reloaded?.linkage.observationToSupervision.basis).toBe("decision_record_id");

    const byConversation = await store.getAttributionTruthsByConversation(conversation.conversationId);
    expect(byConversation).toHaveLength(2);
    expect(byConversation.map((record) => record.state)).toEqual(["matched", "unmatched"]);

    const byEpisode = await store.getAttributionTruthsByEpisode("ep_attr_001");
    expect(byEpisode).toHaveLength(1);
    expect(byEpisode[0]?.attributionTruthId).toBe(attributionTruthId);
  });

  it("preserves delayed and matched truth states under distinct autogenerated ids for the same lineage", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "lossless-claw-attribution-truth-auto-id-"));
    tempDirs.push(tempDir);
    const db = getLcmConnection(join(tempDir, "attribution-truth-auto-id.db"));
    runLcmMigrations(db);

    const conversationStore = new ConversationStore(db);
    const conversation = await conversationStore.createConversation({
      sessionId: "attribution-truth-auto-id-session",
      title: "Attribution truth auto-id test",
    });

    const store = new SummaryStore(db);
    const delayed = await store.insertAttributionTruth({
      conversationId: conversation.conversationId,
      episodeId: "ep_attr_auto",
      state: "delayed",
      observation: {
        observationId: "bo_auto",
        episodeId: "ep_attr_auto",
        conversationId: conversation.conversationId,
        traceId: "bt_auto",
        bindingMode: "trace_id",
        requestDigest: "digest_auto",
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "delayed",
          basis: "pending_observation",
          confidence: null,
          detail: "teacher supervision has not arrived yet",
          candidateIds: [],
        },
        supervisionToUpdate: {
          state: "delayed",
          basis: "pending_update",
          confidence: null,
          detail: "no learner update exists yet",
          candidateIds: [],
        },
      },
      createdAt: new Date("2026-04-02T20:00:00.000Z"),
    });

    const matched = await store.insertAttributionTruth({
      conversationId: conversation.conversationId,
      episodeId: "ep_attr_auto",
      state: "matched",
      observation: {
        observationId: "bo_auto",
        episodeId: "ep_attr_auto",
        conversationId: conversation.conversationId,
        traceId: "bt_auto",
        bindingMode: "trace_id",
        requestDigest: "digest_auto",
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        provenanceRef: null,
      },
      supervision: {
        supervisionId: "ts_auto",
        episodeId: "ep_attr_auto",
        conversationId: conversation.conversationId,
        source: "teacher",
        kind: "teacher_review",
        observationId: "bo_auto",
        traceId: "bt_auto",
        teacherTraceId: "bt_auto",
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        bindingMode: "trace_id",
        attributionQuality: "fallback",
        feedbackRichness: "followup_only",
        traceRequestDigest: "digest_auto",
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "matched",
          basis: "trace_id",
          confidence: 1,
          detail: "trace id preserved the attribution link",
          candidateIds: ["bo_auto"],
        },
        supervisionToUpdate: {
          state: "matched",
          basis: "manual",
          confidence: 1,
          detail: "teacher supervision fed the learner update",
          candidateIds: ["upd_auto"],
        },
      },
      createdAt: new Date("2026-04-02T20:05:00.000Z"),
    });

    expect(delayed.attributionTruthId).not.toBe(matched.attributionTruthId);
    expect(delayed.lineageHash).toBe(matched.lineageHash);

    const byEpisode = await store.getAttributionTruthsByEpisode("ep_attr_auto");
    expect(byEpisode).toHaveLength(2);
    expect(byEpisode.map((record) => record.state)).toEqual(["delayed", "matched"]);
    expect(byEpisode.map((record) => record.attributionTruthId)).toEqual([
      delayed.attributionTruthId,
      matched.attributionTruthId,
    ]);
  });
});
