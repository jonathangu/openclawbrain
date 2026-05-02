import type { TurnEventPacket } from './capture.js';
import type { MemoryStore } from './memory-store.js';
import { RouteLearning } from './route-learning.js';

export interface BackgroundLearningReport {
  outcomeResolutions: number;
  routeDecisionsResolved: number;
  routeExamplesCreated: number;
  memoryUpdates: number;
  snapshotId?: string;
  consolidatedMemories?: number;
  prunedMemories: number;
  lastRunAt: string;
}

export class BackgroundLearner {
  private store: MemoryStore;
  private config: any;

  constructor(options: { store: MemoryStore; config: any }) {
    this.store = options.store;
    this.config = options.config;
  }

  processOutcomeClassification(agentId: string, packet: TurnEventPacket): BackgroundLearningReport {
    let outcomeResolutions = 0;
    const observation = packet.toolObservations[0];
    const pending = this.store.getPendingInjections(agentId)
      .filter((injection) => (packet.runId ? injection.runId === packet.runId : true))
      .filter((injection) => (packet.turnId ? injection.turnId === packet.turnId : true))
      .slice(0, 10);

    if (observation) {
      for (const injection of pending) {
        this.store.resolveInjectionOutcome(injection.id, observation.ok ? 'tool_success' : 'tool_failure', observation.errorClass);
        outcomeResolutions += 1;
      }
    }

    const routeLearning = new RouteLearning({ store: this.store, config: this.config });
    const learned = routeLearning.run(agentId);

    return {
      outcomeResolutions,
      routeDecisionsResolved: learned.resolvedDecisions,
      routeExamplesCreated: learned.examplesCreated,
      memoryUpdates: learned.memoryUpdates,
      snapshotId: learned.snapshotId,
      prunedMemories: 0,
      lastRunAt: new Date().toISOString(),
    };
  }

  processAgentEnd(agentId: string, packet: TurnEventPacket): BackgroundLearningReport {
    let outcomeResolutions = 0;
    const pending = this.store.getPendingInjections(agentId)
      .filter((injection) => (packet.runId ? injection.runId === packet.runId : true))
      .filter((injection) => (packet.turnId ? injection.turnId === packet.turnId : true))
      .slice(0, 10);

    const correctionSignal = isCorrectionAfterInjection(packet.latestUserMessageRedacted);
    if (correctionSignal) {
      for (const injection of pending) {
        this.store.resolveInjectionOutcome(injection.id, 'user_corrected', correctionSignal);
        outcomeResolutions += 1;
      }
    } else if (pending.length > 0) {
      for (const injection of pending) {
        this.store.resolveInjectionOutcome(injection.id, 'accepted');
        outcomeResolutions += 1;
      }
    }

    const routeLearning = new RouteLearning({ store: this.store, config: this.config });
    const learned = routeLearning.run(agentId);

    return {
      outcomeResolutions,
      routeDecisionsResolved: learned.resolvedDecisions,
      routeExamplesCreated: learned.examplesCreated,
      memoryUpdates: learned.memoryUpdates,
      snapshotId: learned.snapshotId,
      prunedMemories: 0,
      lastRunAt: new Date().toISOString(),
    };
  }

  runMaintenance(agentId: string): BackgroundLearningReport {
    const learned = new RouteLearning({ store: this.store, config: this.config }).run(agentId);
    const consolidatedMemories = this.store.consolidateMemories(agentId);
    this.store.decayFreshness(agentId);
    const prunedMemories = this.store.pruneMemories(agentId, this.config.learning.maxMemoryNodesPerAgent);
    return {
      outcomeResolutions: 0,
      routeDecisionsResolved: learned.resolvedDecisions,
      routeExamplesCreated: learned.examplesCreated,
      memoryUpdates: learned.memoryUpdates,
      snapshotId: learned.snapshotId,
      consolidatedMemories,
      prunedMemories,
      lastRunAt: new Date().toISOString(),
    };
  }
}

function isCorrectionAfterInjection(message: string): string | null {
  const lower = message.toLowerCase();
  const correctionPatterns = [
    { pattern: /\b(actually|no[,\s]|not like that|wrong|incorrect|fix this|that's wrong)\b/, signal: 'explicit_correction' },
    { pattern: /\b(use |switch to|should be|should have been)\b.*\binstead\b/, signal: 'correction_with_replacement' },
    { pattern: /\b(i said|like i told|don't use|don't do|stop using)\b/, signal: 'repeated_correction' },
    { pattern: /\b(no[,\s].*should|but.*not|wrong.*should)\b/, signal: 'implicit_correction' },
  ];
  for (const { pattern, signal } of correctionPatterns) {
    if (pattern.test(lower)) return signal;
  }
  return null;
}
