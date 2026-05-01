import type { TurnEventPacket } from './capture.js';
import type { MemoryStore } from './memory-store.js';
import { RouteLearning } from './route-learning.js';

export interface BackgroundLearningReport {
  outcomeResolutions: number;
  routeDecisionsResolved: number;
  routeExamplesCreated: number;
  memoryUpdates: number;
  snapshotId?: string;
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

  runMaintenance(agentId: string): BackgroundLearningReport {
    const learned = new RouteLearning({ store: this.store, config: this.config }).run(agentId);
    const prunedMemories = this.store.pruneMemories(agentId, this.config.learning.maxMemoryNodesPerAgent);
    return {
      outcomeResolutions: 0,
      routeDecisionsResolved: learned.resolvedDecisions,
      routeExamplesCreated: learned.examplesCreated,
      memoryUpdates: learned.memoryUpdates,
      snapshotId: learned.snapshotId,
      prunedMemories,
      lastRunAt: new Date().toISOString(),
    };
  }
}
