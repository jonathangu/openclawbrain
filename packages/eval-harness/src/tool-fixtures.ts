import { readFile } from "node:fs/promises";

import type { EvalTrace, TraceToolCall } from "./trace.ts";

export interface ToolFixture {
  fixture_id: string;
  tool_name: string;
  read_only: true;
  captured_from: string;
  counts_as_product_evidence: boolean;
  result: unknown;
}

export type ToolFixtureMap = ReadonlyMap<string, Readonly<ToolFixture>>;

export interface ToolInvocationResult {
  tool_call_id: string;
  tool_name: string;
  fixture_id: string;
  read_only: true;
  result: unknown;
}

export interface EvalToolRuntime {
  readonly fixturesPath: string;
  invokeFixture(toolCall: TraceToolCall): ToolInvocationResult;
  invokeAll(trace: EvalTrace): ToolInvocationResult[];
}

export async function loadToolFixtures(fixturesPath: string): Promise<ToolFixtureMap> {
  const raw = await readFile(fixturesPath, "utf8");
  const parsed = JSON.parse(raw) as Record<string, ToolFixture>;
  const fixtures = new Map<string, Readonly<ToolFixture>>();

  for (const [fixtureId, fixture] of Object.entries(parsed)) {
    if (fixtureId !== fixture.fixture_id) {
      throw new Error(`Fixture key ${fixtureId} does not match fixture_id ${fixture.fixture_id}`);
    }
    if (fixture.read_only !== true) {
      throw new Error(`Fixture ${fixtureId} must be read_only=true`);
    }
    fixtures.set(fixtureId, deepFreeze(fixture));
  }

  return fixtures;
}

export function createFixtureRuntime(
  fixturesPath: string,
  fixtures: ToolFixtureMap,
): EvalToolRuntime {
  return {
    fixturesPath,
    invokeFixture(toolCall: TraceToolCall): ToolInvocationResult {
      const fixture = fixtures.get(toolCall.fixture_id);
      if (!fixture) {
        throw new Error(`Missing tool fixture: ${toolCall.fixture_id}`);
      }
      if (fixture.tool_name !== toolCall.name) {
        throw new Error(
          `Tool fixture ${fixture.fixture_id} is for ${fixture.tool_name}, not ${toolCall.name}`,
        );
      }
      if (toolCall.read_only !== true || fixture.read_only !== true) {
        throw new Error(`Tool call ${toolCall.id} is not read-only fixture-backed`);
      }
      return deepFreeze({
        tool_call_id: toolCall.id,
        tool_name: toolCall.name,
        fixture_id: toolCall.fixture_id,
        read_only: true,
        result: fixture.result,
      });
    },
    invokeAll(trace: EvalTrace): ToolInvocationResult[] {
      return (trace.tool_calls ?? []).map((toolCall) => this.invokeFixture(toolCall));
    },
  };
}

function deepFreeze<T>(value: T): Readonly<T> {
  if (value && typeof value === "object") {
    Object.freeze(value);
    for (const child of Object.values(value)) {
      deepFreeze(child);
    }
  }
  return value as Readonly<T>;
}
