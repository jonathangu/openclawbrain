import { describe, expect, it } from "vitest";

import {
  prepareBinaryGateV2SplitReplayInputsV1,
  type TrancheManifest,
} from "../../scripts/grade-binary-gate-v2-splits.ts";

function makeManifest(trancheId: string): TrancheManifest {
  return {
    contract: "learned_route_eval_tranche_manifest.v1",
    trancheId,
    anchors: [
      {
        traceId: "trace-1",
        sourcePath: "trace-1.json",
      },
    ],
  };
}

describe("grade-binary-gate-v2-splits", () => {
  it("collapses exact duplicate rows for split audits while leaving merged lanes untouched", () => {
    const duplicateRouteRows = [
      { row_id: "activation-first:must_not_fire_binary_gate_v2:trace-1" },
      { row_id: "activation-first:must_not_fire_binary_gate_v2:trace-1" },
    ] as any;
    const duplicatePolicyRows = [
      {
        row_id: "ps_trace_1",
        trace_id: "trace-1",
        trace_slice: {
          route_row_id: "activation-first:must_not_fire_binary_gate_v2:trace-1",
        },
      },
      {
        row_id: "ps_trace_1",
        trace_id: "trace-1",
        trace_slice: {
          route_row_id: "activation-first:must_not_fire_binary_gate_v2:trace-1",
        },
      },
    ] as any;

    const splitPrepared = prepareBinaryGateV2SplitReplayInputsV1({
      manifest: makeManifest("trap_operator_artifact"),
      routeRows: duplicateRouteRows,
      policyRows: duplicatePolicyRows,
    });
    expect(splitPrepared.laneKey).toBe("mustNotFire");
    expect(splitPrepared.routeRows).toHaveLength(1);
    expect(splitPrepared.policyRows).toHaveLength(1);
    expect(splitPrepared.dedupe).toEqual({
      exactDuplicateRouteRowsCollapsed: 1,
      exactDuplicatePolicyRowsCollapsed: 1,
      conflictingDuplicateRouteRowIds: [],
      conflictingDuplicatePolicyRowIds: [],
    });

    const mergedPrepared = prepareBinaryGateV2SplitReplayInputsV1({
      manifest: makeManifest("must_not_fire_binary_gate_v2"),
      routeRows: duplicateRouteRows,
      policyRows: duplicatePolicyRows,
    });
    expect(mergedPrepared.routeRows).toHaveLength(2);
    expect(mergedPrepared.policyRows).toHaveLength(2);
    expect(mergedPrepared.dedupe).toEqual({
      exactDuplicateRouteRowsCollapsed: 0,
      exactDuplicatePolicyRowsCollapsed: 0,
      conflictingDuplicateRouteRowIds: [],
      conflictingDuplicatePolicyRowIds: [],
    });
  });
});
