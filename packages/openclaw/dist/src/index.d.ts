import type { CompileSelectionMode } from "@openclawbrain/compiler";
import type {
  ActivationPointerRecordV1,
  BrainServeHotPathTimingV1,
  ContextCompactionMode,
  RouteMode,
  RuntimeCompileResponseV1,
} from "@openclawbrain/contracts";
import type { ActivationSlotInspection } from "@openclawbrain/pack-format";

export {
  clearOpenClawProfileRuntimeLoadProof,
  listOpenClawProfileRuntimeLoadProofs,
  recordOpenClawProfileRuntimeLoadProof,
  resolveAttachmentRuntimeLoadProofsPath,
  type OpenClawHomeProfileSource,
  type OpenClawProfileRuntimeLoadProofRecordV1,
  type OpenClawProfileRuntimeLoadProofSetV1,
  type OpenClawProfileRuntimeLoadProofsV1,
} from "./attachment-truth.js";

export interface CompileServeRouteBreadcrumbInput {
  invocationSurface: "direct_compile_call" | "installed_extension_before_prompt_build" | "runtime_turn_helper";
  hostEvent: "before_prompt_build" | null;
  installedEntryPath?: string | null;
}

export interface CompileRuntimeContextInput {
  activationRoot: string;
  message: string;
  agentId?: string;
  maxContextBlocks?: number;
  budgetStrategy?: "fixed_v1" | "empirical_v1";
  maxContextChars?: number;
  mode?: RouteMode;
  selectionMode?: CompileSelectionMode;
  compactionMode?: ContextCompactionMode;
  runtimeHints?: readonly string[];
  sessionId?: string;
  channel?: string;
  _suppressServeLog?: boolean;
  _serveRouteBreadcrumbs?: CompileServeRouteBreadcrumbInput;
}

export interface ActiveCompileTarget {
  activationRoot: string;
  activePointer: ActivationPointerRecordV1;
  inspection: ActivationSlotInspection;
}

export interface RuntimeCompileSuccess {
  ok: true;
  fallbackToStaticContext: false;
  hardRequirementViolated: false;
  activationRoot: string;
  activePackId: string;
  packRootDir: string;
  compileResponse: RuntimeCompileResponseV1;
  brainContext: string;
  timing: BrainServeHotPathTimingV1;
}

export interface RuntimeCompileFailOpenFailure {
  ok: false;
  fallbackToStaticContext: true;
  hardRequirementViolated: false;
  activationRoot: string;
  error: string;
  brainContext: string;
  timing: BrainServeHotPathTimingV1;
}

export interface RuntimeCompileHardFailure {
  ok: false;
  fallbackToStaticContext: false;
  hardRequirementViolated: true;
  activationRoot: string;
  error: string;
  brainContext: string;
  timing: BrainServeHotPathTimingV1;
}

export type RuntimeCompileFailure = RuntimeCompileFailOpenFailure | RuntimeCompileHardFailure;
export type RuntimeCompileResult = RuntimeCompileSuccess | RuntimeCompileFailure;

export declare function compileRuntimeContext(input: CompileRuntimeContextInput): RuntimeCompileResult;
