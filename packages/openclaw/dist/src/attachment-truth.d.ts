export type OpenClawHomeProfileSource =
  | "openclaw_json_profile"
  | "openclaw_json_single_profile_key"
  | "directory_name"
  | "none";

declare const ATTACHMENT_RUNTIME_LOAD_PROOFS_CONTRACT: "openclaw_profile_runtime_load_proofs.v1";

export interface OpenClawProfileRuntimeLoadProofRecordV1 {
  openclawHome: string;
  profileId: string | null;
  profileSource: OpenClawHomeProfileSource;
  loadedAt: string;
  extensionEntryPath: string;
}

export interface OpenClawProfileRuntimeLoadProofsV1 {
  contract: typeof ATTACHMENT_RUNTIME_LOAD_PROOFS_CONTRACT;
  runtimeOwner: "openclaw";
  activationRoot: string;
  updatedAt: string;
  profiles: OpenClawProfileRuntimeLoadProofRecordV1[];
}

export interface OpenClawProfileRuntimeLoadProofSetV1 {
  path: string;
  proofs: OpenClawProfileRuntimeLoadProofsV1 | null;
  error: string | null;
}

export declare function resolveAttachmentRuntimeLoadProofsPath(activationRoot: string): string;
export declare function listOpenClawProfileRuntimeLoadProofs(
  activationRoot: string,
): OpenClawProfileRuntimeLoadProofSetV1;
export declare function recordOpenClawProfileRuntimeLoadProof(input: {
  activationRoot: string;
  extensionEntryPath: string;
  loadedAt?: string | null;
}): OpenClawProfileRuntimeLoadProofRecordV1;
export declare function clearOpenClawProfileRuntimeLoadProof(input: {
  activationRoot: string;
  openclawHome: string;
  clearedAt?: string | null;
}): boolean;
