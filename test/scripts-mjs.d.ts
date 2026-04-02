declare module "../scripts/proof-cron.mjs" {
  export const PROOF_CRON_MANIFEST_LAYOUT: any;
  export const buildHealthSnapshot: any;
  export const buildNightlyAggregate: any;
  export const buildProofManifestSkeleton: any;
  export const buildProofManifestSmoke: any;
  export const buildReplayManifestSkeletonSet: any;
  export const collectBundleCandidates: any;
  export const formatHealthMarkdown: any;
  export const formatNightlyMarkdown: any;
  export const loadConfig: any;
  export const summarizeScan: any;
  export const writeHealthOutputs: any;
  export const writeNightlyOutputs: any;
}

declare module "../scripts/verify-proof-smoke.mjs" {
  export const DEFAULT_MAX_AGE_DAYS: any;
  export const REQUIRED_ASSERTION_KEYS: any;
  export const REQUIRED_PROOF_FILES: any;
  export const verifyProofSmoke: any;
}

declare module "../scripts/release-plan.mjs" {
  export const buildReleasePlan: any;
  export const verifyReleasePlan: any;
}
