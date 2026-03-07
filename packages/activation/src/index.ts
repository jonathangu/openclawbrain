export {
  ACTIVATION_LAYOUT,
  activatePack,
  describeActivationObservability,
  describeActivationTarget,
  describePackCompileTarget,
  inspectActivationState,
  loadPackFromActivation,
  loadActivationPointers,
  promoteCandidatePack,
  rollbackActivePack,
  stageCandidatePack,
  validatePackActivationReadiness,
  type ActivationObservabilityOptions,
  type ActivationObservabilityReport,
  type ActivationInspection,
  type ActivationOperationPreview,
  type GraphDynamicsFreshnessReport,
  type LearnedRouteFnFreshnessReport,
  type PromotionFreshnessDeltaReport,
  type PromotionFreshnessReport,
  type ActivationSlotInspection,
  type ActivationStateDescriptor
} from "@openclawbrain/pack-format";

export {
  FIXTURE_ACTIVATION_POINTERS,
  type ActivationPointerRecordV1,
  type ActivationPointerSlot,
  type ActivationPointersV1
} from "@openclawbrain/contracts";
