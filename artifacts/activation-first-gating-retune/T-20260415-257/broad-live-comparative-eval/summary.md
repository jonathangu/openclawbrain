# Comparative Eval Runner

- status: `ok`
- gate: `fail`
- gate decisive: `true`
- manifest path: `/Users/guclaw/.openclaw/workspace/task-artifacts/T-20260415-250/semantic-rich-live-535-extracted/manifest.json`
- manifest contract: `frozen_recorded_session_eval_manifest.v1`
- manifest id: `extracted-semantic-rich-live-535`
- git sha: `13431ef43a4b41ac24a32c43ef92cd41ffff422a`
- traces: 403/403
- scorecard hash: `sha256-3a3719f2fc53555d68cdf5612a5426addcdb64bcc7f454fbfed3358dd17668fb`
- explainable scorecard hash: `sha256-aed3ba53bbc1f1301db29c8236da25d507e886d2f632a1b3b2b07a973afc52a9`

## Explainable Scorecard
- learned_route was worse than graph_prior_only on 30/403 validated traces.
- learned_route tied or beat graph_prior_only on 373/403 validated traces (better 1, tied 372, worse 30).
- learned_route retrieved required replay phrases at 0.022837 recall versus 0.064904 for graph_prior_only.
- learned_route used not available (estimated prompt cost per replay-successful trace) versus graph_prior_only.
- Summary-routing telemetry for expand-before-assert is not available in this bundle.
- learned_route tie-or-better vs graph_prior_only (traces): 373/403 (0.925558)
- learned_route vs graph_prior_only (traces): 1 better, 372 tied, 30 worse
- learned_route tie-or-better vs graph_prior_only (turns): 373/403 (0.925558)
- learned_route vs graph_prior_only (turns): 1 better, 372 tied, 30 worse
- regressions vs graph_prior_only: 30/403 (0.074442)
- regressions vs no_brain: 0/403 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 19/832 required-context phrases vs graph_prior_only 54/832
- correction absorption: correction absorption is unavailable in comparative eval because no feedback-bearing turns were recorded in the validated set
- success-adjusted economics: learned_route estimated prompt cost per validated trace = 0.000142 vs graph_prior_only 0.000387
- fail-open: fail-open posture is not modeled in comparative eval replay bundles; use proof-cron health surfaces for degraded-serve reporting

## Public / Operator Metrics
- learned_route was worse than graph_prior_only on 30/403 validated traces.
- learned_route tied or beat graph_prior_only on 373/403 validated traces (better 1, tied 372, worse 30).
- learned_route retrieved required replay phrases at 0.022837 recall versus 0.064904 for graph_prior_only.
- learned_route used not available (estimated prompt cost per replay-successful trace) versus graph_prior_only.
- Summary-routing telemetry for expand-before-assert is not available in this bundle.
| metric | availability | value | formula | language |
| --- | --- | ---: | --- | --- |
| brain_on_regression_rate_vs_prior | proxy | 0.074442 | worseThanPriorCount / comparableTraceCount | learned_route was worse than graph_prior_only on 30/403 validated traces. |
| brain_on_regression_rate_vs_no_brain | proxy | 0 | worseThanNoBrainCount / comparableTraceCount | learned_route was worse than no_brain on 0/403 validated traces. |
| critical_regression_rate_vs_prior | proxy | 0.074442 | criticalRegressionCount / comparableTraceCount | critical regressions were observed on 30/403 validated traces when compile coverage or required-context hits worsened versus graph_prior_only. |
| tie_or_better_rate_vs_prior | proxy | 0.925558 | (betterThanPriorCount + tiedWithPriorCount) / comparableTraceCount | learned_route tied or beat graph_prior_only on 373/403 validated traces (better 1, tied 372, worse 30). |
| required_context_recall | measured | 0.022837 | retrievedRequiredEvidenceCount / totalRequiredEvidenceCount | learned_route retrieved required replay phrases at 0.022837 recall versus 0.064904 for graph_prior_only. |
| missing_required_context_rate | measured | 0.977163 | missingRequiredEvidenceCount / totalRequiredEvidenceCount | learned_route missed 813/832 required replay phrases. |
| estimated_prompt_tokens_per_successful_trace_delta_vs_prior | proxy | null | (candidatePromptTokens / candidateSuccessfulTraceProxyCount) - (priorPromptTokens / priorSuccessfulTraceProxyCount) | learned_route used not available (estimated prompt tokens per replay-successful trace) versus graph_prior_only. |
| estimated_prompt_cost_per_successful_trace_delta_vs_prior | proxy | null | (candidatePromptCostUsd / candidateSuccessfulTraceProxyCount) - (priorPromptCostUsd / priorSuccessfulTraceProxyCount) | learned_route used not available (estimated prompt cost per replay-successful trace) versus graph_prior_only. |
| estimated_prompt_tokens_per_successful_trace_delta_vs_no_brain | proxy | null | (candidatePromptTokens / candidateSuccessfulTraceProxyCount) - (noBrainPromptTokens / noBrainSuccessfulTraceProxyCount) | learned_route used not available (estimated prompt tokens per replay-successful trace) versus no_brain. |
| estimated_prompt_cost_per_successful_trace_delta_vs_no_brain | proxy | null | (candidatePromptCostUsd / candidateSuccessfulTraceProxyCount) - (noBrainPromptCostUsd / noBrainSuccessfulTraceProxyCount) | learned_route used not available (estimated prompt cost per replay-successful trace) versus no_brain. |
| expand_before_assert_rate | not_available | null | expandToSourceCount / summaryRoutingCount | Summary-routing telemetry for expand-before-assert is not available in this bundle. |
| branch_heavy_expand_to_source_rate | not_available | null | branchHeavyExpandToSourceCount / branchHeavySummaryRoutingCount | Branch-heavy compact-history telemetry is not available in this bundle. |
| non_fresh_summary_prevalence | not_available | null | nonFreshSummaryCount / summaryCount | Non-fresh summary telemetry is not available in this bundle. |
| snapshot_vs_condense_share | not_available | null | snapshotPassCount / compactionPassCount | Snapshot-versus-condense telemetry is not available in this bundle. |
| token_reduction_per_compaction_pass | not_available | null | (tokensBefore - tokensAfter) / compactionPassCount | Compaction token-reduction telemetry is not available in this bundle. |
| safe_fallback_rate | not_available | null | safeFallbackCount / degradedBrainInvocationCount | Comparative replay does not observe live safe-fallback invocations, so safe fallback rate is not computed here. |
| worker_down_safe_serve_rate | not_available | null | workerDownSafeServeCount / workerDownInvocationCount | Comparative replay does not simulate worker-down serving, so worker-down safe serve rate is not computed here. |
| brain_disabled_comparable_success_rate | proxy | 0 | successfulBrainDisabledTasks / comparableBrainDisabledTasks | the no_brain floor met the replay success proxy on 0/403 validated traces. |

## Fail-Open Language
- Comparative replay does not prove live safe-fallback or worker-down serving. It does expose a no_brain floor anchor: 0/403 validated traces met the replay success proxy under no_brain.

## Policy
- candidate mode: `learned_route`
- baseline mode: `graph_prior_only`
- floor mode: `no_brain`
- comparable traces: 403
- successful traces: 403
- failed traces: 0
| check | status | observed | threshold |
| --- | --- | --- | --- |
| trace_coverage_complete | pass | requestedTraceCount=403, successfulTraceCount=403, failedTraceCount=0 | maxFailedTraceCount=0 |
| candidate_trace_tie_or_better_vs_baseline | fail | candidateMode=learned_route, baselineMode=graph_prior_only, comparableTraceCount=403, candidateTraceTieOrBetterCountVsBaseline=373, candidateTraceTieOrBetterRateVsBaseline=0.925558 | minCandidateTraceTieOrBetterRateVsBaseline=1 |
| candidate_tie_promotion_delta_vs_baseline | pass | candidateMode=learned_route, baselineMode=graph_prior_only, candidateTieTraceCountVsBaseline=372, candidateTiePromotionDeltaVsBaseline=0 | maxCandidateTiePromotionDeltaVsBaseline=0 |
| candidate_mean_quality_regression_vs_baseline | pass | baselineMeanQualityScore=43.424318, candidateMeanQualityScore=40.992556, candidateMeanQualityRegressionVsBaseline=2.431762 | maxCandidateMeanQualityRegressionVsBaseline=5 |
| baseline_mean_quality_gain_vs_floor | pass | baselineMeanQualityScore=43.424318, floorMeanQualityScore=0, baselineMeanQualityGainVsFloor=43.424318 | minBaselineMeanQualityGainVsFloor=5 |

## Internal Diagnostics
- qualityScore and winnerMode are preserved only as internal deterministic replay diagnostics; they are not the public/operator definition of success.
| metric | value | language |
| --- | ---: | --- |
| diagnostic_quality_score_mean_by_mode | 40.992556 | qualityScore remains an internal deterministic replay composite for smoke comparisons and tuning only. |
| diagnostic_ranked_winner_count_by_mode | 1 | winnerMode is retained only as an internal tie-break and ranking surface. |
| diagnostic_shared_top_score_trace_count_by_mode | 366 | Shared-top counts remain an internal replay diagnostic for tie analysis. |

## Diagnostic Pairwise
| pair | traces | left/right/tied | left tie-or-better rate | right tie-or-better rate | mean quality delta |
| --- | ---: | --- | ---: | ---: | ---: |
| no_brain vs vector_only | 403 | 0-403-0 | 0 | 1 | -44.119107 |
| no_brain vs graph_prior_only | 403 | 0-403-0 | 0 | 1 | -43.424318 |
| no_brain vs learned_route | 403 | 0-403-0 | 0 | 1 | -40.992556 |
| vector_only vs graph_prior_only | 403 | 8-0-395 | 1 | 0.980149 | 0.694789 |
| vector_only vs learned_route | 403 | 37-0-366 | 1 | 0.908189 | 3.126551 |
| graph_prior_only vs learned_route | 403 | 30-1-372 | 0.997519 | 0.925558 | 2.431762 |

## Trace Coverage
| trace | status | validation ok | candidate vs prior | candidate vs floor | diagnostic top mode | score spread | error |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003 | ok | true | worse | better | graph_prior_only | 70 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-002 | ok | true | worse | better | graph_prior_only | 80 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-006 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-010 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-020 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-023 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-025 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-029 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-053 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-062 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-065 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-071 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-077 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-080 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-082 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-085 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088 | ok | true | tied | better | graph_prior_only | 70 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005 | ok | true | tied | better | graph_prior_only | 70 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-main-1df6876b-e41e-4352-8c17-b6d259ab93af-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-40299bc1-00ef-445f-960b-1b1147ffd61f-window-001 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-468355da-cd1f-40fe-adc8-e1dc6dfa55ea-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002 | ok | true | worse | better | graph_prior_only | 80 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009 | ok | true | tied | better | vector_only | 60 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-011 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002 | ok | true | tied | better | vector_only | 80 | none |
| live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002 | ok | true | tied | better | vector_only | 70 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015 | ok | true | tied | better | vector_only | 100 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035 | ok | true | worse | better | graph_prior_only | 100 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048 | ok | true | tied | better | vector_only | 100 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-059 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-060 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-079 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-006 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004 | ok | true | better | better | learned_route | 60 | none |
| live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003 | ok | true | tied | better | vector_only | 70 | none |
| live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-120 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181 | ok | true | tied | better | graph_prior_only | 80 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182 | ok | true | tied | better | graph_prior_only | 80 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184 | ok | true | tied | better | graph_prior_only | 80 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233 | ok | true | worse | better | graph_prior_only | 60 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257 | ok | true | worse | better | graph_prior_only | 70 | none |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-012 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-013 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002 | ok | true | tied | better | vector_only | 60 | none |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005 | ok | true | worse | better | graph_prior_only | 70 | none |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002 | ok | true | worse | better | graph_prior_only | 70 | none |
| live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-013 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-014 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |

## Policy Reasons
- candidate_trace_tie_or_better_vs_baseline: candidate missed the configured per-trace tie-or-better rate versus the baseline

## Notes
- default manifest path is /Users/guclaw/.openclaw/workspace/openclawbrain/evals/recorded-session-replay/canonical-frozen-20/manifest.json
- mode order is no_brain, vector_only, graph_prior_only, learned_route
- the comparative runner delegates replay execution to writeRecordedSessionReplayProofLane so each trace still runs through the real replay/runtime path
- learned_route replay override artifact: /Users/guclaw/.openclaw/workspace/openclawbrain/artifacts/activation-first-gating-retune/T-20260415-257/candidate-artifact
- candidate override runs are non-authoritative for served learned-route hotpath truth because replay override keeps usedLearnedRouteFn=false by construction
- Extracted from /Users/guclaw/.openclaw/workspace/task-artifacts/T-20260415-250/semantic-rich-live-535.json
- Internal local-only live-history replay traces.
- One-turn traces with prior session messages converted into seed cues.
- Not approved for public export without a separate redaction pass.

## Assumptions
- accepted manifest contracts: canonical_recorded_session_trace_set_manifest.v1, frozen_recorded_session_eval_manifest.v1
- manifest trace paths resolve relative to the manifest file location
- traceHash, when present in the manifest, is checksumJsonPayload(trace-json)
- scorecard prompt-cost metrics are cheap deterministic proxies derived from selected context chars
- learned_route is the candidate mode, graph_prior_only is the baseline mode, and no_brain is the floor anchor for the explicit comparative policy
- when provided, learned_route replay uses the supplied candidate artifact instead of replay-trained route_fn state
- candidate override replay does not bind the candidate as the served learned-route router, so authoritative broad-live verdicts still require a served-pack bridge
- this scaffold does not finalize the frozen trace set or widen proof-bundle generation scope
