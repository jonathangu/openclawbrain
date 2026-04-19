# Comparative Eval Runner

- status: `ok`
- gate: `pass`
- gate decisive: `true`
- manifest path: `/Users/guclaw/.openclaw/workspace/task-artifacts/T-20260415-257/activation-first-retune-harness/felt-resume-eval.manifest.json`
- manifest contract: `frozen_recorded_session_eval_manifest.v1`
- manifest id: `felt_resume_25-eval`
- git sha: `627639721049957f7920995f24b550f618afa58d`
- traces: 25/25
- scorecard hash: `sha256-6c5125fd585593420504a74d9a30f86bdb53d33468adb7e70dd78ead1fef8218`
- explainable scorecard hash: `sha256-0679fd7bae5c6980ea96e3075d93f1e93ee3243787b83f4cfd05bef553d84eb8`

## Explainable Scorecard
- learned_route was worse than graph_prior_only on 0/25 validated traces.
- learned_route tied or beat graph_prior_only on 25/25 validated traces (better 0, tied 25, worse 0).
- learned_route retrieved required replay phrases at 0.047619 recall versus 0.047619 for graph_prior_only.
- learned_route used not available (estimated prompt cost per replay-successful trace) versus graph_prior_only.
- Summary-routing telemetry for expand-before-assert is not available in this bundle.
- learned_route tie-or-better vs graph_prior_only (traces): 25/25 (1)
- learned_route vs graph_prior_only (traces): 0 better, 25 tied, 0 worse
- learned_route tie-or-better vs graph_prior_only (turns): 25/25 (1)
- learned_route vs graph_prior_only (turns): 0 better, 25 tied, 0 worse
- regressions vs graph_prior_only: 0/25 (0)
- regressions vs no_brain: 0/25 (0) (critical regressions: 0)
- required-context recall: learned_route recalled 3/63 required-context phrases vs graph_prior_only 3/63
- correction absorption: correction absorption is unavailable in comparative eval because no feedback-bearing turns were recorded in the validated set
- success-adjusted economics: learned_route estimated prompt cost per validated trace = 0.000168 vs graph_prior_only 0.000358
- fail-open: fail-open posture is not modeled in comparative eval replay bundles; use proof-cron health surfaces for degraded-serve reporting

## Public / Operator Metrics
- learned_route was worse than graph_prior_only on 0/25 validated traces.
- learned_route tied or beat graph_prior_only on 25/25 validated traces (better 0, tied 25, worse 0).
- learned_route retrieved required replay phrases at 0.047619 recall versus 0.047619 for graph_prior_only.
- learned_route used not available (estimated prompt cost per replay-successful trace) versus graph_prior_only.
- Summary-routing telemetry for expand-before-assert is not available in this bundle.
| metric | availability | value | formula | language |
| --- | --- | ---: | --- | --- |
| brain_on_regression_rate_vs_prior | proxy | 0 | worseThanPriorCount / comparableTraceCount | learned_route was worse than graph_prior_only on 0/25 validated traces. |
| brain_on_regression_rate_vs_no_brain | proxy | 0 | worseThanNoBrainCount / comparableTraceCount | learned_route was worse than no_brain on 0/25 validated traces. |
| critical_regression_rate_vs_prior | proxy | 0 | criticalRegressionCount / comparableTraceCount | critical regressions were observed on 0/25 validated traces when compile coverage or required-context hits worsened versus graph_prior_only. |
| tie_or_better_rate_vs_prior | proxy | 1 | (betterThanPriorCount + tiedWithPriorCount) / comparableTraceCount | learned_route tied or beat graph_prior_only on 25/25 validated traces (better 0, tied 25, worse 0). |
| required_context_recall | measured | 0.047619 | retrievedRequiredEvidenceCount / totalRequiredEvidenceCount | learned_route retrieved required replay phrases at 0.047619 recall versus 0.047619 for graph_prior_only. |
| missing_required_context_rate | measured | 0.952381 | missingRequiredEvidenceCount / totalRequiredEvidenceCount | learned_route missed 60/63 required replay phrases. |
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
| brain_disabled_comparable_success_rate | proxy | 0 | successfulBrainDisabledTasks / comparableBrainDisabledTasks | the no_brain floor met the replay success proxy on 0/25 validated traces. |

## Fail-Open Language
- Comparative replay does not prove live safe-fallback or worker-down serving. It does expose a no_brain floor anchor: 0/25 validated traces met the replay success proxy under no_brain.

## Policy
- candidate mode: `learned_route`
- baseline mode: `graph_prior_only`
- floor mode: `no_brain`
- comparable traces: 25
- successful traces: 25
- failed traces: 0
| check | status | observed | threshold |
| --- | --- | --- | --- |
| trace_coverage_complete | pass | requestedTraceCount=25, successfulTraceCount=25, failedTraceCount=0 | maxFailedTraceCount=0 |
| candidate_trace_tie_or_better_vs_baseline | pass | candidateMode=learned_route, baselineMode=graph_prior_only, comparableTraceCount=25, candidateTraceTieOrBetterCountVsBaseline=25, candidateTraceTieOrBetterRateVsBaseline=1 | minCandidateTraceTieOrBetterRateVsBaseline=1 |
| candidate_tie_promotion_delta_vs_baseline | pass | candidateMode=learned_route, baselineMode=graph_prior_only, candidateTieTraceCountVsBaseline=25, candidateTiePromotionDeltaVsBaseline=0 | maxCandidateTiePromotionDeltaVsBaseline=0 |
| candidate_mean_quality_regression_vs_baseline | pass | baselineMeanQualityScore=42.8, candidateMeanQualityScore=42.8, candidateMeanQualityRegressionVsBaseline=0 | maxCandidateMeanQualityRegressionVsBaseline=5 |
| baseline_mean_quality_gain_vs_floor | pass | baselineMeanQualityScore=42.8, floorMeanQualityScore=0, baselineMeanQualityGainVsFloor=42.8 | minBaselineMeanQualityGainVsFloor=5 |

## Internal Diagnostics
- qualityScore and winnerMode are preserved only as internal deterministic replay diagnostics; they are not the public/operator definition of success.
| metric | value | language |
| --- | ---: | --- |
| diagnostic_quality_score_mean_by_mode | 42.8 | qualityScore remains an internal deterministic replay composite for smoke comparisons and tuning only. |
| diagnostic_ranked_winner_count_by_mode | 0 | winnerMode is retained only as an internal tie-break and ranking surface. |
| diagnostic_shared_top_score_trace_count_by_mode | 25 | Shared-top counts remain an internal replay diagnostic for tie analysis. |

## Diagnostic Pairwise
| pair | traces | left/right/tied | left tie-or-better rate | right tie-or-better rate | mean quality delta |
| --- | ---: | --- | ---: | ---: | ---: |
| no_brain vs vector_only | 25 | 0-25-0 | 0 | 1 | -42.8 |
| no_brain vs graph_prior_only | 25 | 0-25-0 | 0 | 1 | -42.8 |
| no_brain vs learned_route | 25 | 0-25-0 | 0 | 1 | -42.8 |
| vector_only vs graph_prior_only | 25 | 0-0-25 | 1 | 1 | 0 |
| vector_only vs learned_route | 25 | 0-0-25 | 1 | 1 | 0 |
| graph_prior_only vs learned_route | 25 | 0-0-25 | 1 | 1 | 0 |

## Trace Coverage
| trace | status | validation ok | candidate vs prior | candidate vs floor | diagnostic top mode | score spread | error |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002 | ok | true | tied | better | graph_prior_only | 60 | none |
| live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003 | ok | true | tied | better | graph_prior_only | 70 | none |
| live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-a96180ee-512c-47d8-b6a0-b2db38789889-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016 | ok | true | tied | better | graph_prior_only | 40 | none |
| live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002 | ok | true | tied | better | graph_prior_only | 40 | none |

## Policy Reasons
- none

## Notes
- default manifest path is /Users/guclaw/.openclaw/workspace/openclawbrain/evals/recorded-session-replay/canonical-frozen-20/manifest.json
- mode order is no_brain, vector_only, graph_prior_only, learned_route
- the comparative runner delegates replay execution to writeRecordedSessionReplayProofLane so each trace still runs through the real replay/runtime path
- learned_route replay override artifact: /Users/guclaw/.openclaw/workspace/openclawbrain/artifacts/activation-first-gating-retune/T-20260415-257-feedback-rankdup1-bonus1/candidate-artifact
- candidate override runs are non-authoritative for served learned-route hotpath truth because replay override keeps usedLearnedRouteFn=false by construction
- Primary felt optimize lane for session continuity / task resume wins where learned_route should recover recent task state and produce a materially better continuation than graph_prior_only.
- This felt lane is intentionally concrete: it is built from real resume and continuity prompts where the current turn alone is too thin to pick the right next action.
- The tranche is complete at 25 anchors so the next harness / proof loops can treat it as the headline optimize lane instead of a partial placeholder.
- selection rule: Trace must be a real resume, continue, handoff, or boot-recovery turn where the current prompt is materially under-specified without recent task state.
- selection rule: Prefer traces where the right continuation depends on active task, blocker, deliverable, or runtime state rather than wording polish.
- selection rule: Exclude trivial wrappers that are pure boilerplate and do not require recovering a concrete prior workstream.
- selection rule: Include a mix of fresh-session boot resumes, interrupted coding resumes, narrowed-scope continuations, and operator keep-going turns.
- selection rule: Measure success as materially better continuation choice, not tie-heavy restatement.

## Assumptions
- accepted manifest contracts: canonical_recorded_session_trace_set_manifest.v1, frozen_recorded_session_eval_manifest.v1
- manifest trace paths resolve relative to the manifest file location
- traceHash, when present in the manifest, is checksumJsonPayload(trace-json)
- scorecard prompt-cost metrics are cheap deterministic proxies derived from selected context chars
- learned_route is the candidate mode, graph_prior_only is the baseline mode, and no_brain is the floor anchor for the explicit comparative policy
- when provided, learned_route replay uses the supplied candidate artifact instead of replay-trained route_fn state
- candidate override replay does not bind the candidate as the served learned-route router, so authoritative broad-live verdicts still require a served-pack bridge
- this scaffold does not finalize the frozen trace set or widen proof-bundle generation scope
