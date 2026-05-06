import { hashText, redactText } from './redact.js';
import { buildCalibrationSummaryV3, applyCalibrationV3 } from './route-policy-v3-calibration.js';
import { canonicalActionKeyV3, compactnessSummaryV3, mergeRuleCandidatesV3, normalizeQueryTemplateFamilyV3, pruneDominatedRulesV3 } from './route-policy-v3-normalize.js';
import { summarizeReplayEvaluationV3 } from './route-policy-v3-eval.js';
import { detectRoutingModeV3, hybridWeightsForRoutingModeV3, prototypeRiskPenaltyV3 } from './route-policy-v3-routing-mode.js';
const ROUTES = ['no_memory', 'capture_only', 'retrieve_memory', 'retrieve_and_distill', 'high_confidence_correction_only'];
const RETRIEVAL_ROUTES = new Set(['retrieve_memory', 'retrieve_and_distill', 'high_confidence_correction_only']);
const MEMORY_TYPES = ['correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'recall_rule', 'outcome', 'context'];
const SYNC_MODES = ['no', 'never_unless_ambiguous', 'allowed', 'prefer'];
const EMBED_DIM = 24;
function routePolicyV3UpdateMode(config) {
    const mode = String(config.routeLearning?.policyV3?.updateMode || 'gated_active');
    return mode === 'collect_only' || mode === 'distill_shadow' || mode === 'manual_review_required'
        ? mode
        : 'gated_active';
}
function activationCooldownActiveV3(existing, config) {
    const cooldownMs = Math.max(0, Number(config.routeLearning?.policyV3?.activationCooldownMs ?? 0));
    if (!existing || cooldownMs <= 0 || !existing.createdAt)
        return false;
    const ageMs = Date.now() - Date.parse(existing.createdAt);
    return Number.isFinite(ageMs) && ageMs >= 0 && ageMs < cooldownMs;
}
function chooseSnapshotStatusV3(validation, existing, config, updateMode) {
    if (!validation.ok)
        return 'rejected';
    if (updateMode === 'distill_shadow' || updateMode === 'manual_review_required')
        return 'shadow';
    if (activationCooldownActiveV3(existing, config))
        return 'shadow';
    return config.routeLearning?.policyV3?.shadowBeforeActivate === true ? 'shadow' : 'active';
}
function chooseActivationReasonV3(validation, existing, config, updateMode) {
    if (!validation.ok)
        return validation.reason;
    if (updateMode === 'distill_shadow')
        return 'update_mode_distill_shadow';
    if (updateMode === 'manual_review_required')
        return 'manual_review_required';
    if (activationCooldownActiveV3(existing, config))
        return 'activation_cooldown_active';
    if (config.routeLearning?.policyV3?.shadowBeforeActivate === true)
        return 'shadow_before_activate';
    return validation.reason;
}
export function ingestRouteLearningArtifactsV3(store, agentId, decision, routeFrame, teacherRun, counterfactuals, lessons, config) {
    const chosenPrototype = upsertPrototypeForDecision(store, agentId, decision, lessons, 'learned', config);
    const rewardComponents = rewardComponentsForDecision(decision);
    const frame = store.insertRouteFrameV3({
        agentId,
        routeDecisionId: decision.id,
        routeFrameId: decision.routeFrameId,
        redactedTurnSummary: redactText(routeFrame?.redactedTurnSummary || decision.turnFrame.summary || '', 500),
        taskType: sanitizeTaskType(routeFrame?.taskType || decision.turnFrame.taskType),
        turnSignals: stableSignals(routeFrame?.turnSignals || signalsFromDecision(decision)),
        projectHint: routeFrame?.projectHint,
        repoHint: routeFrame?.repoHint,
        toolHints: stableSignals(decision.turnFrame.activeObjects.filter((object) => object.kind === 'tool').map((object) => object.value)),
        routeHintFlags: routeHintFlags(decision.turnFrame),
        chosenActionId: chosenPrototype.id,
        chosenRoute: chosenPrototype.route,
        chosenMemoryTypes: chosenPrototype.memoryTypes,
        chosenGraphDepth: chosenPrototype.graphDepth,
        chosenSyncPlanner: chosenPrototype.syncPlanner,
        policySnapshotId: decision.policySnapshotId,
        policyRuleId: decision.policyRuleId,
        routingMode: decision.routingMode,
        rawPolicyScore: decision.rawPolicyScore,
        calibratedPolicyScore: decision.calibratedPolicyScore,
        policyThreshold: decision.policyThreshold,
        abstained: decision.abstained,
        fallbackSource: decision.fallbackSource,
        outcome: decision.outcome,
        reward: Number(decision.reward || 0),
        rewardComponents,
        payloadHash: hashText(JSON.stringify({
            summary: routeFrame?.redactedTurnSummary || decision.turnFrame.summary || '',
            taskType: decision.turnFrame.taskType,
            route: decision.route,
            memoryTypes: decision.retrievalPlan?.memoryTypes || [],
            queries: decision.retrievalPlan?.queries || [],
            reward: decision.reward,
            outcome: decision.outcome,
        })),
    });
    const banditFeedback = store.insertRouteBanditFeedbackV3({
        agentId,
        frameId: frame.id,
        chosenActionId: chosenPrototype.id,
        reward: Number(decision.reward || 0),
        rewardComponents,
        cost: decision.syncLlmUsed ? 1 : 0,
        latencyMs: Number(decision.syncLatencyMs || 0),
        outcomeLabel: outcomeLabelForDecision(decision),
        learningBucket: Math.abs(Number(decision.reward || 0)) >= Number(config.routeLearning?.teacher?.minResolvedRewardMagnitude ?? 0),
    });
    store.upsertRouteBanditStateV3(updateBanditState(store.getRouteBanditStateV3(agentId), banditFeedback, config));
    const prototypeIds = new Set([chosenPrototype.id]);
    const pairExamples = [];
    const teacherPrototype = upsertPrototypeForTeacher(store, agentId, teacherRun, lessons, 'distilled', config);
    prototypeIds.add(teacherPrototype.id);
    if (teacherPrototype.id !== chosenPrototype.id) {
        const pair = store.insertRoutePairExampleV3({
            agentId,
            frameId: frame.id,
            positiveActionId: teacherPrototype.id,
            negativeActionId: chosenPrototype.id,
            labelSource: 'teacher',
            marginWeight: learningMarginWeightV3(Number(teacherRun.confidence || 0.5), decision),
            evidenceIds: unique([decision.id, teacherRun.id]),
        });
        pairExamples.push(pair);
    }
    for (const cf of counterfactuals) {
        if (cf.estimatedOutcome !== 'likely_helpful' && cf.estimatedOutcome !== 'likely_missed')
            continue;
        if (Number(cf.confidence || 0) < 0.35)
            continue;
        const prototype = upsertPrototypeForCounterfactual(store, agentId, decision, cf, 'distilled', config);
        prototypeIds.add(prototype.id);
        if (prototype.id === chosenPrototype.id)
            continue;
        const counterfactualBeatsChosen = cf.estimatedOutcome === 'likely_helpful';
        pairExamples.push(store.insertRoutePairExampleV3({
            agentId,
            frameId: frame.id,
            positiveActionId: counterfactualBeatsChosen ? prototype.id : chosenPrototype.id,
            negativeActionId: counterfactualBeatsChosen ? chosenPrototype.id : prototype.id,
            labelSource: 'counterfactual',
            marginWeight: learningMarginWeightV3(Number(cf.confidence || teacherRun.confidence || 0.5), decision),
            evidenceIds: unique([decision.id, teacherRun.id, cf.id]),
        }));
    }
    if ((decision.reward || 0) < 0 && chosenPrototype.route !== 'no_memory') {
        const silencePrototype = upsertPrototype(store, {
            agentId,
            route: 'no_memory',
            memoryTypes: [],
            graphDepth: 0,
            syncPlanner: 'no',
            queryTemplateFamily: [],
            sparseSignature: ['silence', sanitizeTaskType(decision.turnFrame.taskType)],
            denseEmbedding: embedTextParts([decision.turnFrame.summary || '', 'no_memory', decision.turnFrame.taskType]),
            supportPrior: 1,
            harmPrior: 0,
            status: 'active',
            provenance: 'distilled',
            sourceExampleIds: [decision.id],
        }, config);
        prototypeIds.add(silencePrototype.id);
        pairExamples.push(store.insertRoutePairExampleV3({
            agentId,
            frameId: frame.id,
            positiveActionId: silencePrototype.id,
            negativeActionId: chosenPrototype.id,
            labelSource: 'outcome',
            marginWeight: learningMarginWeightV3(Math.abs(Number(decision.reward || 0)), decision),
            evidenceIds: unique([decision.id]),
        }));
    }
    return {
        frameId: frame.id,
        chosenActionId: chosenPrototype.id,
        prototypeIds: [...prototypeIds],
        pairExamples: pairExamples.length,
        banditFeedbackId: banditFeedback.id,
    };
}
export function maybeDistillAndStorePolicyV3(store, agentId, config) {
    if (config.routeLearning?.policyV3?.enabled === false) {
        return { framesConsidered: 0, pairExamplesConsidered: 0, prototypesConsidered: 0, rulesGenerated: 0 };
    }
    const updateMode = routePolicyV3UpdateMode(config);
    const frames = store.listRouteFramesV3(agentId, 400);
    const pairs = store.listRoutePairExamplesV3(agentId, 1200);
    const listedPrototypes = promoteEligibleColdStartPrototypesV3(store, store.listRouteActionPrototypesV3(agentId, 200), config);
    const retirement = retireStalePrototypesV3(store, listedPrototypes, config);
    const prototypes = listedPrototypes.filter((prototype) => prototype.status !== 'retired' && prototype.status !== 'cold_start' && !retirement.retiredPrototypeIds.includes(prototype.id));
    const banditState = store.getRouteBanditStateV3(agentId);
    const minFrames = Number(config.routeLearning?.policyV3?.minFrames ?? 3);
    const existing = store.getActivePolicySnapshotV3(agentId);
    if (frames.length < minFrames || prototypes.length === 0) {
        return {
            snapshot: existing ?? undefined,
            framesConsidered: frames.length,
            pairExamplesConsidered: pairs.length,
            prototypesConsidered: prototypes.length,
            rulesGenerated: existing?.rules.length ?? 0,
        };
    }
    if (updateMode === 'collect_only') {
        return {
            snapshot: existing ?? undefined,
            framesConsidered: frames.length,
            pairExamplesConsidered: pairs.length,
            prototypesConsidered: prototypes.length,
            rulesGenerated: 0,
        };
    }
    const rules = distillPolicyRulesV3(frames, pairs, prototypes, banditState, config);
    if (rules.length === 0) {
        const rejected = store.insertPolicySnapshotV3(buildSnapshotV3(agentId, [], {}, [], [], config, 'rejected', {
            frames: frames.length,
            pairExamples: pairs.length,
            prototypes: prototypes.length,
            projectedSyncPlannerRate: 0,
            noisyActionRate: 1,
            harmRate: 1,
            activationDecision: 'rejected',
            activationStatusReason: 'no_valid_rules',
            validationErrors: ['no_valid_rules'],
            compactness: compactnessSummaryV3([], [], [], 0, 0),
        }, undefined, { previousSnapshotId: existing?.id, comparedAgainstSnapshotId: existing?.id, retiredPrototypeIds: retirement.retiredPrototypeIds }));
        return {
            snapshot: rejected,
            validation: validatePolicySnapshotV3(rejected, config, existing),
            framesConsidered: frames.length,
            pairExamplesConsidered: pairs.length,
            prototypesConsidered: prototypes.length,
            rulesGenerated: 0,
        };
    }
    const priors = buildActionPriors(frames, pairs, prototypes, banditState);
    const preCalibration = buildSnapshotV3(agentId, rules, priors, frames, prototypes, config, 'candidate', undefined, undefined, {
        previousSnapshotId: existing?.id,
        comparedAgainstSnapshotId: existing?.id,
        retiredPrototypeIds: retirement.retiredPrototypeIds,
    });
    const calibrationArtifacts = buildSnapshotCalibrationArtifactsV3(preCalibration, frames, config);
    const calibration = calibrationArtifacts.summary;
    const replay = buildSnapshotReplaySummaryV3({ ...preCalibration, calibration }, frames, existing, config);
    const evalCaseArtifacts = buildSnapshotEvalCaseArtifactsV3({ ...preCalibration, calibration }, frames, config);
    const draft = buildSnapshotV3(agentId, rules, priors, frames, prototypes, config, 'candidate', {
        ...(preCalibration.evalSummary ?? {}),
        replay,
    }, calibration, {
        previousSnapshotId: existing?.id,
        comparedAgainstSnapshotId: existing?.id,
        retiredPrototypeIds: retirement.retiredPrototypeIds,
    });
    const validation = validatePolicySnapshotV3(draft, config, existing);
    const identical = existing && stablePolicyBodyV3(existing) === stablePolicyBodyV3(draft);
    if (identical) {
        return {
            snapshot: existing,
            validation,
            framesConsidered: frames.length,
            pairExamplesConsidered: pairs.length,
            prototypesConsidered: prototypes.length,
            rulesGenerated: rules.length,
        };
    }
    const activationReason = chooseActivationReasonV3(validation, existing, config, updateMode);
    const finalStatus = chooseSnapshotStatusV3(validation, existing, config, updateMode);
    const shadowDecisions = store.listRouteShadowDecisionsV3?.(agentId, 2000) || [];
    const rollbackRecommendation = buildRollbackRecommendationV3(existing, shadowDecisions);
    const stored = store.insertPolicySnapshotV3({
        ...draft,
        status: finalStatus,
        evalSummary: {
            ...(draft.evalSummary ?? {}),
            activationDecision: finalStatus,
            activationSummary: {
                mode: updateMode,
                status: finalStatus,
                reason: activationReason,
            },
            activationStatusReason: activationReason,
            rollbackRecommendation,
            validationErrors: validation.errors,
            validationWarnings: validation.warnings,
            projectedSyncPlannerRate: validation.projectedSyncPlannerRate,
            noisyActionRate: validation.noisyActionRate,
            harmRate: validation.harmRate,
        },
    });
    store.replaceRouteCalibrationExamplesV3?.(agentId, stored.id, calibrationArtifacts.examples.map((example) => ({ ...example, split: 'holdout' })));
    store.replaceRouteEvalCasesV3?.(agentId, stored.id, evalCaseArtifacts.cases.map((item) => ({ ...item, split: 'replay_eval' })));
    for (const stats of buildActionFamilyStatsV3(agentId, frames, pairs, prototypes, banditState, shadowDecisions)) {
        store.upsertRouteActionFamilyStatsV3?.(stats);
    }
    store.insertRoutePolicyCandidateReportV3?.(buildPolicyCandidateReportV3(stored, draft, existing, shadowDecisions, retirement.retiredPrototypeIds, activationReason));
    return {
        snapshot: stored,
        validation: { ...validation, status: finalStatus },
        framesConsidered: frames.length,
        pairExamplesConsidered: pairs.length,
        prototypesConsidered: prototypes.length,
        rulesGenerated: rules.length,
    };
}
export function distillPolicyRulesV3(frames, pairs, prototypes, banditState, config) {
    const pairCounts = summarizePairCounts(pairs);
    const rawRules = [];
    const minConfidence = Number(config.routeLearning?.policyV3?.minRuleConfidence ?? 0.55);
    const globalTaskTypes = topCounts(frames.map((frame) => frame.taskType), 2);
    const globalSignals = topCounts(frames.flatMap((frame) => frame.turnSignals), 12);
    const globalProjects = topCounts(frames.map((frame) => frame.projectHint || '').filter(Boolean), 2);
    const globalRepoPresent = frames.some((frame) => Boolean(frame.repoHint));
    for (const prototype of prototypes) {
        const matchingFrames = frames.filter((frame) => frame.chosenActionId === prototype.id);
        const taskTypes = matchingFrames.length ? topCounts(matchingFrames.map((frame) => frame.taskType), 1) : globalTaskTypes;
        const signals = matchingFrames.length ? topCounts(matchingFrames.flatMap((frame) => frame.turnSignals), 8) : globalSignals;
        const projects = matchingFrames.length ? topCounts(matchingFrames.map((frame) => frame.projectHint || '').filter(Boolean), 1) : globalProjects;
        const priors = priorsForPrototype(prototype.id, matchingFrames, pairCounts, banditState);
        const confidence = clamp01(0.42 +
            Math.min(0.18, priors.support * 0.03) -
            Math.min(0.2, priors.harm * 0.05) +
            priors.pairWinRate * 0.18 +
            Math.max(-0.1, Math.min(0.15, priors.banditMeanReward * 0.18)) -
            Math.min(0.08, Number(priors.ambiguityPenaltyMean || 0) * 0.12));
        const isNoMemoryRule = prototype.route === 'no_memory';
        const effectiveMinConfidence = isNoMemoryRule ? 0.18 : minConfidence;
        if (confidence < effectiveMinConfidence)
            continue;
        const family = ruleFamilyV3(prototype.route, prototype.memoryTypes, prototype.syncPlanner);
        const repoHintPresent = (matchingFrames.length ? matchingFrames.some((frame) => Boolean(frame.repoHint)) : globalRepoPresent) || undefined;
        const rule = {
            id: hashText(`${prototype.id}:${taskTypes.join('|')}:${signals.join('|')}:${priors.support}:${priors.harm}`).slice(7, 25),
            priority: isNoMemoryRule ? 95 : 55,
            actionId: prototype.id,
            family,
            match: {
                taskType: taskTypes[0] || undefined,
                turnSignals: stableSignals(signals).slice(0, Math.max(4, Number(config.routeLearning?.policyV3?.maxRuleSignals ?? 8))),
                projectHint: projects[0] || undefined,
                repoHintPresent,
                safetySignalsAbsent: isNoMemoryRule ? ['unsafe'] : undefined,
            },
            route: prototype.route,
            memoryTypes: prototype.memoryTypes,
            queries: normalizeQueryTemplateFamilyV3(prototype.queryTemplateFamily, 8),
            graphDepth: prototype.graphDepth,
            syncPlanner: prototype.syncPlanner,
            confidence,
            rawConfidence: confidence,
            matchSpecificityScore: specificityScoreV3({
                taskType: taskTypes[0] || undefined,
                signals,
                projectHint: projects[0] || undefined,
                repoHintPresent,
                graphDepth: prototype.graphDepth,
                syncPlanner: prototype.syncPlanner,
            }),
            evidenceIds: unique([...prototype.sourceExampleIds, ...matchingFrames.map((frame) => frame.routeDecisionId)]).slice(0, 40),
            priors,
            riskFlags: riskFlagsForRuleV3(prototype.route, prototype.memoryTypes, prototype.graphDepth, prototype.syncPlanner),
            diagnosticNotes: [`family:${family}`, `prototype:${prototype.id}`],
            reason: `distilled_from_prototype:${prototype.id}`,
        };
        rule.canonicalActionKey = canonicalActionKeyV3(rule);
        rule.dominanceGroupKey = dominanceGroupKeyV3(rule);
        if (validateRuleShapeV3(rule, config).length === 0)
            rawRules.push(rule);
    }
    const merged = mergeRuleCandidatesV3(rawRules, config);
    const pruned = pruneDominatedRulesV3(merged.rules);
    const maxRulesPerRoute = Math.max(1, Number(config.routeLearning?.policyV3?.maxRulesPerRoute ?? 10));
    const capped = [];
    const byRoute = new Map();
    for (const rule of pruned.rules.sort((a, b) => Number(b.priority || 0) - Number(a.priority || 0) || b.confidence - a.confidence)) {
        const count = byRoute.get(rule.route) || 0;
        if (count >= maxRulesPerRoute)
            continue;
        capped.push(rule);
        byRoute.set(rule.route, count + 1);
    }
    const limited = capped.slice(0, Math.max(4, Number(config.routeLearning?.policyV3?.maxRules ?? 32)));
    const compactness = compactnessSummaryV3(rawRules, merged.rules, limited, merged.duplicateGroups, pruned.prunedRuleIds.length);
    return limited.map((rule) => ({
        ...rule,
        canonicalActionKey: rule.canonicalActionKey || canonicalActionKeyV3(rule),
        dominanceGroupKey: rule.dominanceGroupKey || dominanceGroupKeyV3(rule),
        reason: compactness.mergedAway > 0 || compactness.dominatedPruned > 0
            ? `${rule.reason || 'distilled'}|compact:${compactness.afterPrune}/${compactness.beforeMerge}`
            : rule.reason,
    }));
}
export function validatePolicySnapshotV3(snapshot, config = {}, existing) {
    const errors = [];
    const warnings = [];
    if (snapshot.version !== 'route-policy-v3')
        errors.push('unsupported_policy_version');
    const rules = Array.isArray(snapshot.rules) ? snapshot.rules : [];
    if (rules.length === 0)
        errors.push('no_rules');
    for (const rule of rules)
        errors.push(...validateRuleShapeV3(rule, config));
    const syncRules = rules.filter((rule) => rule.syncPlanner === 'allowed' || rule.syncPlanner === 'prefer').length;
    const projectedSyncPlannerRate = rules.length ? syncRules / rules.length : 0;
    const maxSyncPlannerRate = Number(snapshot.globalBudgets?.maxSyncPlannerRate ?? config.routeLearning?.policyV3?.maxSyncPlannerRate ?? 0.1);
    if (projectedSyncPlannerRate > maxSyncPlannerRate)
        errors.push(`sync_planner_rate_exceeds_budget:${projectedSyncPlannerRate.toFixed(3)}>${maxSyncPlannerRate}`);
    const evalSummary = snapshot.evalSummary ?? {};
    const noisyActionRate = Number(evalSummary.noisyActionRate ?? 0);
    const harmRate = Number(evalSummary.harmRate ?? 0);
    const maxHarmRate = Number(config.routeLearning?.policyV3?.maxHarmRate ?? 0.2);
    if (harmRate > maxHarmRate)
        errors.push(`harm_rate_exceeds_gate:${harmRate.toFixed(3)}>${maxHarmRate}`);
    if (noisyActionRate > Math.min(0.5, maxHarmRate + 0.15))
        errors.push(`noisy_action_rate_exceeds_gate:${noisyActionRate.toFixed(3)}`);
    const compactness = evalSummary.compactness ?? {};
    const duplicateRate = Number(compactness.beforeMerge || 0) > 0
        ? Number(compactness.mergedAway || 0) / Number(compactness.beforeMerge || 1)
        : 0;
    const maxDuplicateRate = Number(config.routeLearning?.policyV3?.compactnessMaxDuplicateRate ?? 0.35);
    if (duplicateRate > maxDuplicateRate)
        warnings.push(`duplicate_rate_high:${duplicateRate.toFixed(3)}>${maxDuplicateRate}`);
    if (Number(compactness.maxRulesPerRoute || 0) > Number(config.routeLearning?.policyV3?.maxRulesPerRoute ?? 10)) {
        errors.push(`max_rules_per_route_exceeded:${compactness.maxRulesPerRoute}`);
    }
    const replay = evalSummary.replay ?? {};
    const projectedImprovement = Number(replay.estimatedImprovement ?? 0);
    const minProjectedImprovement = Number(config.routeLearning?.policyV3?.minProjectedImprovement ?? -0.01);
    if (Number(replay.frames || 0) > 0 && projectedImprovement < minProjectedImprovement) {
        errors.push(`candidate_projected_value_regressed:${projectedImprovement.toFixed(3)}<${minProjectedImprovement}`);
    }
    const calibration = snapshot.calibration;
    const minCalibratedConfidence = Number(snapshot.globalBudgets?.minCalibratedConfidence ?? config.routeLearning?.policyV3?.minCalibratedConfidence ?? 0.62);
    if (calibration && Number(calibration.globalThreshold || 0) + Number(calibration.abstainMargin || 0) < minCalibratedConfidence - 0.12) {
        warnings.push('calibration_threshold_suspiciously_low');
    }
    if (existing?.evalSummary) {
        const oldHarmRate = Number(existing.evalSummary.harmRate ?? 0);
        if (harmRate > oldHarmRate + 0.05)
            errors.push('candidate_harm_rate_regressed');
    }
    const rollback = evalSummary.rollbackRecommendation ?? {};
    if (rollback.shouldRollback === true)
        warnings.push(`rollback_recommended:${rollback.reason || 'shadow_disagreement'}`);
    const reason = errors.length === 0 ? 'passed_activation_gates' : errors[0];
    return { ok: errors.length === 0, status: errors.length === 0 ? 'active' : 'rejected', reason, errors, warnings, projectedSyncPlannerRate, noisyActionRate, harmRate };
}
export function scorePolicySnapshotV3(snapshot, turnFrame, message = '', options = {}) {
    const requireActive = options.requireActive !== false;
    if (!snapshot || snapshot.version !== 'route-policy-v3' || (requireActive && snapshot.status !== 'active') || !Array.isArray(snapshot.rules)) {
        return { matched: false, score: 0, reasonCode: 'no_active_policy_v3' };
    }
    const haystack = `${message} ${turnFrame.summary} ${turnFrame.userGoal} ${turnFrame.impliedNeeds.join(' ')} ${turnFrame.memoryQuestions.join(' ')}`.toLowerCase();
    const mode = detectRoutingModeV3(turnFrame, message);
    let best = { matched: false, score: 0, reasonCode: 'no_matching_policy_rule_v3', rawScore: 0, calibratedScore: 0, threshold: 0, abstained: false };
    for (const rule of snapshot.rules) {
        const match = scoreRuleV3(rule, turnFrame, haystack, snapshot.actionPriors?.[rule.actionId], mode);
        if (!match.matched)
            continue;
        if (match.score > best.score)
            best = match;
    }
    if (!best.matched || !best.rule)
        return best;
    const calibrated = applyCalibrationV3(best.rawScore ?? best.score, best.rule.route, snapshot.calibration, mode);
    const threshold = Math.max(calibrated.threshold, familyThresholdFloorV3(best.rule, mode, snapshot));
    if (calibrated.abstained || calibrated.calibratedScore < threshold) {
        return {
            matched: false,
            rule: best.rule,
            score: calibrated.calibratedScore,
            rawScore: calibrated.rawScore,
            calibratedScore: calibrated.calibratedScore,
            threshold,
            abstained: true,
            reasonCode: `policy_v3_abstain:${best.rule.id}:${mode}`,
        };
    }
    return {
        ...best,
        score: calibrated.calibratedScore,
        calibratedScore: calibrated.calibratedScore,
        rawScore: calibrated.rawScore,
        threshold,
        abstained: false,
        reasonCode: `${best.reasonCode}:${mode}`,
    };
}
export function rankActionPrototypesV3(frame, prototypes, banditState) {
    const mode = detectRoutingModeV3(frame, frame.redactedTurnSummary || '');
    const weights = hybridWeightsForRoutingModeV3(mode);
    const queryEmbedding = embedTextParts([
        frame.redactedTurnSummary,
        frame.taskType,
        ...(frame.turnSignals || []),
        ...(frame.toolHints || []),
        ...(frame.routeHintFlags || []),
        frame.projectHint || '',
        frame.repoHint || '',
    ]);
    return prototypes.map((prototype) => {
        const sparse = sparseMatch(frame, prototype);
        const dense = bilinearScore(queryEmbedding, prototype.denseEmbedding || []);
        const stats = banditState?.actionStats?.[prototype.id];
        const bonus = stats ? (stats.rewardMean + Number((banditState?.explorationAlpha ?? 0.35)) / Math.sqrt(Math.max(1, stats.count))) : 0;
        const riskPenalty = prototypeRiskPenaltyV3(prototype, mode);
        const total = sparse * weights.sparse +
            dense * weights.dense +
            Math.max(-0.15, Math.min(0.15, bonus * weights.bandit)) -
            riskPenalty * weights.risk;
        return { prototype, score: total, sparse, dense, bonus, riskPenalty, mode };
    }).sort((a, b) => b.score - a.score);
}
function scoreRuleV3(rule, turnFrame, haystack, priors, mode = detectRoutingModeV3(turnFrame, haystack)) {
    const match = rule.match ?? {};
    if (match.taskType && match.taskType !== turnFrame.taskType)
        return { matched: false, rule, score: 0, reasonCode: 'task_type_mismatch' };
    const signals = Array.isArray(match.turnSignals) ? match.turnSignals.map((value) => String(value).toLowerCase()).filter(Boolean) : [];
    const overlap = signals.filter((signal) => haystack.includes(signal)).length;
    if (signals.length > 0 && overlap === 0)
        return { matched: false, rule, score: 0, reasonCode: 'turn_signal_mismatch' };
    const weights = hybridWeightsForRoutingModeV3(mode);
    const priority = Number(rule.priority || 0) / 200;
    const signalBonus = signals.length ? Math.min(0.18, overlap * weights.signalBonus) : 0;
    const routeHintBonus = routeHintCompatibility(rule, turnFrame);
    const banditBonus = priors ? Math.max(-0.08, Math.min(0.12, Number(priors.banditMeanReward || 0) * 0.12 + Number(priors.pairWinRate || 0) * 0.08)) : 0;
    const supportBonus = priors ? Math.min(0.08, Number(priors.support || 0) * 0.01) : 0;
    const ambiguityPenalty = priors ? Math.min(0.08, Number(priors.ambiguityPenaltyMean || 0) * 0.12) : 0;
    const rawScore = clamp01(Number(rule.confidence || 0) + priority + signalBonus + routeHintBonus + banditBonus + supportBonus - ambiguityPenalty);
    return { matched: true, rule, score: rawScore, rawScore, calibratedScore: rawScore, abstained: false, reasonCode: `policy_v3_rule:${rule.id}` };
}
function routeHintCompatibility(rule, turnFrame) {
    if (!RETRIEVAL_ROUTES.has(rule.route))
        return 0;
    let bonus = 0;
    if (rule.memoryTypes.includes('correction') && turnFrame.routeHints.likelyNeedsCorrections)
        bonus += 0.03;
    if (rule.memoryTypes.includes('preference') && turnFrame.routeHints.likelyNeedsPreferences)
        bonus += 0.03;
    if (rule.memoryTypes.includes('workflow') && turnFrame.routeHints.likelyNeedsWorkflow)
        bonus += 0.03;
    if ((rule.memoryTypes.includes('project_fact') || rule.memoryTypes.includes('context')) && turnFrame.routeHints.likelyNeedsProjectContext)
        bonus += 0.03;
    return bonus;
}
function ruleFamilyV3(route, memoryTypes, syncPlanner) {
    if (route === 'no_memory')
        return 'silence';
    if (syncPlanner === 'allowed' || syncPlanner === 'prefer')
        return 'sync_enabling';
    if ((memoryTypes || []).includes('correction'))
        return 'correction';
    if ((memoryTypes || []).includes('workflow'))
        return 'workflow';
    if ((memoryTypes || []).some((type) => type === 'project_fact' || type === 'context'))
        return 'project_context';
    return 'general_retrieval';
}
function specificityScoreV3(input) {
    return clamp01((input.taskType ? 0.25 : 0) +
        Math.min(0.35, (input.signals?.length || 0) * 0.06) +
        (input.projectHint ? 0.12 : 0) +
        (input.repoHintPresent ? 0.08 : 0) +
        Math.min(0.12, Number(input.graphDepth || 0) * 0.06) +
        ((input.syncPlanner === 'allowed' || input.syncPlanner === 'prefer') ? 0.08 : 0));
}
function dominanceGroupKeyV3(rule) {
    return [
        ruleFamilyV3(rule.route, rule.memoryTypes, rule.syncPlanner),
        rule.route,
        [...new Set(rule.memoryTypes || [])].sort().join(','),
        Number(rule.graphDepth || 0),
        String(rule.syncPlanner || 'no'),
        String(rule.match?.taskType || ''),
        String(rule.match?.projectHint || ''),
        Boolean(rule.match?.repoHintPresent) ? 'repo' : 'norepo',
    ].join('::');
}
function riskFlagsForRuleV3(route, memoryTypes, graphDepth, syncPlanner) {
    const flags = [];
    if (route !== 'no_memory' && memoryTypes.length === 0)
        flags.push('empty_memory_type_retrieval');
    if (graphDepth > 0)
        flags.push('graph_expansion');
    if (syncPlanner === 'allowed' || syncPlanner === 'prefer')
        flags.push('sync_cost');
    if (route !== 'no_memory' && !memoryTypes.includes('workflow') && !memoryTypes.includes('correction'))
        flags.push('broad_semantic_retrieval');
    return flags;
}
function familyThresholdFloorV3(rule, mode, snapshot) {
    const base = Number(snapshot.globalBudgets?.minCalibratedConfidence ?? 0.62);
    const family = String(rule.family || 'general_retrieval');
    const familyFloor = family === 'silence'
        ? Math.max(0.45, base - 0.1)
        : family === 'workflow'
            ? Math.max(0.58, base)
            : family === 'project_context'
                ? Math.max(0.62, base + 0.02)
                : family === 'correction'
                    ? Math.max(0.72, base + 0.08)
                    : family === 'sync_enabling'
                        ? Math.max(0.76, base + 0.1)
                        : Math.max(0.66, base + 0.04);
    const modeAdjustment = mode === 'ambiguous_general' ? 0.03 : mode === 'exact_correction' ? -0.01 : 0;
    const graphAdjustment = Number(rule.graphDepth || 0) > 0 ? 0.02 : 0;
    return clamp01(familyFloor + modeAdjustment + graphAdjustment);
}
function validateRuleShapeV3(rule, config) {
    const errors = [];
    if (!rule || typeof rule !== 'object')
        return ['rule_not_object'];
    if (!rule.id)
        errors.push('rule_missing_id');
    if (!rule.actionId)
        errors.push('rule_missing_action_id');
    if (!ROUTES.includes(rule.route))
        errors.push(`unsupported_route:${rule.route}`);
    if (typeof rule.confidence !== 'number' || rule.confidence < 0 || rule.confidence > 1)
        errors.push(`bad_rule_confidence:${rule.id}`);
    const graphDepth = Number(rule.graphDepth ?? 0);
    const maxGraphDepth = Number(config.routeLearning?.counterfactuals?.maxGraphDepth ?? 2);
    if (graphDepth < 0 || graphDepth > maxGraphDepth || graphDepth > 2)
        errors.push(`graph_depth_out_of_bounds:${rule.id}`);
    for (const type of rule.memoryTypes ?? []) {
        if (!MEMORY_TYPES.includes(type))
            errors.push(`unknown_memory_type:${type}`);
    }
    const signals = Array.isArray(rule.match?.turnSignals) ? rule.match.turnSignals : [];
    if (rule.route !== 'no_memory' && !rule.match?.taskType && signals.length === 0)
        errors.push(`broad_retrieval_rule:${rule.id}`);
    if (RETRIEVAL_ROUTES.has(rule.route) && (!Array.isArray(rule.memoryTypes) || rule.memoryTypes.length === 0) && (!Array.isArray(rule.queries) || rule.queries.length === 0)) {
        errors.push(`retrieval_rule_without_types_or_queries:${rule.id}`);
    }
    if (!SYNC_MODES.includes(rule.syncPlanner))
        errors.push(`bad_sync_planner:${rule.id}`);
    return errors;
}
function upsertPrototypeForDecision(store, agentId, decision, lessons, provenance, config) {
    const lesson = lessons.find((item) => item.route === decision.route) || lessons[0];
    return upsertPrototype(store, {
        agentId,
        route: sanitizeRoute(decision.route),
        memoryTypes: sanitizeMemoryTypes(decision.retrievalPlan?.memoryTypes?.length ? decision.retrievalPlan.memoryTypes : lesson?.memoryTypes ?? []),
        graphDepth: clampGraphDepth(decision.retrievalPlan?.graphDepth ?? lesson?.graphDepth ?? 0),
        syncPlanner: decision.syncLlmUsed ? 'allowed' : 'no',
        queryTemplateFamily: normalizeQueryTemplateFamilyV3(decision.retrievalPlan?.queries?.length ? decision.retrievalPlan.queries : lesson?.queryTemplates ?? [], 8),
        sparseSignature: stableSignals([
            sanitizeTaskType(decision.turnFrame.taskType),
            ...signalsFromDecision(decision),
            ...routeHintFlags(decision.turnFrame),
            ...sanitizeMemoryTypes(decision.retrievalPlan?.memoryTypes ?? []),
        ]),
        denseEmbedding: embedTextParts([
            decision.turnFrame.summary || '',
            decision.route,
            decision.turnFrame.taskType,
            ...(decision.retrievalPlan?.queries || []),
            ...sanitizeMemoryTypes(decision.retrievalPlan?.memoryTypes ?? []),
        ]),
        supportPrior: Math.max(0, Number(decision.reward || 0)),
        harmPrior: Math.max(0, -Number(decision.reward || 0)),
        status: 'active',
        provenance,
        sourceExampleIds: unique([decision.id, ...(lesson ? [lesson.id] : [])]),
    }, config);
}
function upsertPrototypeForTeacher(store, agentId, teacherRun, lessons, provenance, config) {
    const memoryTypes = sanitizeMemoryTypes(lessons.flatMap((lesson) => lesson.route === teacherRun.teacherRoute ? lesson.memoryTypes : []).length
        ? lessons.flatMap((lesson) => lesson.route === teacherRun.teacherRoute ? lesson.memoryTypes : [])
        : []);
    return upsertPrototype(store, {
        agentId,
        route: sanitizeRoute(teacherRun.teacherRoute),
        memoryTypes,
        graphDepth: clampGraphDepth(teacherRun.teacherGraphDepth ?? 0),
        syncPlanner: teacherRun.syncPlannerWorthIt ? 'allowed' : 'no',
        queryTemplateFamily: normalizeQueryTemplateFamilyV3(teacherRun.teacherQueries ?? [], 8),
        sparseSignature: stableSignals([
            sanitizeTaskType(lessons[0]?.taskType || 'other'),
            ...lessons.flatMap((lesson) => lesson.turnSignals || []),
            ...memoryTypes,
        ]),
        denseEmbedding: embedTextParts([
            teacherRun.teacherRoute,
            ...(teacherRun.teacherQueries || []),
            ...memoryTypes,
        ]),
        supportPrior: Math.max(0.25, Number(teacherRun.confidence || 0.5)),
        harmPrior: teacherRun.verdict === 'over_injected' ? 1 : 0,
        status: 'active',
        provenance,
        sourceExampleIds: unique([teacherRun.id, ...lessons.map((lesson) => lesson.id)]),
    }, config);
}
function upsertPrototypeForCounterfactual(store, agentId, decision, cf, provenance, config) {
    const route = cf.kind === 'stay_silent' || cf.kind === 'no_memory'
        ? 'no_memory'
        : cf.memoryTypes.includes('correction')
            ? 'high_confidence_correction_only'
            : cf.memoryIds.length > 0
                ? 'retrieve_memory'
                : 'no_memory';
    return upsertPrototype(store, {
        agentId,
        route,
        memoryTypes: sanitizeMemoryTypes(cf.memoryTypes ?? []),
        graphDepth: clampGraphDepth(cf.graphDepth ?? 0),
        syncPlanner: cf.kind === 'sync_planner' ? 'allowed' : 'no',
        queryTemplateFamily: normalizeQueryTemplateFamilyV3(decision.retrievalPlan?.queries ?? [], 8),
        sparseSignature: stableSignals([
            sanitizeTaskType(decision.turnFrame.taskType),
            String(cf.kind || ''),
            ...sanitizeMemoryTypes(cf.memoryTypes ?? []),
        ]),
        denseEmbedding: embedTextParts([
            decision.turnFrame.summary || '',
            route,
            ...(cf.memoryTypes || []),
            String(cf.kind || ''),
        ]),
        supportPrior: Math.max(0.2, Number(cf.confidence || 0.5)),
        harmPrior: cf.estimatedOutcome === 'likely_noise' || cf.estimatedOutcome === 'likely_harmful' ? 1 : 0,
        status: 'active',
        provenance,
        sourceExampleIds: unique([decision.id, cf.id]),
    }, config);
}
function upsertPrototype(store, input, config) {
    const prototypeId = input.id || prototypeIdFor(input);
    const existing = store.getRouteActionPrototypeV3?.(prototypeId);
    const support = Number(existing?.supportPrior || 0) + Number(input.supportPrior || 0);
    const harm = Number(existing?.harmPrior || 0) + Number(input.harmPrior || 0);
    const status = derivePrototypeStatusV3(input.status, input.provenance, support, harm, config);
    return store.upsertRouteActionPrototypeV3({ ...input, id: prototypeId, status });
}
function derivePrototypeStatusV3(current, provenance, support, harm, config) {
    if (current === 'retired')
        return 'retired';
    if (current === 'shadow')
        return 'shadow';
    if (provenance !== 'learned')
        return 'active';
    const minSamples = Number(config.routeLearning?.policyV3?.coldStartMinSamples ?? 3);
    return (support + harm) < minSamples ? 'cold_start' : 'active';
}
function promoteEligibleColdStartPrototypesV3(store, prototypes, config) {
    const minSamples = Number(config.routeLearning?.policyV3?.coldStartMinSamples ?? 3);
    return prototypes.map((prototype) => {
        if (prototype.status !== 'cold_start')
            return prototype;
        const sampleCount = Number(prototype.supportPrior || 0) + Number(prototype.harmPrior || 0);
        if (sampleCount < minSamples)
            return prototype;
        return store.setRouteActionPrototypeStatusV3?.(prototype.id, 'active') || { ...prototype, status: 'active' };
    });
}
function prototypeIdFor(input) {
    return hashText(JSON.stringify({
        route: input.route,
        memoryTypes: [...new Set(input.memoryTypes)].sort(),
        graphDepth: input.graphDepth,
        syncPlanner: input.syncPlanner,
        queries: [...new Set(input.queryTemplateFamily)].sort(),
        sparse: [...new Set(input.sparseSignature)].sort(),
    })).slice(7, 23);
}
function buildActionPriors(frames, pairs, prototypes, banditState) {
    const pairCounts = summarizePairCounts(pairs);
    return Object.fromEntries(prototypes.map((prototype) => [prototype.id, priorsForPrototype(prototype.id, frames.filter((frame) => frame.chosenActionId === prototype.id), pairCounts, banditState)]));
}
function actionFamilyKeyV3(input) {
    return `${input.route}::${[...new Set(input.memoryTypes || [])].sort().join(',')}::${input.graphDepth}::${input.syncPlanner}`;
}
function buildActionFamilyStatsV3(agentId, frames, pairs, prototypes, banditState, shadowDecisions) {
    const pairCounts = summarizePairCounts(pairs);
    const shadowByRoute = shadowDecisions.reduce((acc, decision) => {
        acc[decision.proposedRoute] ||= [];
        acc[decision.proposedRoute].push(decision);
        return acc;
    }, {});
    return prototypes.map((prototype) => {
        const familyKey = actionFamilyKeyV3(prototype);
        const prototypeFrames = frames.filter((frame) => frame.chosenActionId === prototype.id);
        const rewards = prototypeFrames.map((frame) => Number(frame.reward || 0));
        const meanReward = rewards.length ? rewards.reduce((sum, value) => sum + value, 0) / rewards.length : 0;
        const rewardVariance = rewards.length
            ? rewards.reduce((sum, value) => sum + Math.pow(value - meanReward, 2), 0) / rewards.length
            : 0;
        const pair = pairCounts[prototype.id] || { wins: 0, losses: 0 };
        const stats = banditState?.actionStats?.[prototype.id];
        const routeShadow = shadowByRoute[prototype.route] || [];
        const shadowAgreementRate = routeShadow.length
            ? routeShadow.filter((decision) => decision.matchedObservedRoute === true).length / routeShadow.length
            : 0;
        return {
            familyKey,
            agentId,
            route: prototype.route,
            memoryTypes: prototype.memoryTypes,
            graphDepth: prototype.graphDepth,
            syncPlanner: prototype.syncPlanner,
            supportCount: prototypeFrames.filter((frame) => Number(frame.reward || 0) >= 0).length,
            harmCount: prototypeFrames.filter((frame) => Number(frame.reward || 0) < 0).length,
            meanReward: Number(meanReward.toFixed(4)),
            rewardVariance: Number(rewardVariance.toFixed(4)),
            pairWinRate: Number(((pair.wins + pair.losses) > 0 ? pair.wins / (pair.wins + pair.losses) : 0.5).toFixed(4)),
            banditMeanReward: Number(Number(stats?.rewardMean || 0).toFixed(4)),
            banditCount: Number(stats?.count || 0),
            shadowAgreementRate: Number(shadowAgreementRate.toFixed(4)),
            updatedAt: new Date().toISOString(),
        };
    });
}
function buildPolicyCandidateReportV3(stored, draft, existing, shadowDecisions, retiredPrototypeIds, activationReason) {
    const compactness = stored.evalSummary?.compactness || draft.evalSummary?.compactness || {};
    const replay = stored.evalSummary?.replay || draft.evalSummary?.replay || {};
    return {
        agentId: stored.agentId,
        snapshotId: stored.id,
        previousSnapshotId: existing?.id,
        status: stored.status,
        bodyHash: stablePolicyBodyV3(stored),
        ruleCount: stored.rules.length,
        compactnessBefore: Number(compactness.beforeMerge || stored.rules.length),
        compactnessAfter: Number(compactness.afterPrune || stored.rules.length),
        duplicateGroups: Number(compactness.duplicateGroups || 0),
        mergedAway: Number(compactness.mergedAway || 0),
        dominatedPruned: Number(compactness.dominatedPruned || 0),
        estimatedImprovement: Number(replay.estimatedImprovement || 0),
        projectedSyncPlannerRate: Number(stored.evalSummary?.projectedSyncPlannerRate || 0),
        noisyActionRate: Number(stored.evalSummary?.noisyActionRate || 0),
        harmRate: Number(stored.evalSummary?.harmRate || 0),
        calibrationHoldoutFrames: Number(stored.calibration?.holdoutFrames || 0),
        shadowDecisionCount: shadowDecisions.filter((decision) => decision.snapshotId === stored.id).length,
        retiredPrototypeIds: [...retiredPrototypeIds],
        activationReason,
    };
}
function buildRollbackRecommendationV3(existing, shadowDecisions) {
    if (!existing) {
        return {
            shouldRollback: false,
            reason: 'no_existing_snapshot',
            shadowDisagreementRate: 0,
            shadowSampleCount: 0,
        };
    }
    const recent = [...shadowDecisions]
        .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
        .slice(0, 200);
    const comparable = recent.filter((decision) => typeof decision.matchedObservedRoute === 'boolean');
    const sampleCount = comparable.length;
    const disagreements = comparable.filter((decision) => decision.matchedObservedRoute === false).length;
    const disagreementRate = sampleCount ? disagreements / sampleCount : 0;
    const shouldRollback = sampleCount >= 12 && disagreementRate >= 0.55;
    return {
        shouldRollback,
        reason: shouldRollback ? 'shadow_disagreement_spike' : 'shadow_stable',
        shadowDisagreementRate: Number(disagreementRate.toFixed(4)),
        shadowSampleCount: sampleCount,
    };
}
function priorsForPrototype(prototypeId, frames, pairCounts, banditState) {
    const support = frames.filter((frame) => Number(frame.reward || 0) >= 0).length;
    const harm = frames.filter((frame) => Number(frame.reward || 0) < 0).length;
    const ambiguityPenaltyMean = frames.length
        ? frames.reduce((sum, frame) => sum + Number(frame.rewardComponents?.ambiguityPenalty || 0), 0) / frames.length
        : 0;
    const teacherConfidenceMean = frames.length
        ? frames.reduce((sum, frame) => sum + Number(frame.rewardComponents?.teacherConfidence || 0), 0) / frames.length
        : 0;
    const validatorConfidenceMean = frames.length
        ? frames.reduce((sum, frame) => sum + Number(frame.rewardComponents?.validatorConfidence || 0), 0) / frames.length
        : 0;
    const pair = pairCounts[prototypeId] || { wins: 0, losses: 0 };
    const pairWinRate = pair.wins + pair.losses > 0 ? pair.wins / (pair.wins + pair.losses) : 0.5;
    const stats = banditState?.actionStats?.[prototypeId];
    return {
        support,
        harm,
        banditMeanReward: Number(stats?.rewardMean || 0),
        banditCount: Number(stats?.count || 0),
        pairWinRate,
        teacherConfidenceMean: Number(teacherConfidenceMean.toFixed(4)),
        validatorConfidenceMean: Number(validatorConfidenceMean.toFixed(4)),
        ambiguityPenaltyMean: Number(ambiguityPenaltyMean.toFixed(4)),
    };
}
function summarizePairCounts(pairs) {
    const map = {};
    for (const pair of pairs) {
        map[pair.positiveActionId] ||= { wins: 0, losses: 0 };
        map[pair.negativeActionId] ||= { wins: 0, losses: 0 };
        map[pair.positiveActionId].wins += pair.marginWeight || 1;
        map[pair.negativeActionId].losses += pair.marginWeight || 1;
    }
    return map;
}
function buildSnapshotV3(agentId, rules, actionPriors, frames, prototypes, config, status, evalOverride, calibration, lineage) {
    const harms = frames.filter((frame) => Number(frame.reward || 0) < 0).length;
    const noisy = frames.filter((frame) => Number(frame.rewardComponents?.noisyInjectionPenalty || 0) > 0).length;
    const syncRules = rules.filter((rule) => rule.syncPlanner === 'allowed' || rule.syncPlanner === 'prefer').length;
    const compactness = compactnessSummaryV3(rules, rules, rules, 0, 0);
    const minCalibratedConfidence = Number(config.routeLearning?.policyV3?.minCalibratedConfidence ?? 0.62);
    const familyThresholds = Object.fromEntries([...new Set(rules.map((rule) => rule.family || ruleFamilyV3(rule.route, rule.memoryTypes, rule.syncPlanner)))].map((family) => [
        family,
        familyThresholdFloorV3({ family }, 'mixed', { globalBudgets: { minCalibratedConfidence } }),
    ]));
    return {
        agentId,
        version: 'route-policy-v3',
        status,
        rules,
        actionPriors,
        globalBudgets: {
            maxSyncPlannerRate: Number(config.routeLearning?.policyV3?.maxSyncPlannerRate ?? 0.1),
            maxInjectedMemories: Number(config.routing?.maxInjectedMemories ?? 8),
            maxInjectedChars: Number(config.routing?.maxInjectedChars ?? 2500),
            defaultGraphDepth: clampGraphDepth(config.routeLearning?.counterfactuals?.maxGraphDepth ?? 1),
            minCalibratedConfidence,
            abstainMargin: Number(config.routeLearning?.policyV3?.abstainMargin ?? 0.05),
        },
        evalSummary: evalOverride ?? {
            frames: frames.length,
            pairExamples: 0,
            prototypes: prototypes.length,
            projectedSyncPlannerRate: rules.length ? syncRules / rules.length : 0,
            noisyActionRate: frames.length ? noisy / frames.length : 0,
            harmRate: frames.length ? harms / frames.length : 0,
            activationSummary: {
                mode: routePolicyV3UpdateMode(config),
                status,
                reason: `snapshot_status:${status}`,
            },
            rollbackRecommendation: {
                shouldRollback: false,
                reason: 'none',
                shadowDisagreementRate: 0,
                shadowSampleCount: 0,
            },
            thresholds: {
                global: minCalibratedConfidence,
                byRoute: calibration?.routeThresholds,
                byFamily: familyThresholds,
            },
            compactness,
        },
        calibration,
        lineage,
        sourceFrameIds: frames.map((frame) => frame.id).slice(0, 200),
        sourcePrototypeIds: prototypes.map((prototype) => prototype.id).slice(0, 200),
        model: config.llm?.learningModel || 'deterministic-route-policy-v3',
        promptVersion: 'route-policy-v3-distiller-v1',
    };
}
function buildSnapshotCalibrationArtifactsV3(snapshot, frames, config) {
    const holdout = holdoutFramesV3(frames, config);
    const predictions = holdout.map((frame) => {
        const match = scoreSnapshotAgainstFrameV3(snapshot, frame);
        return {
            rawScore: match.rawScore,
            route: match.route,
            observedSuccess: frameSuccessLabelV3(frame),
            comparable: match.route === frame.chosenRoute,
        };
    });
    const summary = buildCalibrationSummaryV3(predictions, config);
    const examples = holdout.map((frame) => {
        const match = scoreSnapshotAgainstFrameV3(snapshot, frame);
        const mode = detectRoutingModeV3({
            taskType: frame.taskType,
            turnSignals: frame.turnSignals,
            routeHintFlags: frame.routeHintFlags,
            redactedTurnSummary: frame.redactedTurnSummary,
        }, frame.redactedTurnSummary);
        const calibrated = applyCalibrationV3(match.rawScore, match.route, summary, mode);
        return {
            frameId: frame.id,
            route: match.route,
            actionId: match.rule?.actionId,
            ruleId: match.rule?.id,
            routingMode: mode,
            rawScore: match.rawScore,
            calibratedScore: calibrated.calibratedScore,
            observedSuccess: frameSuccessLabelV3(frame),
            comparable: match.route === frame.chosenRoute,
            split: 'holdout',
        };
    });
    return { summary, examples };
}
function buildSnapshotEvalCaseArtifactsV3(snapshot, frames, config) {
    const holdout = holdoutFramesV3(frames, config);
    const cases = holdout.map((frame) => {
        const match = scoreSnapshotAgainstFrameV3(snapshot, frame);
        const mode = detectRoutingModeV3({
            taskType: frame.taskType,
            turnSignals: frame.turnSignals,
            routeHintFlags: frame.routeHintFlags,
            redactedTurnSummary: frame.redactedTurnSummary,
        }, frame.redactedTurnSummary);
        const reward = Number(frame.reward || 0);
        const success = frameSuccessLabelV3(frame);
        const quality = success && Math.abs(reward) >= 0.35
            ? 'trusted'
            : success || Math.abs(reward) >= 0.15
                ? 'usable'
                : Math.abs(reward) > 0
                    ? 'weak'
                    : 'ambiguous';
        return {
            frameId: frame.id,
            routingMode: mode,
            observedRoute: frame.chosenRoute,
            expectedRoute: success ? frame.chosenRoute : (match.route || frame.chosenRoute),
            reward,
            quality,
            humanReviewed: false,
            promotionSafe: quality === 'trusted' || quality === 'usable',
            notes: `holdout_frame:${frame.id}`,
            split: 'replay_eval',
            labels: [{
                    source: 'outcome',
                    preferredRoute: success ? frame.chosenRoute : (match.route || frame.chosenRoute),
                    confidence: success ? 0.85 : (quality === 'ambiguous' ? 0.35 : 0.55),
                    notes: success ? 'outcome_supported' : 'outcome_weak_or_negative',
                }],
        };
    });
    return { cases };
}
function buildSnapshotReplaySummaryV3(snapshot, frames, existing, config) {
    const holdout = holdoutFramesV3(frames, config);
    const predictions = holdout.map((frame) => replayPredictionForFrameV3(snapshot, frame));
    const baseline = holdout.map((frame) => replayPredictionForFrameV3(existing || null, frame));
    return summarizeReplayEvaluationV3(predictions, baseline, snapshot.calibration);
}
function holdoutFramesV3(frames, config) {
    const ordered = [...frames].sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)));
    const fraction = Number(config.routeLearning?.policyV3?.holdoutFraction ?? 0.3);
    const minHoldout = Number(config.routeLearning?.policyV3?.minHoldoutFrames ?? 2);
    const count = Math.min(ordered.length, Math.max(minHoldout, Math.round(ordered.length * fraction)));
    return ordered.slice(-count);
}
function replayPredictionForFrameV3(snapshot, frame) {
    const mode = detectRoutingModeV3({
        taskType: frame.taskType,
        turnSignals: frame.turnSignals,
        routeHintFlags: frame.routeHintFlags,
        redactedTurnSummary: frame.redactedTurnSummary,
    }, frame.redactedTurnSummary);
    if (!snapshot || snapshot.version !== 'route-policy-v3') {
        return {
            frameId: frame.id,
            route: null,
            rawScore: 0,
            calibratedScore: 0,
            abstained: true,
            comparable: false,
            matchedObservedRoute: false,
            reward: Number(frame.reward || 0),
            mode,
        };
    }
    const match = scoreSnapshotAgainstFrameV3(snapshot, frame);
    const calibrated = applyCalibrationV3(match.rawScore, match.route, snapshot.calibration, mode);
    return {
        frameId: frame.id,
        route: match.route,
        rawScore: match.rawScore,
        calibratedScore: calibrated.calibratedScore,
        abstained: calibrated.abstained,
        comparable: match.route === frame.chosenRoute,
        matchedObservedRoute: match.route === frame.chosenRoute,
        reward: Number(frame.reward || 0),
        mode,
    };
}
function scoreSnapshotAgainstFrameV3(snapshot, frame) {
    const turnFrame = turnFrameFromStoredFrameV3(frame);
    const haystack = `${frame.redactedTurnSummary} ${frame.turnSignals.join(' ')} ${frame.toolHints.join(' ')}`.toLowerCase();
    const mode = detectRoutingModeV3({
        taskType: frame.taskType,
        turnSignals: frame.turnSignals,
        routeHintFlags: frame.routeHintFlags,
        redactedTurnSummary: frame.redactedTurnSummary,
    }, frame.redactedTurnSummary);
    let bestRule;
    let bestScore = 0;
    for (const rule of snapshot.rules || []) {
        const match = scoreRuleV3(rule, turnFrame, haystack, snapshot.actionPriors?.[rule.actionId], mode);
        if (!match.matched || !match.rule)
            continue;
        if ((match.rawScore ?? match.score) > bestScore) {
            bestScore = match.rawScore ?? match.score;
            bestRule = match.rule;
        }
    }
    return {
        route: bestRule?.route || 'no_memory',
        rawScore: bestScore,
        rule: bestRule,
    };
}
function turnFrameFromStoredFrameV3(frame) {
    return {
        summary: frame.redactedTurnSummary,
        userGoal: frame.redactedTurnSummary,
        taskType: frame.taskType,
        activeObjects: [
            ...(frame.projectHint ? [{ kind: 'concept', value: frame.projectHint }] : []),
            ...(frame.repoHint ? [{ kind: 'repo', value: frame.repoHint }] : []),
            ...(frame.toolHints || []).map((tool) => ({ kind: 'tool', value: tool })),
        ],
        impliedNeeds: [...(frame.turnSignals || [])],
        memoryQuestions: [],
        constraints: [],
        routeHints: {
            likelyNeedsCorrections: (frame.routeHintFlags || []).includes('needs_correction') || frame.chosenMemoryTypes.includes('correction'),
            likelyNeedsPreferences: (frame.routeHintFlags || []).includes('needs_preference') || frame.chosenMemoryTypes.includes('preference'),
            likelyNeedsWorkflow: (frame.routeHintFlags || []).includes('needs_workflow') || frame.chosenMemoryTypes.includes('workflow'),
            likelyNeedsProjectContext: (frame.routeHintFlags || []).includes('needs_project_context') || Boolean(frame.projectHint || frame.repoHint),
        },
    };
}
function frameSuccessLabelV3(frame) {
    if (frame.chosenRoute === 'no_memory')
        return Number(frame.reward || 0) >= 0;
    return Number(frame.reward || 0) > 0;
}
function retireStalePrototypesV3(store, prototypes, config) {
    const minCount = Number(config.routeLearning?.policyV3?.prototypeRetirementMinCount ?? 3);
    const harmRate = Number(config.routeLearning?.policyV3?.prototypeRetirementHarmRate ?? 0.7);
    const retiredPrototypeIds = [];
    for (const prototype of prototypes) {
        const sampleCount = Math.max(0, Number(prototype.supportPrior || 0) + Number(prototype.harmPrior || 0));
        if (prototype.status === 'retired' || sampleCount < minCount)
            continue;
        const observedHarmRate = sampleCount > 0 ? Number(prototype.harmPrior || 0) / sampleCount : 0;
        if (observedHarmRate < harmRate)
            continue;
        retiredPrototypeIds.push(prototype.id);
        store.setRouteActionPrototypeStatusV3?.(prototype.id, 'retired');
    }
    return { retiredPrototypeIds };
}
function stablePolicyBodyV3(snapshot) {
    return JSON.stringify({
        rules: snapshot.rules,
        actionPriors: snapshot.actionPriors,
        globalBudgets: snapshot.globalBudgets,
        calibration: snapshot.calibration ? {
            globalThreshold: snapshot.calibration.globalThreshold,
            routeThresholds: snapshot.calibration.routeThresholds,
        } : null,
    });
}
function rewardComponentsForDecision(decision) {
    const reward = Number(decision.reward || 0);
    const hadMemory = (decision.selectedMemoryIds?.length || 0) > 0;
    const ambiguityPenalty = reward === 0 ? 0.15 : Math.max(0, 0.1 - Math.abs(reward) * 0.05);
    const teacherConfidence = Math.max(0.3, Math.min(1, Math.abs(reward) || Number(decision.confidence || 0.5)));
    const validatorConfidence = reward !== 0 ? Math.max(0.45, Math.min(1, Math.abs(reward))) : 0.4;
    const abstainGain = !hadMemory && reward >= 0 ? Math.min(0.35, 0.12 + reward * 0.25) : 0;
    return {
        retrievalHelpGain: reward > 0 && hadMemory ? reward : 0,
        correctionPreventionGain: reward > 0 && decision.route === 'high_confidence_correction_only' ? reward : 0,
        acceptedMemoryGain: reward > 0 && hadMemory ? Math.min(1, reward) : 0,
        noisyInjectionPenalty: reward < 0 && hadMemory ? Math.abs(reward) : 0,
        unnecessarySyncPenalty: decision.syncLlmUsed && reward <= 0 ? 0.25 : 0,
        graphOverreachPenalty: Number(decision.retrievalPlan?.graphDepth || 0) > 1 && reward < 0 ? 0.15 : 0,
        latencyPenalty: Math.min(1, Number(decision.syncLatencyMs || 0) / 5000),
        ambiguityPenalty,
        teacherConfidence,
        validatorConfidence,
        abstainGain,
    };
}
function learningMarginWeightV3(baseConfidence, decision) {
    const ambiguityPenalty = rewardComponentsForDecision(decision).ambiguityPenalty || 0;
    return clamp01(Math.max(0.12, baseConfidence * (1 - ambiguityPenalty)));
}
function updateBanditState(state, feedback, config) {
    const next = state ? {
        ...state,
        actionStats: { ...(state.actionStats || {}) },
    } : {
        agentId: feedback.agentId,
        learnerVersion: 'linucb-lite-v1',
        featureSchemaVersion: 'route-v3-hybrid-24d-v1',
        explorationAlpha: Number(config.routeLearning?.policyV3?.explorationAlpha ?? 0.35),
        sharedWeights: [1, 0.75, 0.5, 0.35],
        actionStats: {},
        updatedAt: feedback.createdAt,
    };
    const current = next.actionStats[feedback.chosenActionId] || {
        count: 0,
        rewardSum: 0,
        rewardMean: 0,
        rewardVariance: 0,
        lastReward: 0,
        positiveCount: 0,
        negativeCount: 0,
        updatedAt: feedback.createdAt,
    };
    const rewardWeight = Math.max(0.2, 1 - Number(feedback.rewardComponents?.ambiguityPenalty || 0));
    const effectiveReward = Number(feedback.reward || 0) * rewardWeight;
    const count = current.count + 1;
    const rewardSum = current.rewardSum + effectiveReward;
    const rewardMean = rewardSum / count;
    const delta = effectiveReward - current.rewardMean;
    const rewardVariance = count > 1 ? ((current.rewardVariance * current.count) + delta * (effectiveReward - rewardMean)) / count : 0;
    next.actionStats[feedback.chosenActionId] = {
        count,
        rewardSum,
        rewardMean,
        rewardVariance,
        lastReward: effectiveReward,
        positiveCount: current.positiveCount + (effectiveReward >= 0 ? 1 : 0),
        negativeCount: current.negativeCount + (effectiveReward < 0 ? 1 : 0),
        updatedAt: feedback.createdAt,
    };
    next.updatedAt = feedback.createdAt;
    return next;
}
function outcomeLabelForDecision(decision) {
    if (Number(decision.reward || 0) > 0)
        return 'accepted';
    if (Number(decision.reward || 0) < 0)
        return 'rejected';
    return 'ambiguous';
}
function sparseMatch(frame, prototype) {
    let score = 0;
    if (prototype.sparseSignature.includes(frame.taskType))
        score += 0.35;
    const signalOverlap = (frame.turnSignals || []).filter((signal) => prototype.sparseSignature.includes(signal)).length;
    score += Math.min(0.35, signalOverlap * 0.05);
    if (frame.projectHint && prototype.sparseSignature.includes(frame.projectHint))
        score += 0.1;
    if (frame.repoHint && prototype.sparseSignature.includes('repo_present'))
        score += 0.05;
    if ((frame.routeHintFlags || []).some((flag) => prototype.sparseSignature.includes(flag)))
        score += 0.15;
    return score;
}
function bilinearScore(left, right) {
    const dim = Math.min(left.length, right.length, EMBED_DIM);
    if (dim === 0)
        return 0;
    let sum = 0;
    for (let i = 0; i < dim; i += 1) {
        const weight = 1 - i / (dim * 1.5);
        sum += left[i] * right[i] * weight;
    }
    return clamp01(0.5 + sum / Math.max(1, dim * 0.5));
}
function embedTextParts(parts) {
    const vector = new Array(EMBED_DIM).fill(0);
    const tokens = stableSignals(parts.flatMap((value) => String(value || '').split(/[^a-z0-9_-]+/i)));
    for (const token of tokens) {
        const hash = hashText(token);
        const index = parseInt(hash.slice(0, 8), 16) % EMBED_DIM;
        const sign = parseInt(hash.slice(8, 10), 16) % 2 === 0 ? 1 : -1;
        vector[index] += sign * (0.5 + (token.length % 7) * 0.08);
    }
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
    return vector.map((value) => Number((value / norm).toFixed(6)));
}
function signalsFromDecision(decision) {
    return stableSignals([
        ...decision.turnFrame.impliedNeeds,
        ...decision.turnFrame.memoryQuestions,
        ...decision.retrievalPlan?.queries || [],
        ...decision.retrievalPlan?.memoryTypes || [],
        ...decision.turnFrame.activeObjects.map((object) => object.value),
    ]);
}
function routeHintFlags(turnFrame) {
    return [
        turnFrame.routeHints.likelyNeedsCorrections ? 'needs_correction' : '',
        turnFrame.routeHints.likelyNeedsPreferences ? 'needs_preference' : '',
        turnFrame.routeHints.likelyNeedsWorkflow ? 'needs_workflow' : '',
        turnFrame.routeHints.likelyNeedsProjectContext ? 'needs_project_context' : '',
    ].filter(Boolean);
}
function topCounts(values, limit) {
    const counts = new Map();
    for (const value of values.map((item) => String(item || '').trim()).filter(Boolean)) {
        counts.set(value, (counts.get(value) || 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit).map(([value]) => value);
}
function sanitizeRoute(value) {
    return ROUTES.includes(value) ? value : 'retrieve_memory';
}
function sanitizeTaskType(value) {
    const taskType = String(value || 'other');
    return ['coding', 'planning', 'debugging', 'writing', 'preference_update', 'correction', 'general_question', 'other'].includes(taskType) ? taskType : 'other';
}
function sanitizeMemoryTypes(values) {
    return unique((values || []).map((value) => String(value || '')).filter((value) => MEMORY_TYPES.includes(value))).slice(0, 8);
}
function sanitizeStrings(values, limit) {
    return stableSignals((values || []).map((value) => String(value || ''))).slice(0, limit);
}
function stableSignals(values) {
    return unique(values.map((value) => String(value || '').toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '')).filter((value) => value.length >= 2 && value.length <= 60));
}
function unique(values) {
    return [...new Set(values)];
}
function clampGraphDepth(value) {
    const n = Math.max(0, Math.min(2, Number(value || 0)));
    return n >= 2 ? 2 : n >= 1 ? 1 : 0;
}
function clamp01(value) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
