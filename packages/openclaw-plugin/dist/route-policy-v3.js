import { hashText, redactText } from './redact.js';
const ROUTES = ['no_memory', 'capture_only', 'retrieve_memory', 'retrieve_and_distill', 'high_confidence_correction_only'];
const RETRIEVAL_ROUTES = new Set(['retrieve_memory', 'retrieve_and_distill', 'high_confidence_correction_only']);
const MEMORY_TYPES = ['correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'recall_rule', 'outcome', 'context'];
const SYNC_MODES = ['no', 'never_unless_ambiguous', 'allowed', 'prefer'];
const EMBED_DIM = 24;
export function ingestRouteLearningArtifactsV3(store, agentId, decision, routeFrame, teacherRun, counterfactuals, lessons, config) {
    const chosenPrototype = upsertPrototypeForDecision(store, agentId, decision, lessons, 'learned');
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
    const teacherPrototype = upsertPrototypeForTeacher(store, agentId, teacherRun, lessons, 'distilled');
    prototypeIds.add(teacherPrototype.id);
    if (teacherPrototype.id !== chosenPrototype.id) {
        const pair = store.insertRoutePairExampleV3({
            agentId,
            frameId: frame.id,
            positiveActionId: teacherPrototype.id,
            negativeActionId: chosenPrototype.id,
            labelSource: 'teacher',
            marginWeight: clamp01(Math.max(0.2, Number(teacherRun.confidence || 0.5))),
            evidenceIds: unique([decision.id, teacherRun.id]),
        });
        pairExamples.push(pair);
    }
    for (const cf of counterfactuals) {
        if (cf.estimatedOutcome !== 'likely_helpful' && cf.estimatedOutcome !== 'likely_missed')
            continue;
        const prototype = upsertPrototypeForCounterfactual(store, agentId, decision, cf, 'distilled');
        prototypeIds.add(prototype.id);
        if (prototype.id === chosenPrototype.id)
            continue;
        pairExamples.push(store.insertRoutePairExampleV3({
            agentId,
            frameId: frame.id,
            positiveActionId: prototype.id,
            negativeActionId: chosenPrototype.id,
            labelSource: 'counterfactual',
            marginWeight: clamp01(Math.max(0.15, Number(cf.confidence || teacherRun.confidence || 0.5))),
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
        });
        prototypeIds.add(silencePrototype.id);
        pairExamples.push(store.insertRoutePairExampleV3({
            agentId,
            frameId: frame.id,
            positiveActionId: silencePrototype.id,
            negativeActionId: chosenPrototype.id,
            labelSource: 'outcome',
            marginWeight: clamp01(Math.abs(Number(decision.reward || 0))),
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
    const frames = store.listRouteFramesV3(agentId, 400);
    const pairs = store.listRoutePairExamplesV3(agentId, 1200);
    const prototypes = store.listRouteActionPrototypesV3(agentId, 200).filter((prototype) => prototype.status !== 'retired');
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
        }));
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
    const draft = buildSnapshotV3(agentId, rules, priors, frames, prototypes, config, 'candidate');
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
    const finalStatus = validation.ok
        ? (config.routeLearning?.policyV3?.shadowBeforeActivate === true ? 'shadow' : 'active')
        : 'rejected';
    const stored = store.insertPolicySnapshotV3({
        ...draft,
        status: finalStatus,
        evalSummary: {
            ...(draft.evalSummary ?? {}),
            activationDecision: finalStatus,
            activationStatusReason: validation.reason,
            validationErrors: validation.errors,
            validationWarnings: validation.warnings,
            projectedSyncPlannerRate: validation.projectedSyncPlannerRate,
            noisyActionRate: validation.noisyActionRate,
            harmRate: validation.harmRate,
        },
    });
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
    const rules = [];
    const minConfidence = Number(config.routeLearning?.policyV3?.minRuleConfidence ?? 0.55);
    for (const prototype of prototypes) {
        const rows = frames.filter((frame) => frame.chosenActionId === prototype.id);
        const taskTypes = topCounts(rows.map((frame) => frame.taskType), 1);
        const signals = topCounts(rows.flatMap((frame) => frame.turnSignals), 8);
        const projects = topCounts(rows.map((frame) => frame.projectHint || '').filter(Boolean), 1);
        const priors = priorsForPrototype(prototype.id, rows, pairCounts, banditState);
        const confidence = clamp01(0.42 +
            Math.min(0.18, priors.support * 0.03) -
            Math.min(0.2, priors.harm * 0.05) +
            priors.pairWinRate * 0.18 +
            Math.max(-0.1, Math.min(0.15, priors.banditMeanReward * 0.18)));
        if (confidence < minConfidence)
            continue;
        const rule = {
            id: hashText(`${prototype.id}:${taskTypes.join('|')}:${signals.join('|')}:${priors.support}:${priors.harm}`).slice(7, 25),
            priority: prototype.route === 'no_memory' ? 95 : 55,
            actionId: prototype.id,
            match: {
                taskType: taskTypes[0] || undefined,
                turnSignals: signals,
                projectHint: projects[0] || undefined,
                repoHintPresent: rows.some((frame) => Boolean(frame.repoHint)) || undefined,
                safetySignalsAbsent: prototype.route === 'no_memory' ? ['unsafe'] : undefined,
            },
            route: prototype.route,
            memoryTypes: prototype.memoryTypes,
            queries: prototype.queryTemplateFamily,
            graphDepth: prototype.graphDepth,
            syncPlanner: prototype.syncPlanner,
            confidence,
            evidenceIds: unique([...prototype.sourceExampleIds, ...rows.map((frame) => frame.routeDecisionId)]).slice(0, 40),
            priors,
            reason: `distilled_from_prototype:${prototype.id}`,
        };
        if (validateRuleShapeV3(rule, config).length === 0)
            rules.push(rule);
    }
    return rules
        .sort((a, b) => Number(b.priority || 0) - Number(a.priority || 0) || b.confidence - a.confidence)
        .slice(0, Math.max(4, Number(config.routeLearning?.policyV3?.maxRules ?? 32)));
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
    if (existing?.evalSummary) {
        const oldHarmRate = Number(existing.evalSummary.harmRate ?? 0);
        if (harmRate > oldHarmRate + 0.05)
            errors.push('candidate_harm_rate_regressed');
    }
    const reason = errors.length === 0 ? 'passed_activation_gates' : errors[0];
    return { ok: errors.length === 0, status: errors.length === 0 ? 'active' : 'rejected', reason, errors, warnings, projectedSyncPlannerRate, noisyActionRate, harmRate };
}
export function scorePolicySnapshotV3(snapshot, turnFrame, message = '') {
    if (!snapshot || snapshot.version !== 'route-policy-v3' || snapshot.status !== 'active' || !Array.isArray(snapshot.rules)) {
        return { matched: false, score: 0, reasonCode: 'no_active_policy_v3' };
    }
    const haystack = `${message} ${turnFrame.summary} ${turnFrame.userGoal} ${turnFrame.impliedNeeds.join(' ')} ${turnFrame.memoryQuestions.join(' ')}`.toLowerCase();
    let best = { matched: false, score: 0, reasonCode: 'no_matching_policy_rule_v3' };
    for (const rule of snapshot.rules) {
        const match = scoreRuleV3(rule, turnFrame, haystack, snapshot.actionPriors?.[rule.actionId]);
        if (!match.matched)
            continue;
        if (match.score > best.score)
            best = match;
    }
    return best;
}
export function rankActionPrototypesV3(frame, prototypes, banditState) {
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
        const total = sparse * 0.6 + dense * 0.25 + Math.max(-0.15, Math.min(0.15, bonus * 0.15));
        return { prototype, score: total, sparse, dense, bonus };
    }).sort((a, b) => b.score - a.score);
}
function scoreRuleV3(rule, turnFrame, haystack, priors) {
    const match = rule.match ?? {};
    if (match.taskType && match.taskType !== turnFrame.taskType)
        return { matched: false, rule, score: 0, reasonCode: 'task_type_mismatch' };
    const signals = Array.isArray(match.turnSignals) ? match.turnSignals.map((value) => String(value).toLowerCase()).filter(Boolean) : [];
    const overlap = signals.filter((signal) => haystack.includes(signal)).length;
    if (signals.length > 0 && overlap === 0)
        return { matched: false, rule, score: 0, reasonCode: 'turn_signal_mismatch' };
    const priority = Number(rule.priority || 0) / 200;
    const signalBonus = signals.length ? Math.min(0.15, overlap * 0.03) : 0;
    const routeHintBonus = routeHintCompatibility(rule, turnFrame);
    const banditBonus = priors ? Math.max(-0.08, Math.min(0.12, Number(priors.banditMeanReward || 0) * 0.12 + Number(priors.pairWinRate || 0) * 0.08)) : 0;
    const supportBonus = priors ? Math.min(0.08, Number(priors.support || 0) * 0.01) : 0;
    const score = clamp01(Number(rule.confidence || 0) + priority + signalBonus + routeHintBonus + banditBonus + supportBonus);
    return { matched: true, rule, score, reasonCode: `policy_v3_rule:${rule.id}` };
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
function upsertPrototypeForDecision(store, agentId, decision, lessons, provenance) {
    const lesson = lessons.find((item) => item.route === decision.route) || lessons[0];
    return upsertPrototype(store, {
        agentId,
        route: sanitizeRoute(decision.route),
        memoryTypes: sanitizeMemoryTypes(decision.retrievalPlan?.memoryTypes?.length ? decision.retrievalPlan.memoryTypes : lesson?.memoryTypes ?? []),
        graphDepth: clampGraphDepth(decision.retrievalPlan?.graphDepth ?? lesson?.graphDepth ?? 0),
        syncPlanner: decision.syncLlmUsed ? 'allowed' : 'no',
        queryTemplateFamily: sanitizeStrings(decision.retrievalPlan?.queries?.length ? decision.retrievalPlan.queries : lesson?.queryTemplates ?? [], 8),
        sparseSignature: stableSignals([
            sanitizeTaskType(decision.turnFrame.taskType),
            ...signalsFromDecision(decision),
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
    });
}
function upsertPrototypeForTeacher(store, agentId, teacherRun, lessons, provenance) {
    const memoryTypes = sanitizeMemoryTypes(lessons.flatMap((lesson) => lesson.route === teacherRun.teacherRoute ? lesson.memoryTypes : []).length
        ? lessons.flatMap((lesson) => lesson.route === teacherRun.teacherRoute ? lesson.memoryTypes : [])
        : []);
    return upsertPrototype(store, {
        agentId,
        route: sanitizeRoute(teacherRun.teacherRoute),
        memoryTypes,
        graphDepth: clampGraphDepth(teacherRun.teacherGraphDepth ?? 0),
        syncPlanner: teacherRun.syncPlannerWorthIt ? 'allowed' : 'no',
        queryTemplateFamily: sanitizeStrings(teacherRun.teacherQueries ?? [], 8),
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
    });
}
function upsertPrototypeForCounterfactual(store, agentId, decision, cf, provenance) {
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
        queryTemplateFamily: sanitizeStrings(decision.retrievalPlan?.queries ?? [], 8),
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
    });
}
function upsertPrototype(store, input) {
    const prototypeId = input.id || prototypeIdFor(input);
    return store.upsertRouteActionPrototypeV3({ ...input, id: prototypeId });
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
function priorsForPrototype(prototypeId, frames, pairCounts, banditState) {
    const support = frames.filter((frame) => Number(frame.reward || 0) >= 0).length;
    const harm = frames.filter((frame) => Number(frame.reward || 0) < 0).length;
    const pair = pairCounts[prototypeId] || { wins: 0, losses: 0 };
    const pairWinRate = pair.wins + pair.losses > 0 ? pair.wins / (pair.wins + pair.losses) : 0.5;
    const stats = banditState?.actionStats?.[prototypeId];
    return {
        support,
        harm,
        banditMeanReward: Number(stats?.rewardMean || 0),
        banditCount: Number(stats?.count || 0),
        pairWinRate,
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
function buildSnapshotV3(agentId, rules, actionPriors, frames, prototypes, config, status, evalOverride) {
    const harms = frames.filter((frame) => Number(frame.reward || 0) < 0).length;
    const noisy = frames.filter((frame) => Number(frame.rewardComponents?.noisyInjectionPenalty || 0) > 0).length;
    const syncRules = rules.filter((rule) => rule.syncPlanner === 'allowed' || rule.syncPlanner === 'prefer').length;
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
        },
        evalSummary: evalOverride ?? {
            frames: frames.length,
            pairExamples: 0,
            prototypes: prototypes.length,
            projectedSyncPlannerRate: rules.length ? syncRules / rules.length : 0,
            noisyActionRate: frames.length ? noisy / frames.length : 0,
            harmRate: frames.length ? harms / frames.length : 0,
        },
        sourceFrameIds: frames.map((frame) => frame.id).slice(0, 200),
        sourcePrototypeIds: prototypes.map((prototype) => prototype.id).slice(0, 200),
        model: config.llm?.learningModel || 'deterministic-route-policy-v3',
        promptVersion: 'route-policy-v3-distiller-v1',
    };
}
function stablePolicyBodyV3(snapshot) {
    return JSON.stringify({ rules: snapshot.rules, actionPriors: snapshot.actionPriors, globalBudgets: snapshot.globalBudgets });
}
function rewardComponentsForDecision(decision) {
    const reward = Number(decision.reward || 0);
    const hadMemory = (decision.selectedMemoryIds?.length || 0) > 0;
    return {
        retrievalHelpGain: reward > 0 && hadMemory ? reward : 0,
        correctionPreventionGain: reward > 0 && decision.route === 'high_confidence_correction_only' ? reward : 0,
        acceptedMemoryGain: reward > 0 && hadMemory ? Math.min(1, reward) : 0,
        noisyInjectionPenalty: reward < 0 && hadMemory ? Math.abs(reward) : 0,
        unnecessarySyncPenalty: decision.syncLlmUsed && reward <= 0 ? 0.25 : 0,
        graphOverreachPenalty: Number(decision.retrievalPlan?.graphDepth || 0) > 1 && reward < 0 ? 0.15 : 0,
        latencyPenalty: Math.min(1, Number(decision.syncLatencyMs || 0) / 5000),
    };
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
    const count = current.count + 1;
    const rewardSum = current.rewardSum + Number(feedback.reward || 0);
    const rewardMean = rewardSum / count;
    const delta = Number(feedback.reward || 0) - current.rewardMean;
    const rewardVariance = count > 1 ? ((current.rewardVariance * current.count) + delta * (Number(feedback.reward || 0) - rewardMean)) / count : 0;
    next.actionStats[feedback.chosenActionId] = {
        count,
        rewardSum,
        rewardMean,
        rewardVariance,
        lastReward: Number(feedback.reward || 0),
        positiveCount: current.positiveCount + (feedback.reward >= 0 ? 1 : 0),
        negativeCount: current.negativeCount + (feedback.reward < 0 ? 1 : 0),
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
