export function detectCaptureIntent(input) {
    const raw = String(input.latestUserMessageRedacted ?? '').trim();
    const text = normalize(raw);
    const scope = inferScope(raw, input);
    if (!text)
        return result(false, 'one_off', 0.5, 'No user text to evaluate', [], [], scope);
    if (matchesForget(text)) {
        return result(true, 'delete_or_suppress', 0.92, 'User asked to forget, delete, suppress, or stop using memory', ['forget/delete'], ['delete_requested'], scope);
    }
    if (matchesCredentialSecret(raw)) {
        return result(true, 'sensitive_secret', 0.96, 'User text contains credential-like secret storage language', ['credential_secret'], ['credential_like_secret'], scope);
    }
    if (matchesRecallRule(text)) {
        return result(true, 'recall_rule', 0.88, 'User defined an explicit future recall response', ['if/when ask answer'], riskHintsForText(raw, 'recall_rule'), scope);
    }
    if (matchesRetrievalQuestion(text)) {
        return result(false, 'retrieval_question', 0.78, 'User is asking for existing memory rather than defining new memory', ['retrieval_question'], riskHintsForText(raw, 'retrieval_question'), scope);
    }
    if (matchesRouting(text)) {
        return result(true, 'routing_rule', 0.78, 'User stated a routing or delegation rule', ['routing'], riskHintsForText(raw, 'routing_rule'), scope);
    }
    if (matchesAgentAssignment(text)) {
        return result(true, 'agent_assignment', 0.74, 'User stated an ownership or responsibility assignment', ['agent_assignment'], riskHintsForText(raw, 'agent_assignment'), scope);
    }
    if (matchesExplicitStore(text)) {
        const intent = /\b(codeword|passphrase|auth phrase|authentication phrase)\b/i.test(raw) && !/\b(project\s+code\s*name|project\s+codename|codename|code\s*name)\b/i.test(raw)
            ? 'ambiguous'
            : 'explicit_store';
        const confidence = intent === 'ambiguous' ? 0.62 : 0.82;
        return result(true, intent, confidence, intent === 'ambiguous' ? 'User used explicit memory language around a codeword-like value without a recall trigger' : 'User explicitly asked to remember or store durable information', ['explicit_memory_language'], riskHintsForText(raw, intent), scope);
    }
    if (matchesCorrection(text)) {
        return result(true, 'explicit_update', 0.78, 'User corrected or updated prior behavior/facts', ['correction'], riskHintsForText(raw, 'explicit_update'), scope);
    }
    if (matchesPreference(text)) {
        return result(true, 'standing_preference', 0.72, 'User stated a durable preference', ['preference'], riskHintsForText(raw, 'standing_preference'), scope);
    }
    if (matchesWorkflow(text)) {
        return result(true, 'standing_workflow', 0.72, 'User stated a standing workflow, project convention, or future-facing instruction', ['workflow'], riskHintsForText(raw, 'standing_workflow'), scope);
    }
    if (matchesToolConvention(text)) {
        return result(true, 'tool_convention', 0.7, 'User stated a recurring tool convention', ['tool_convention'], riskHintsForText(raw, 'tool_convention'), scope);
    }
    if (matchesProjectFact(text)) {
        return result(true, 'project_fact', 0.68, 'User stated a durable project or environment fact', ['project_fact'], riskHintsForText(raw, 'project_fact'), scope);
    }
    if (matchesOutcome(text)) {
        return result(true, 'standing_workflow', 0.66, 'User reported a tool/workflow outcome worth evaluating', ['outcome'], riskHintsForText(raw, 'standing_workflow'), scope);
    }
    if (looksOneOff(text)) {
        return result(false, 'one_off', 0.72, 'Message looks like a one-off task, not durable memory', ['one_off'], riskHintsForText(raw, 'one_off'), scope);
    }
    return result(false, 'ambiguous', 0.3, 'No durable capture signal found', [], riskHintsForText(raw, 'ambiguous'), scope);
}
export function detectRetrievalIntent(input) {
    const raw = String(input.latestUserMessageRedacted ?? '').trim();
    const text = normalize(raw);
    const scopeHints = [inferScope(raw, input)].filter(Boolean);
    const query = raw || '';
    if (/\b(what(?:'s| is)|tell me|give me|remind me)\b.*\b(codeword|passphrase|phrase|answer)\b/i.test(raw)) {
        return retrieval(true, 'recall_value_request', 0.88, query, scopeHints, true, ['recall_rule']);
    }
    if (matchesForget(text)) {
        return retrieval(true, 'memory_management', 0.82, query, scopeHints, false, ['correction', 'preference', 'workflow', 'routing_rule', 'context', 'recall_rule']);
    }
    if (matchesRetrievalQuestion(text) || /\b(as before|same as last time|continue from|we discussed|like i said|my preference|our workflow|thing from yesterday)\b/i.test(raw)) {
        return retrieval(true, 'needs_memory', 0.78, query, scopeHints, false, ['correction', 'preference', 'workflow', 'routing_rule', 'project_fact', 'context']);
    }
    if (/\b(actually|instead|wrong|no,)\b/i.test(raw)) {
        return retrieval(true, 'may_need_memory', 0.65, query, scopeHints, false, ['correction']);
    }
    if (/\b(plan|design|architecture|implementation|install|dependency|dependencies|pnpm|npm|yarn|build|test|setup|repo|project)\b/i.test(raw)) {
        return retrieval(true, 'may_need_memory', 0.62, query, scopeHints, false, ['correction', 'preference', 'workflow', 'tool_convention', 'project_fact', 'agent_assignment', 'outcome', 'context']);
    }
    return retrieval(false, 'no_retrieval', 0.7, query, scopeHints, false, []);
}
export function classifySensitiveValue(text, captureIntent) {
    if (matchesCredentialSecret(text)) {
        return { kind: 'credential_secret', plaintextAllowed: false, proactiveInjectionAllowed: false, reason: 'Credential-like value detected' };
    }
    if (/\b(codeword|passphrase|auth phrase|authentication phrase)\b/i.test(text)) {
        if (captureIntent === 'recall_rule' && matchesRecallRule(normalize(text))) {
            return { kind: 'user_authorized_recall', plaintextAllowed: true, proactiveInjectionAllowed: false, reason: 'Explicit if/when-asked recall rule' };
        }
        if (/\b(project\s+code\s*name|project\s+codename|codename|code\s*name)\b/i.test(text)) {
            return { kind: 'ordinary', plaintextAllowed: true, proactiveInjectionAllowed: true, reason: 'Benign project codename language' };
        }
        return { kind: 'ambiguous_codeword', plaintextAllowed: false, proactiveInjectionAllowed: false, reason: 'Codeword-like value lacks explicit recall trigger' };
    }
    return { kind: 'ordinary', plaintextAllowed: true, proactiveInjectionAllowed: true, reason: 'No sensitive value signal' };
}
export function captureStoreThreshold(intent) {
    switch (intent) {
        case 'explicit_update': return 0.6;
        case 'explicit_store': return 0.65;
        case 'standing_preference': return 0.7;
        case 'standing_workflow': return 0.75;
        case 'project_fact': return 0.75;
        case 'tool_convention': return 0.75;
        case 'routing_rule': return 0.7;
        case 'agent_assignment': return 0.7;
        case 'recall_rule': return 0.85;
        default: return 0.7;
    }
}
function result(shouldConsiderCapture, intent, confidence, reason, matchedSignals, riskHints, proposedScope) {
    return { shouldConsiderCapture, intent, confidence, reason, matchedSignals, riskHints, proposedScope };
}
function retrieval(shouldRetrieve, intent, confidence, query, scopeHints, includeRecallRules, memoryTypes) {
    return { shouldRetrieve, intent, confidence, query, scopeHints, includeRecallRules, memoryTypes };
}
function normalize(value) {
    return value.toLowerCase().replace(/[“”]/g, '"').replace(/[’]/g, "'").replace(/\s+/g, ' ').trim();
}
function matchesExplicitStore(text) {
    return /\b(remember that|remember this|store this|save this|keep this in mind|memorize|don't forget|do not forget)\b/.test(text);
}
function matchesRetrievalQuestion(text) {
    return /\b(do you remember|what did i tell you|what do you remember|did i tell you|remember when)\b/.test(text);
}
function matchesCorrection(text) {
    return /\b(actually|no,|wrong|that's wrong|correction|the correct|use .+ instead of|not .+, .+)\b/.test(text);
}
function matchesPreference(text) {
    return /\b(i prefer|i like|i don't like|i do not like|i hate|my preference|use this style|i want .+ replies|keep (it|this|replies) .+)\b/.test(text);
}
function matchesWorkflow(text) {
    return /\b(going forward|from now on|next time|in the future|whenever|for this repo|for this project|always|never|must|should|run .+ before)\b/.test(text);
}
function matchesRouting(text) {
    return /\b(route .+ to|send .+ to|assign .+ to|use the .+ agent|delegate .+ to)\b/.test(text);
}
function matchesAgentAssignment(text) {
    return /\b(.+ agent owns|.+ owns .+ tasks|.+ is responsible for|.+ approves .+ policy|.+ live install host)\b/.test(text);
}
function matchesToolConvention(text) {
    return /\b(when editing|when using|use .+ cli|use .+ for .+ workflow|do not use .+ for|don't use .+ for|run .+ from)\b/.test(text);
}
function matchesProjectFact(text) {
    return /\b(runs on|repo uses|project uses|deploys from|staging database|production database|is local-only|codename is|code name is)\b/.test(text);
}
function matchesRecallRule(text) {
    return /\b(if|when) i (?:ask|say|mention).{1,160}\b(answer|tell|give|respond with|say)\b/.test(text)
        || /\b(if|when) i (?:ask|say|mention).{1,160}\b(the codeword is|the phrase is)\b/.test(text);
}
function matchesForget(text) {
    return /\b(forget|delete|remove|don't remember|do not remember|do not store|don't store|stop using|suppress)\b/.test(text);
}
function matchesCredentialSecret(text) {
    return /\b(api[_ -]?key|password|passwd|private key|ssh key|recovery phrase|seed phrase|session cookie|bearer token|access token|refresh token|client secret|secret key)\b/i.test(text)
        || /\b(?:sk|pk|ghp|github_pat|xox[baprs])-?[A-Za-z0-9_\-]{16,}\b/.test(text)
        || /-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----/.test(text);
}
function looksOneOff(text) {
    return /\b(summarize|write a poem|convert this|translate this|remind me to|set a reminder|make this one|draft this|send this)\b/.test(text)
        && !matchesExplicitStore(text)
        && !matchesWorkflow(text)
        && !matchesPreference(text);
}
function matchesOutcome(text) {
    return /\b(that worked|that fixed it|that failed|that broke|this was the right command|this is the working command)\b/.test(text);
}
function riskHintsForText(text, intent) {
    const hints = new Set();
    if (matchesCredentialSecret(text))
        hints.add('credential_like_secret');
    if (/\b(codeword|passphrase|auth phrase|authentication phrase)\b/i.test(text))
        hints.add(intent === 'recall_rule' ? 'codeword_like_value' : 'ambiguous_sensitive_recall');
    if (intent === 'recall_rule')
        hints.add('benign_recall');
    if (hints.size === 0)
        hints.add('ordinary');
    return [...hints];
}
function inferScope(text, input) {
    if (/\btelegram\b/i.test(text))
        return { kind: 'channel', key: 'telegram' };
    if (/\bcormorantai\b/i.test(text))
        return { kind: 'app', key: 'CormorantAI' };
    if (/\bopenclawbrain|openclaw brain|ocb\b/i.test(text))
        return { kind: 'repo', key: 'openclawbrain' };
    if (/\bpelican\b/i.test(text))
        return { kind: 'project', key: 'Pelican' };
    if (/\bbountiful\b/i.test(text))
        return { kind: 'project', key: 'Bountiful Garden' };
    const forProject = text.match(/\bfor\s+([A-Z][A-Za-z0-9_-]{2,})\b/);
    if (forProject)
        return { kind: 'project', key: forProject[1] };
    if (/\bthis repo\b/i.test(text))
        return { kind: 'repo', key: 'current_repo' };
    if (/\bthis project\b/i.test(text))
        return { kind: 'project', key: 'current_project' };
    if (input.sessionId || input.sessionKey)
        return { kind: 'agent', key: input.agentId || 'main' };
    return { kind: 'global_user', key: 'default' };
}
