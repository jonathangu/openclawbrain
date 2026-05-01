export class FakeLlmClient {
    handler;
    responses;
    attempts = 0;
    constructor(options = {}) {
        this.handler = options.handler;
        this.responses = [...(options.responses ?? [])];
    }
    async runJson(call) {
        this.attempts += 1;
        if (this.handler)
            return this.handler(call, this.attempts);
        if (this.responses.length === 0)
            throw new Error('FakeLlmClient has no queued response');
        return this.responses.shift();
    }
}
export class OpenAICompatibleLlmClient {
    baseUrl;
    apiKey;
    path;
    fetchImpl;
    headers;
    constructor(options) {
        this.baseUrl = options.baseUrl.replace(/\/$/, '');
        this.apiKey = options.apiKey;
        this.path = options.path ?? '/chat/completions';
        this.fetchImpl = options.fetchImpl ?? fetch;
        this.headers = options.headers ?? {};
    }
    async runJson(call) {
        const response = await this.fetchImpl(`${this.baseUrl}${this.path}`, {
            method: 'POST',
            headers: {
                'content-type': 'application/json',
                ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}),
                ...this.headers,
            },
            body: JSON.stringify({
                model: call.model,
                temperature: call.temperature,
                max_tokens: call.maxTokens,
                response_format: { type: 'json_object' },
                messages: [
                    { role: 'system', content: call.systemPrompt },
                    {
                        role: 'user',
                        content: JSON.stringify({
                            task: call.task,
                            input: call.input,
                            schema: call.schema ?? null,
                        }),
                    },
                ],
            }),
        });
        if (!response.ok) {
            const text = await response.text().catch(() => '');
            throw new Error(`OpenAI-compatible JSON call failed: ${response.status} ${response.statusText} ${text}`.trim());
        }
        const payload = await response.json();
        const content = payload?.choices?.[0]?.message?.content;
        if (typeof content !== 'string') {
            throw new Error('OpenAI-compatible JSON call returned no message content');
        }
        return content;
    }
}
