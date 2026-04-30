export declare const DECISIONS: readonly string[];
export declare const SLICES: readonly string[];
export declare function decidePolicy(input?: any): {
    kind: string;
    slice: string;
    reasonCode: string;
    verificationHint: boolean;
};
export declare function classifyTurn(input?: any): string;
export declare function normalizeSlice(value: any): string;
