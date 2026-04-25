export declare const GRAPHIFY_BRIDGE_CUT_CONTRACT_V1 = "graphify_bridge_cut.v1";
export declare const GRAPHIFY_BRIDGE_DUAL_SOURCE_BUNDLE_CONTRACT_V1 = "graphify_dual_source_bundle_manifest.v1";
export declare const GRAPHIFY_BRIDGE_VERSION_V1 = "graphify-bridge-cut@1";
export declare const GRAPHIFY_BRIDGE_CUT_LAYOUT_V1: {
    sourceBundleRoot: string;
    canonicalSourceBundle: string;
    projectionSourceBundle: string;
    sourceBundleManifest: string;
    graphifyRun: string;
    compiledArtifactPack: string;
    status: string;
    summary: string;
};
export declare function runGraphifyBridgeCut(options?: any): any;
