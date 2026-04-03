import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadFunction({ file, startMarker, endMarker, prelude = "" }) {
    const source = readFileSync(path.join(__dirname, "..", "src", file), "utf8");
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker} in ${file}`);
    }
    const block = source.slice(start, end).replace(/^export\s+/gmu, "");
    const match = /function\s+([A-Za-z0-9_]+)/u.exec(startMarker);
    if (match === null) {
        throw new Error(`failed to extract function name from ${startMarker}`);
    }
    return new Function(`${prelude}\n${block}\nreturn ${match[1]};`)();
}

test("current profile status with report reuses one live operator snapshot", () => {
    const describeCurrentProfileBrainStatusWithReport = loadFunction({
        file: "index.js",
        startMarker: "export function describeCurrentProfileBrainStatusWithReport",
        endMarker: "export function describeCurrentProfileBrainStatus(input)",
        prelude: `
            let buildCalls = 0;
            function buildOperatorSurfaceReport() {
                buildCalls += 1;
                return {
                    marker: buildCalls,
                    manyProfile: {
                        declaredAttachmentPolicy: "dedicated"
                    }
                };
            }
            function buildCurrentProfileBrainStatusFromReport(report, policyMode, profileId) {
                return {
                    marker: report.marker,
                    policyMode,
                    profileId
                };
            }
            function normalizeOptionalString(value) {
                return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
            }
            globalThis.__ocbBuildCalls = () => buildCalls;
        `
    });

    const result = describeCurrentProfileBrainStatusWithReport({
        profileId: " current_profile "
    });

    assert.equal(globalThis.__ocbBuildCalls(), 1);
    assert.equal(result.report.marker, 1);
    assert.deepEqual(result.status, {
        marker: 1,
        policyMode: "dedicated",
        profileId: "current_profile"
    });

    delete globalThis.__ocbBuildCalls;
});
