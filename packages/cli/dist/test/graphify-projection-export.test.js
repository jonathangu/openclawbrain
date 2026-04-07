import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, lstatSync, mkdirSync, readFileSync, readlinkSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { exportGraphifyProjection } from "../src/import-export.js";
import { parseOperatorCliArgs, runOperatorCli } from "../src/cli.js";

function createTempRoot(t) {
    const tempRoot = path.join(os.tmpdir(), `openclawbrain-graphify-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    mkdirSync(tempRoot, { recursive: true });
    t.after(() => {
        rmSync(tempRoot, { recursive: true, force: true });
    });
    return tempRoot;
}

function writeFixtureFile(filePath, content) {
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, `${content}\n`, "utf8");
}

function createGraphifyFixture(t) {
    const root = createTempRoot(t);
    const activationRoot = path.join(root, "activation");
    const workspaceRoot = path.join(root, "workspace");
    const repoRoot = path.join(root, "repo");
    const bundleRoot = path.join(root, "artifacts", "graphify-source-bundles");
    const docsRoot = path.join(repoRoot, "docs");
    const codeRoot = path.join(repoRoot, "packages", "cli", "dist", "src");
    const sessionSourcePath = path.join(workspaceRoot, "sessions", "lane-c-session.md");
    const proofSummarySourcePath = path.join(workspaceRoot, "proof-source.md");

    mkdirSync(path.join(activationRoot, "state"), { recursive: true });
    writeFileSync(path.join(activationRoot, "activation-pointers.json"), JSON.stringify({
        activePack: "pack-123",
        previousPack: "pack-122"
    }, null, 2), "utf8");

    writeFixtureFile(path.join(workspaceRoot, "MEMORY.md"), [
        "# MEMORY",
        "- graphify should remain non-authoritative",
        "- preserve source-bundle linkage"
    ].join("\n"));
    writeFixtureFile(path.join(workspaceRoot, "TASKS.md"), [
        "# TASKS",
        "- lane c projection export",
        "- keep the projection explicit"
    ].join("\n"));

    writeFixtureFile(path.join(docsRoot, "README.md"), "# docs root");
    writeFixtureFile(path.join(docsRoot, "architecture", "overview.md"), "# architecture overview");
    writeFixtureFile(path.join(docsRoot, "architecture", "compiled-artifacts.md"), "# compiled artifacts");
    writeFixtureFile(path.join(docsRoot, "proof", "README.md"), "# proof docs");
    writeFixtureFile(path.join(codeRoot, "cli.js"), "export const cli = true;");
    writeFixtureFile(path.join(codeRoot, "import-export.js"), "export const importExport = true;");
    writeFixtureFile(path.join(codeRoot, "semantic-metadata.js"), "export const semantic = true;");

    writeFixtureFile(sessionSourcePath, [
        "# session source",
        "- session key: lane-c",
        "- current work is graphify projection export"
    ].join("\n"));
    writeFixtureFile(proofSummarySourcePath, [
        "# proof source",
        "- proof remains secondary to the machine export"
    ].join("\n"));

    return {
        root,
        activationRoot,
        workspaceRoot,
        repoRoot,
        bundleRoot,
        docsRoot,
        codeRoot,
        sessionSourcePath,
        proofSummarySourcePath
    };
}

test("graphify export writes a non-authoritative source bundle with provenance and mirrors", (t) => {
    const fixture = createGraphifyFixture(t);
    const generatedAt = "2026-04-06T22:34:00.000Z";
    const result = exportGraphifyProjection({
        activationRoot: fixture.activationRoot,
        outputRoot: fixture.bundleRoot,
        runId: "lane-c-run",
        repoRoot: fixture.repoRoot,
        workspaceRoot: fixture.workspaceRoot,
        sessionKey: "lane-c",
        sessionTimestamp: generatedAt,
        sessionSourcePath: fixture.sessionSourcePath,
        proofSummarySourcePath: fixture.proofSummarySourcePath,
        docsRoot: fixture.docsRoot,
        codeRoot: fixture.codeRoot,
        generatedAt
    });

    assert.equal(result.ok, true);
    assert.equal(result.runId, "lane-c-run");
    assert.equal(result.bundleRoot, path.join(fixture.bundleRoot, "lane-c-run"));
    assert.ok(result.sourceBundleHash);
    assert.ok(result.canonicalArchivePath.endsWith(path.join("canonical", "machine-export.tar.gz")));
    assert.ok(result.manifestPath);
    assert.ok(result.sessionProjectionPath);
    assert.ok(result.workspaceMemoryPath);
    assert.ok(result.workspaceTasksPath);
    assert.ok(result.proofSummaryPath);
    assert.ok(result.docsMirrorRoot);
    assert.ok(result.codeMirrorRoot);
    assert.equal(result.warnings?.length ?? 0, 0);

    const bundleRoot = result.bundleRoot;
    const manifestPath = result.manifestPath;
    const sessionPath = path.join(bundleRoot, "sessions", "lane-c", "2026-04-06T22-34-00.000Z.md");
    assert.ok(existsSync(path.join(bundleRoot, "canonical", "machine-export.tar.gz")));
    assert.ok(existsSync(path.join(bundleRoot, "canonical", "machine-export.json")));
    assert.ok(existsSync(manifestPath));
    assert.ok(existsSync(path.join(bundleRoot, "README.md")));
    assert.ok(existsSync(sessionPath));
    assert.ok(existsSync(path.join(bundleRoot, "workspace", "MEMORY.md")));
    assert.ok(existsSync(path.join(bundleRoot, "workspace", "TASKS.md")));
    assert.ok(existsSync(path.join(bundleRoot, "proof", "summary.md")));
    assert.ok(existsSync(path.join(bundleRoot, "docs")));
    assert.ok(existsSync(path.join(bundleRoot, "code")));

    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    const readme = readFileSync(path.join(bundleRoot, "README.md"), "utf8");
    const sessionMarkdown = readFileSync(sessionPath, "utf8");
    const memoryMarkdown = readFileSync(path.join(bundleRoot, "workspace", "MEMORY.md"), "utf8");
    const tasksMarkdown = readFileSync(path.join(bundleRoot, "workspace", "TASKS.md"), "utf8");
    const proofMarkdown = readFileSync(path.join(bundleRoot, "proof", "summary.md"), "utf8");

    assert.equal(manifest.contract, "graphify_source_bundle.v1");
    assert.equal(manifest.authoritative, false);
    assert.equal(manifest.projectionTruth, "projection_only");
    assert.equal(manifest.canonicalMachineExport.sha256.length, 64);
    assert.equal(manifest.sourceBundleHash.length, 64);
    assert.match(readme, /Projection-only surface/);
    assert.match(readme, /non-authoritative/);
    assert.match(sessionMarkdown, /Graphify session projection/);
    assert.match(sessionMarkdown, /session key: `lane-c`/);
    assert.match(sessionMarkdown, /source bundle linkage/);
    assert.match(memoryMarkdown, /Graphify workspace MEMORY projection/);
    assert.match(memoryMarkdown, /non-authoritative/);
    assert.match(memoryMarkdown, /graphify should remain non-authoritative/);
    assert.match(tasksMarkdown, /Graphify workspace TASKS projection/);
    assert.match(tasksMarkdown, /projection explicit/);
    assert.match(proofMarkdown, /Graphify proof summary projection/);
    assert.match(proofMarkdown, /proof remains secondary/);

    const docsStat = lstatSync(path.join(bundleRoot, "docs"));
    const codeStat = lstatSync(path.join(bundleRoot, "code"));
    assert.ok(docsStat.isSymbolicLink() || docsStat.isDirectory());
    assert.ok(codeStat.isSymbolicLink() || codeStat.isDirectory());
    if (docsStat.isSymbolicLink()) {
        assert.equal(readlinkSync(path.join(bundleRoot, "docs")), fixture.docsRoot);
    }
    if (codeStat.isSymbolicLink()) {
        assert.equal(readlinkSync(path.join(bundleRoot, "code")), fixture.codeRoot);
    }
});

test("graphify-export parses and runs through the public CLI surface", (t) => {
    const fixture = createGraphifyFixture(t);
    const generatedAt = "2026-04-06T22:45:00.000Z";
    const argv = [
        "graphify-export",
        "--activation-root",
        fixture.activationRoot,
        "--output-root",
        fixture.bundleRoot,
        "--run-id",
        "cli-run",
        "--repo-root",
        fixture.repoRoot,
        "--workspace-root",
        fixture.workspaceRoot,
        "--session-key",
        "lane-c",
        "--session-source",
        fixture.sessionSourcePath,
        "--proof-summary-source",
        fixture.proofSummarySourcePath,
        "--docs-root",
        fixture.docsRoot,
        "--code-root",
        fixture.codeRoot,
        "--generated-at",
        generatedAt,
        "--json"
    ];
    const parsed = parseOperatorCliArgs(argv);
    assert.equal(parsed.command, "graphify-export");
    assert.equal(parsed.activationRoot, path.resolve(fixture.activationRoot));
    assert.equal(parsed.outputRoot, path.resolve(fixture.bundleRoot));
    assert.equal(parsed.runId, "cli-run");
    assert.equal(parsed.repoRoot, path.resolve(fixture.repoRoot));
    assert.equal(parsed.workspaceRoot, path.resolve(fixture.workspaceRoot));
    assert.equal(parsed.sessionKey, "lane-c");
    assert.equal(parsed.sessionSourcePath, path.resolve(fixture.sessionSourcePath));
    assert.equal(parsed.proofSummarySourcePath, path.resolve(fixture.proofSummarySourcePath));
    assert.equal(parsed.docsRoot, path.resolve(fixture.docsRoot));
    assert.equal(parsed.codeRoot, path.resolve(fixture.codeRoot));
    assert.equal(parsed.generatedAt, generatedAt);
    assert.equal(parsed.json, true);

    const logs = [];
    const errors = [];
    const originalLog = console.log;
    const originalError = console.error;
    console.log = (...args) => {
        logs.push(args.join(" "));
    };
    console.error = (...args) => {
        errors.push(args.join(" "));
    };
    try {
        const exitCode = runOperatorCli(argv);
        assert.equal(exitCode, 0);
    }
    finally {
        console.log = originalLog;
        console.error = originalError;
    }
    assert.equal(errors.length, 0);
    const payload = JSON.parse(logs.join("\n"));
    assert.equal(payload.ok, true);
    assert.equal(payload.runId, "cli-run");
    assert.equal(payload.authoritative, undefined);
    assert.ok(existsSync(path.join(fixture.bundleRoot, "cli-run", "corpus-manifest.json")));
});
