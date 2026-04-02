import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";

const cliSourcePath = path.resolve(
  import.meta.dirname,
  "..",
  "packages",
  "cli",
  "dist",
  "src",
  "cli.js",
);

describe("shadow extension dependency completeness", () => {
  it("buildExtensionPackageJson always declares @openclawbrain/openclaw as a dependency", () => {
    const cliSource = readFileSync(cliSourcePath, "utf8");

    // The generated shadow extension index.ts imports from @openclawbrain/openclaw.
    // buildExtensionPackageJson must declare it as a dependency even when the
    // CLI package metadata resolves to @openclawbrain/cli.
    const match = cliSource.match(
      /function buildExtensionPackageJson\(\)\s*\{[\s\S]*?return JSON\.stringify\(/,
    );
    expect(match).not.toBeNull();

    // Verify the fix: the function must contain a guard that adds
    // @openclawbrain/openclaw when packageMetadata.name differs.
    const fnBody = cliSource.slice(
      cliSource.indexOf("function buildExtensionPackageJson()"),
      cliSource.indexOf("function buildExtensionPackageJson()") + 1200,
    );
    expect(fnBody).toContain('"@openclawbrain/openclaw"');
    expect(fnBody).toContain('packageMetadata.name !== "@openclawbrain/openclaw"');
  });

  it("extension template imports from @openclawbrain/openclaw", () => {
    // Verify the claim that the extension template actually imports from @openclawbrain/openclaw
    const extensionTemplatePaths = [
      path.resolve(import.meta.dirname, "..", "packages", "cli", "extension", "index.ts"),
      path.resolve(import.meta.dirname, "..", "packages", "openclaw", "extension", "index.ts"),
    ];

    for (const templatePath of extensionTemplatePaths) {
      const source = readFileSync(templatePath, "utf8");
      expect(source).toContain('@openclawbrain/openclaw');
    }
  });
});
