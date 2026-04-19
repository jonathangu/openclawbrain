#!/usr/bin/env node

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildTeacherProposalReportArtifactV1,
  renderTeacherProposalReportArtifactMarkdownV1,
} from "../src/brain-core/teacher-v3-proposal-artifact.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

export const TEACHER_V3_PROPOSAL_ARTIFACT_LAYOUT = {
  markdown: "artifact.md",
  meta: "artifact.meta.json",
};

export const DEFAULT_TEACHER_V3_PROPOSAL_ARTIFACT_PARENT = path.join(
  repoRoot,
  "artifacts",
  "teacher-v3-proposal-artifacts",
);

function ensureDir(dirPath) {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

function renderJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

export function resolveTeacherV3ProposalArtifactOutputDir({
  outputDir = null,
  proposalId = null,
} = {}) {
  if (typeof outputDir === "string" && outputDir.trim().length > 0) {
    return path.resolve(outputDir);
  }
  const leaf = typeof proposalId === "string" && proposalId.trim().length > 0
    ? proposalId.trim()
    : "proposal-artifact";
  return path.join(DEFAULT_TEACHER_V3_PROPOSAL_ARTIFACT_PARENT, leaf);
}

export function buildTeacherV3ProposalArtifact(input) {
  const artifact = buildTeacherProposalReportArtifactV1({
    proposal: input.proposal,
    artifactId: input.artifactId,
    recommendations: input.recommendations,
  });
  const outputDir = resolveTeacherV3ProposalArtifactOutputDir({
    outputDir: input.outputDir,
    proposalId: artifact.proposalId,
  });
  const markdown = renderTeacherProposalReportArtifactMarkdownV1(artifact);

  return {
    artifactId: artifact.artifactId,
    outputDir,
    artifact,
    files: {
      [TEACHER_V3_PROPOSAL_ARTIFACT_LAYOUT.markdown]: markdown,
      [TEACHER_V3_PROPOSAL_ARTIFACT_LAYOUT.meta]: renderJson(artifact),
    },
  };
}

export function writeTeacherV3ProposalArtifact(outputDir, builtArtifact) {
  const resolvedOutputDir = path.resolve(outputDir);
  ensureDir(resolvedOutputDir);

  const writtenFiles = [];
  for (const [relativePath, content] of Object.entries(builtArtifact.files)) {
    const targetPath = path.join(resolvedOutputDir, relativePath);
    ensureDir(path.dirname(targetPath));
    writeFileSync(targetPath, content, "utf8");
    writtenFiles.push(targetPath);
  }

  return {
    outputDir: resolvedOutputDir,
    writtenFiles,
  };
}
