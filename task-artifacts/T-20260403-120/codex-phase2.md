Root cause

- The live watch/session-tail export paths emit assistant interactions as `kind: "message_delivered"` in the packaged runtime surfaces.
- The async teacher labeler only admitted `interaction.kind === "memory_compiled"` when collecting candidates.
- Because many real exports contained `message_delivered` interactions plus empty `feedbackEvents`, those exports were structurally ineligible for teacher-labeler candidate collection and collapsed to `no_matching_interaction_text`, then `no_teacher_artifacts`, then `teacher_materialization=noop`.

Fix

- Widened teacher-labeler candidate admission from `memory_compiled` only to `memory_compiled || message_delivered`.
- Kept the existing narrow guards intact:
  - candidate still requires a matched serve-time decision
  - candidate still requires a non-empty matched `decision.userMessage`
- This restores non-zero teacher-labeler opportunities for session-tail/watch exports without rewriting the exporter, matcher, or watch loop.

Exact files changed

- `packages/cli/dist/src/teacher-labeler.js`
- `packages/openclaw/dist/src/teacher-labeler.js`
- `packages/cli/dist/test/teacher-labeler.test.js`
- `packages/openclaw/dist/test/teacher-labeler.test.js`

Tests run

- `npm --prefix packages/cli test`
- `npm --prefix packages/openclaw test`

Remaining risks

- This patch restores eligibility and verified artifact materialization for `message_delivered` turns, but it does not guarantee the model-backed labeler will emit labels on every export; sparse or low-signal turns can still legitimately return `no_labels_emitted`.
- The repo only contains packaged `dist` sources for this seam, so the fix was applied there directly. If there is an unpublished higher-level source-of-truth elsewhere, it must be kept aligned manually.
