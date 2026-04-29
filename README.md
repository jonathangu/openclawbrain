# OpenClawBrain

OpenClawBrain is starting over from a blank repository.

This rebuild should move slowly, patiently, and honestly. The goal is not to recreate the old project or make broad claims about being generally smarter. The goal is to build a small, reliable intervention layer for OpenClaw that helps at important decision points.

## North star

OpenClawBrain learns when to help, what tiny piece of context or workflow is useful, and when to stay quiet.

A good intervention may be:

- a specific remembered fact that changes the next action,
- a tool or workflow choice that prevents wasted motion,
- a restraint signal that says not to interrupt or not to act,
- a proof surface that lets the operator see why a choice was made.

## Principles

1. Start from observable runtime behavior, not inherited architecture.
2. Prefer one useful bounded loop over a large abstract system.
3. Keep claims smaller than the evidence.
4. Make restraint first-class: silence can be the correct output.
5. Preserve operator trust through clear proof, reversibility, and boring reliability.

## First milestone

Build the smallest dogfoodable loop that can improve one real OpenClaw decision path and show honest evidence of whether it helped.

No old OpenClawBrain code, artifacts, docs, or plans are source material for this rebuild unless Jonathan explicitly asks.
