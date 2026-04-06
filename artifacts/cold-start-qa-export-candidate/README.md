# Cold-start QA snapshot export candidate

This directory holds the first real snapshot-backed QA export candidate path for OpenClawBrain.

## Status
- **Review status:** `under_review`
- **Approved train?** No
- **Snapshot-backed?** Yes

## What is captured
- HotpotQA dev-distractor snapshot bytes
- MuSiQue dev snapshot bytes
- Governing registry rows from `data/cold-start/registry.bootstrap.json`
- Derived route rows compiled from the frozen snapshot bytes

## Generated artifact
- `under-review-qa-export-candidate.v1.json`

## Notes
- The artifact is intentionally honest about its state and does not claim `approved_train`.
- It is a concrete intermediate export candidate, suitable for review and follow-on approval work.
- Snapshot integrity is checked against the frozen registry hashes before compilation.
