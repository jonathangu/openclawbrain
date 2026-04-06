# Cold-start router approved export fixture

This directory holds a tiny checked route-decision export used to prove the
approved-export loader and trainer smoke path end to end.

## What is here

- `approved-router-export.fixture.v1.json` — synthetic curated export bundle

## What it demonstrates

- a governed registry entry that is actually eligible for `approved_train`
- a second registry entry that is still under review
- route rows that are filtered strictly by `approved_train` eligibility
- a small honest fixture that keeps the smoke path realistic without claiming
  any external dataset approval

## Notes

This is a checked fixture, not a real external dataset export.
The approved row set is synthetic and local to the repository so the loader can
be exercised before any real approved exports exist at scale.
