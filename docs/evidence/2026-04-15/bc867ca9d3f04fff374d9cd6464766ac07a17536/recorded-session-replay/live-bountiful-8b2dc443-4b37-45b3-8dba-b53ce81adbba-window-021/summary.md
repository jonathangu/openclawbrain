# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-91a0633d0820892929ee483cd601c44d030e606ed764348767cd65eaee89c88f`
- fixture hash: `sha256-6d906de02d191088a0de23c25acd9ce0dafee05c1498a2c021d3693ce5ce2c41`
- score hash: `sha256-1805b8ea27de048f5c32fd0fee877e31358fc46e42f5bf43b5a2680cc430adf7`
- bundle hash: `sha256-9abc689ae65a1d3db308c776f3a3fb06f69515f4788d95e310ea4a18ec77d9fd`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-028ea247345f633c6b07542e5aaa8c0bafba6aa7cf71e5143111b89053a70408 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b0e541e817ec468f857b1da673b433194267dd863866b31efcdc77bedfa1df4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b18d0eaefa706a4c2ff34a92f3668d319b7bd4c52cbc3112010a081fa0ea4d6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-83cee62d6a81d5404cb2b19104130ee6833219000500f2ff8fb5f486add20e64 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bc4a442f | sha256-abb8eb028ca223f79925c7e42c96b089f0a5bcd708de622a894befa7233d4502 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bc4a442f | sha256-dc01e8f3015130646f530b4fd45d5ddd28b68ceebd8b2e9bdab6cb474cb2c598 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-bc4a442f | sha256-abb8eb028ca223f79925c7e42c96b089f0a5bcd708de622a894befa7233d4502 |
