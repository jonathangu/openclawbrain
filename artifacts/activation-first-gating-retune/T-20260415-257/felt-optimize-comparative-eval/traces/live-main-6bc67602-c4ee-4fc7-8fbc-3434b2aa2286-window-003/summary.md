# Recorded Session Replay Proof Bundle

- trace id: `live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b093cf106a437e24ba93bbfbea56317e62afd65cc282953b847c0fec17c90f`
- fixture hash: `sha256-f186a663337b28243cdd6e62a9c63e0bf0678cf05202237e1d19a1f17b82f110`
- score hash: `sha256-1b34a7ee5e26b85b5979b47d640efff37ae787efde55f3bdf2856dd94880f212`
- bundle hash: `sha256-728029e43b84db92313ad588b3b2a22c7871e2e7d277d250d2bde8834ad25cd0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea127012751163ce5c5c7b6a51409b045b05c15be13611d375e11b98fe528366 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a55a6e3a452bfdcba60e294a1f1e2e3f26f7712593d21a72f63f54daf71c68ba |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a95c25cd686245820a72a4bb350f839f2337d70c4a0d79423bc145bb8d52fbe7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4e391c771f2de9314eaffd34ddcda2c9254e9407cc8508e04bcd6593c3a4ddc2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f68d3e40 | sha256-bcc10d44f413bbc16ccfa163f5f11b3ca6fea788526039a263cd1829c5fe37f9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f68d3e40 | sha256-a5afc1485df73452aa8fdd0abecfb527bf634dc29e2a62451ac2bd34f2ea3d1e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-48fcbfe7 | sha256-563038d7c4999b9933150b9a72c64605dc581dab30a649e23ed852a51d8ce028 |
