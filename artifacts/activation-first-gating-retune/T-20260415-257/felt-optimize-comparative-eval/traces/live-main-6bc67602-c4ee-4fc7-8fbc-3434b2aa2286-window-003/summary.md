# Recorded Session Replay Proof Bundle

- trace id: `live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b093cf106a437e24ba93bbfbea56317e62afd65cc282953b847c0fec17c90f`
- fixture hash: `sha256-f186a663337b28243cdd6e62a9c63e0bf0678cf05202237e1d19a1f17b82f110`
- score hash: `sha256-1bc44acd9d00e8b8bce6fe2686587e6a0391ae2429b136ae875001acf5c1e150`
- bundle hash: `sha256-7c2bd8ba2b98e2ac5acfe61faa22b3f2bc06f9f1a46a9aeb8eb9637436b31ffe`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3e641dbde4bb0ba78103e94540244ef8e79f7565f730eb500bc9574f32a1ccc3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a1b038b2668f994db5a90743280d986f9961742f07ffdac045de767a52cecf29 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a54c0426fb1158f2b83e602f78d2789c92bc4125e1f8d24f6ed37a3d53f7d8a3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-578d20ac | sha256-3b3f3b9955261f6e3773971432a329d02c2489c0a981977096d6601c20acdbcc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-578d20ac | sha256-f37ecd698525fc2a31209ce7faeba56e66408accfcbaec8d4fe441be84eaacd1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a9fca253 | sha256-fc7dbeb3852b0294897ada84bacaf4cc671c57b6c0cfdee9fd690c1cd8fda200 |
