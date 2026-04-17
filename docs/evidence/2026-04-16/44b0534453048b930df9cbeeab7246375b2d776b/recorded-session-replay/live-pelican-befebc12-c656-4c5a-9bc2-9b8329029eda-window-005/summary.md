# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae66dcd407454d64cc41d28f61a0e11b77513e2814da48c47efe5c8c6e3c8baa`
- fixture hash: `sha256-8798537b3abe1b5c15bce4787c7758c4cd08e15c5c204adce3a372ff88067693`
- score hash: `sha256-2e512aefc10c925399135a430f2acdff79de00739d6a80841fba9edfeb5ceb3f`
- bundle hash: `sha256-8a3b19c87f825c1e44b85488484eea7d9ccf608b93607001021ce0a2fe248df4`

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
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2631460258b8349bf5851bc29e43a192f583fb77925024687b67768874305033 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cbeabb266c8537da9e54b5cae40256ce144bc659467294de0110609a27423bac |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0e01939d5edf0373bfa5de66de5b682f85a860d642c705605f2617dbb1bd5591 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-337e2da035fd9d9730bee5d266927ced19b3a55a8317d49736d8d3a63180e8af |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-19991ac6 | sha256-1c862ceb1d29a8bd59aa3c07e71800ff103f3bec6ac090a5b6bf90bbcb24fb82 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-19991ac6 | sha256-61d203dc6d398ea09ca34a4b036894b82e17e09a54809b2206179c2a7d55a4a6 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-e0025985 | sha256-c144ea1553a0dcf3ab19d589d1d177112f2be762fb2e1074b8bf536cd9f80f4f |
