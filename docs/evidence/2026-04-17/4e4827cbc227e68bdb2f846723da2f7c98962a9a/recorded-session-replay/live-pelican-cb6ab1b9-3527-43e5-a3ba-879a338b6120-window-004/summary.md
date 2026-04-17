# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f94a31a75f6f674deb4ed72bb4e73c45b90c17561480d33d8d146b93540cfdaf`
- fixture hash: `sha256-55ad28e1e1c0e357b90d71c5a61455a338c2e0a4ef3a7f6c092d3616039ed272`
- score hash: `sha256-536376415c2689d0bc9ea49bb1a720cb43d720d2cc9a4fcd6388a29473f5eb19`
- bundle hash: `sha256-969dc9cecf831b828ada69a30145c2a64589ba2bb6d58bb7798cdbb8fc5ccd02`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a7e02cf5e88271092f868ed8daefe51bb787b99a9e0166c0444d9f0e9eabb76 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e61239bfdfad3f19e578dcebb6622d3eddb291aef4d9f936c368b34a731b0b74 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23f6f1f84ee9e522d5dddf516f069eb5097439ed78c3cfc7470b8b4de29a8a5b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a9bae01856a7d20567534bd1c1eff931d09f0934c650f327af502bd18ed86e19 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-eb1e715f | sha256-05803ef8173a043c3f691041150882375ad427a911af5edbc1cee8261193b24e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-eb1e715f | sha256-a22a31e8f5604e0f99ee41af771bb513f31a2736e0cb11450186e6ddf22dd52b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c6bede24 | sha256-66ddfed5de97ecc9e1c6d841f3a7406c50cf2d7b70d7cac1dcd8c864d4ce2504 |
