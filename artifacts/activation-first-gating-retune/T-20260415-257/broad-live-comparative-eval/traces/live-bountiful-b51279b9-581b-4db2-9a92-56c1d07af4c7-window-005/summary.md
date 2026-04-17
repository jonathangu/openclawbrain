# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fae163deba3753cd0d67293e5b3321d395cfd004b4c89b21f894e2a2672460a7`
- fixture hash: `sha256-3c60ee2a81318b7745043c018d8fbe0ff4db3777c3e35e5851f7f1b82123cf0f`
- score hash: `sha256-4b9846f088912fe4852a4a986743a20a519b942a248816bd1228d1c7495e4288`
- bundle hash: `sha256-8d02ee8c32fcb20c0bac787d888462cc198a7c17ab72b4920f1f760b040462fa`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-50883b87c06b74bea3cfb62ff5f8bc47778790de2af8fe623d57526aaf2ffdc1 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-2cfb520eeba32c8c2f7d0159fc9c3d1f374de15da89ecd3ed7fe3a0d79d93b77 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-615ab415d756e07c7f58a16b15d943009e24a0e75c46820f1d5f5139ccd350cd |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-9c3fb33ef735ffe8c9f88ec09766fc90b2d51f928a460db921757ae5f6988999 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-1ad01202 | sha256-16544a6ff049cf7a16cc18fe35ae17a4700c47f1e59da0afb90bdb1ea6eff1bb |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-1ad01202 | sha256-56ea6feaebafae611f584692014c4de086bddcc2d89c91a7f1a80d2436753053 |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-19e74c2d | sha256-a020750176f8a10feb0d420154178b00b0186db3e6053b494b7887977a69c23d |
