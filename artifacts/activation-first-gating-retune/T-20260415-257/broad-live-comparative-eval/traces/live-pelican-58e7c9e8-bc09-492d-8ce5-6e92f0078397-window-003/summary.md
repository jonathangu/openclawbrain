# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-529373cf8f7314054ee5a9938b5133a303e70a9153c03b373cae4ff852f394c7`
- fixture hash: `sha256-c16690eb3752325552dd8dd957f6a57c852c3d697d1ce7463c9556556d92ca19`
- score hash: `sha256-5da40ab29069eeb25983211ad19a8d55e7c1406ed5b60d09cbff763e79f4354b`
- bundle hash: `sha256-9c3f5cbe037f59b73e9221e7952a37d10fca8636a68e07dbd204d7a9296258f4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9641da18215ca2d07fc313a19aa471e30d85d3a5754d470ceff969f5080d786d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-594d122537d9e61fbfb6ef8a4896873f6e9b18ae60133392023e33b0008cfece |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d9fcbdd08ccbf486cbebf537df5698b8c0259d81d5a3070a91036e57fc68c0f5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4996667f9c80cd8ce0bc93e6801635b5b72d2ad3b07b94d0df2ccc1de741eaa9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-782c05e2 | sha256-c255e77146636ae050cee14e49480df4d3b1684544b6b2e49650f740e737f404 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-782c05e2 | sha256-a9fb9cb01126eb1ffd56d2efeed1321ae1358439f795e378e042f5e385d419b8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-29eb964d | sha256-5bda4e5b08a0f8feef6f57078fa438f84568d6184c4462c4030b04682cda7a05 |
