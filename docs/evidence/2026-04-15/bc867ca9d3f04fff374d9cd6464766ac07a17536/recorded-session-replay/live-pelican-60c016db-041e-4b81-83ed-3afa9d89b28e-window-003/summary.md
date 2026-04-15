# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-954e7370fea342c102d12700848d801f9cb4f766c26a152991092af096940dab`
- fixture hash: `sha256-71ae1b688e319cb6bd41b60a17bd7289e838d68bd59d5831fb46d3db379a64f9`
- score hash: `sha256-ce01203531afe195ed08b0ec9fd44f412a5e36313555a8b2ee1a1dd04cdeb0fe`
- bundle hash: `sha256-fdc91ec3bc7280193f0acf017d741b5557bcc69da6ab88e84b830f9cab5e882a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c508d0d953d2e20d8464c0689a6d8d9a5c0442d5d485367bc8ffc01689888e09 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a5b54517adb4a5cb423711d1c30f2d6a3af53c7ee780547c0e1f67b4d5a6d37d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4df151b9ccad5948581bfe8b9c73d185a15796bc8618abb3e10aba0a40dce95f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2e51d2e488b43914eb8ba54ba2c1f940a6ca6bff9107fc3bdaa7270feac60083 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a77783ee | sha256-8576fe9e96bbdb236067c9c9551d51c0615b4bd3464fc5902c78e82fd26c3987 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a77783ee | sha256-292c33a794f3b1f6f93b72676675ee6e8d222583f40394c0a7580e46aae7a31e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a77783ee | sha256-8576fe9e96bbdb236067c9c9551d51c0615b4bd3464fc5902c78e82fd26c3987 |
