# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c591f5c65a17f8b728581b0a64e58f54d550808c4d6d87e9681919456c4e7956`
- fixture hash: `sha256-7d10d338cf842d955d253c17c711f61a919941b05a2192e292201851e3214a2a`
- score hash: `sha256-489799966550965ce8a8387b67ae735d95d63d28d406f61316698c5bcd3cc9d0`
- bundle hash: `sha256-242946857faaef5fcae10b2be6a1d1d856215af74f739a4f6398360f1066a232`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dc716d2fe029608c9fe52ecb8defed0c2de7ebf60cb8d8503f70a55a165b4d33 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3c3e7fbfb650d15a994b20bc9e5d6066ea84fbcbe40c99036e507a60bf2829b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5f0c5c2b2fd604cb41f070f7a2b5f3e2f41fc861bb1fbd1f803d915ea5d293ec |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-697becd1a5fe1b73aee3d21a536c5321217fdb1dd3de4dcf29c43b2b61324c0f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c123d979 | sha256-ac84de86c4af4f18906321da2632c015282eef12b02f80718f7462b46627f4c8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c123d979 | sha256-43b2dcf8429b6c2936c55b9a5c52822ec6cd4de85a993dc1da2b1a04b585afa5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c123d979 | sha256-ac84de86c4af4f18906321da2632c015282eef12b02f80718f7462b46627f4c8 |
