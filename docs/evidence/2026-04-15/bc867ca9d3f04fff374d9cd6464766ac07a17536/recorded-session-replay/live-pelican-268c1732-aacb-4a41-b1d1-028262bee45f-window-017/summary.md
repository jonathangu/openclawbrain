# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-41fe4ec7878578e538ea87217d0f1ff26ff7c1c5009495fddc6dab6258a2bbbb`
- fixture hash: `sha256-49ec3fe73495575ef4a5edbb2b2c58d86b67a86df5a7ca6830045265a7717b0f`
- score hash: `sha256-f3917787150040bf8f259e860703403e843d5f579b74582a0aac0c449b63586d`
- bundle hash: `sha256-26383a98153ce359fec7a6fa9ed0d3bc9c41c3b50c4e4fdeca4d479e5101e0d1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5a96d1404eefca9da775b3bd1f6864e8e794d06c6c90d16aa5e90455db3aa7d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b2005cc447faa803a605bde2b7f2d239c0351951e3e46c1486c2e1a112b7baf0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-90fa3504ace4372ac0576f15c85301f4d215fb19b1203aab0d80aa43f5e3615c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3e60a84650f8558deaca50159ddbf40024e106cdc8cfc736e9d6bfe4cb015236 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-66ee20b0 | sha256-0cc6662964004accd130e81eb9633d038c82899391c0d381335f630deb6808f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-66ee20b0 | sha256-696fb6fb7c84fe45a520eb7ff1f7d5f82edeb54d6d1332a9e5c4bf5990ab5b04 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-66ee20b0 | sha256-0cc6662964004accd130e81eb9633d038c82899391c0d381335f630deb6808f7 |
