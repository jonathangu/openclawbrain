# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98ce4509785da1d3e9688496a53303f79675442a91eaedda79bdab30b5e6b8cc`
- fixture hash: `sha256-ab905612bd3cc43deb68d413a855b981990f021bcff6e0685761c3af602b59e1`
- score hash: `sha256-cc5b2a14c9ad1c43e306b18124ac44470fc34d8e71420503057a2c98ed53598a`
- bundle hash: `sha256-d69367348dd663868e2f16081f8415ea14011bb4244ab6c9bb6b080010879967`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16539ac70abd2ef9678c6c7835bb8d35322c600e9de7b2b4d16217df707851eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d80e57f0824a72acc13e89276562b7fe121bce1e7dcd12ebc821ac4dba5134ec |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1ed682849b3688b6cbfe5993a35c3176b96ca5ab9d923a40ea3071710be0f32a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-453f8bec2ed40ced1e24fe5207b15ee0ad33dc5c65e62b55ff121a439260afcd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6fd64e3d | sha256-990ba402390432320f8d270e4a0573a60095008c562ae2959cf1850166e6f355 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6fd64e3d | sha256-9580d1cb323690fdb2af8013c597c15df0125d60a51ffe2c69c6d8cecb41bd58 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-de87cbfa | sha256-59f0611a6e89b626a7552b83409912062b340e0b88ba2b3bace3d342b0d3b0eb |
