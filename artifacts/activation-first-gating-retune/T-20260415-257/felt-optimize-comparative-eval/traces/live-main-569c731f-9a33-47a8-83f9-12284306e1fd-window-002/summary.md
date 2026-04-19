# Recorded Session Replay Proof Bundle

- trace id: `live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d27db54f25bb1682fcfa202523b5f1c6efccc7e2753d8e02e54ba11f6e3abbc5`
- fixture hash: `sha256-0e0db1f3540c6bbafcaa45e48b36b0aa0cc986ef0dddf4d7e13951d4b175679f`
- score hash: `sha256-09611af478d736fd74bbcbb5a94b125bf11de2d95a38990f4f30cb85ec47992f`
- bundle hash: `sha256-9f86ec665069a5f502063ef18dd46e3e63c0a74f08e80797bfe22ad1bf771790`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-71d12c78bbd92c17749c2ba921bc24d7594735564898b2d4c08d5a5f8badb93b |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-d13f0486166cfd5c3c8da3886b8be045e9f967dcb1dfcb11f56783d8855493f1 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-36cdff6075e829e0e705093db75d1352f883b252f2477bd2c64796c7f1df54ee |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-86b8bc9f0a872a96c4210309f7be9e18fceb7d5729fdc14e5754f4dbec4e8588 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e4a902de | sha256-a50038e165bfe6a16b7ab21df7d3ebdfe98a811289a3517ea91613371335f0a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e4a902de | sha256-a50038e165bfe6a16b7ab21df7d3ebdfe98a811289a3517ea91613371335f0a4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e4a902de | sha256-13d92093b859a3a6cb391681101dec735bf83bfd1300f5f5286519b2814155b8 |
