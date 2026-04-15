# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0b1bca8dd8d311ca0f474a7d9deb1193514002f9ff0a549efbdfe8a579f7a8a7`
- fixture hash: `sha256-693be8683846991e932bfa4a0d12773f4fe199b9445b669c78493c22255f8959`
- score hash: `sha256-593992e93ff760415c801e955f64af5e7676776826425f49658935b9a315f91b`
- bundle hash: `sha256-2215483b7ae03aa568ba9c33642a1f94522c550995884509fcfd5ae671e3feb0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a936bbb2a6bddfe389caa1010c92a0418532436fd2f50651530e961a6495d56 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-86607615ae4eba060b2163f944f9af305ff56ca2d058e725a6b0ce57fee993b4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c345d9543254db51c588b9a3cac431989504a1ebda714715f603101337ac5cd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a4a62eb1d36768b59e2598bc6036d98e69940f2aa502f66bc2c0eebce157125f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b380043f | sha256-80e1b6d5097634c6fbe853e6801e26b47556549c1ba63d890241a004107f7261 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b380043f | sha256-17070bc5c1c273c7176820c61e7225c5be499b671f8336c60297a828fdb33519 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b380043f | sha256-80e1b6d5097634c6fbe853e6801e26b47556549c1ba63d890241a004107f7261 |
