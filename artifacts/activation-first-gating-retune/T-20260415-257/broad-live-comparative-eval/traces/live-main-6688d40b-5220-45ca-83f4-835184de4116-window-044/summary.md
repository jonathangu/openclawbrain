# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-69ce0c4e11baa36853be20e1ca688e734c8855423d37366857eb233deb6e9df0`
- fixture hash: `sha256-c3a333635db8e86be19e8bf48de8cbd13aa6939830c506cedd85267cb0e9f51f`
- score hash: `sha256-d0ba1609573f9e01071c0b2f6a16ddcafaa4db2e928009a0eee9bfff64ffc466`
- bundle hash: `sha256-e434cca32d6b6e5f47d006660208f970c75fddbe755c30b365b136753ca206bb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-653724b1c50980255f17a34150c96cf9693658619075d0cdd8b7b4b447cb2cb6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-43290332d2f8344d0643b4081b788e9e0a36432cb92c129e367fada9d6beff53 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8860016896f3c31b23c662c340601253e2c20f1eddd465061200ce7e97eb7ca9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-bafc03edaad713a52c0dc7150629e51332d54d3c39cc7adc1a05b11096cc359c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bee4abd7 | sha256-211114d608c2c75f95b4b70e8b96b8ed81200f358c719ac7c4401c6c42c60c77 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bee4abd7 | sha256-300984ce8a051b1030c05da196c629bcbca4996caf3dfb8635c05ec5b12750df |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6f2ae30a | sha256-ae374a6d23616273dd0299ad314004b34ee14b63e49e9a6954113c6bbae7e5ba |
