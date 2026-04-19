# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fae163deba3753cd0d67293e5b3321d395cfd004b4c89b21f894e2a2672460a7`
- fixture hash: `sha256-3c60ee2a81318b7745043c018d8fbe0ff4db3777c3e35e5851f7f1b82123cf0f`
- score hash: `sha256-62736b6c49e4d47a77d44b82ec6d88c7be6573ae13cad7e8ffa476fe70726a9e`
- bundle hash: `sha256-326fb2a5faca5eeb6ebe161e1d258fd94d77a93a22d65f9b2d1c68a2cc834fd1`

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
| vector_only | 1 | 1 | 0.5 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 1 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-579a95936f15845e4a317da0cd7a4ac068a5d1d93f92f4e7fd94d23756dfbcd8 |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-6d4b8089584a52ee71725ef9e1b544824cbfc33e04becc9039e8f87b07544bcf |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-f6a5b918bc8439ab1744a412c0cc90cf7b979287eefa82da49b5a0d8de2992d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-19e74c2d | sha256-2ce5a87b2eac73c5b5ef2aaed52a11b7fba209c1bb1f1b7e5bceb332ae59e698 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-19e74c2d | sha256-3e4fa2e237d426c260c08e4d5bb1fac706d09d64e9c957bcf2c2f8b0db6ff991 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-19e74c2d | sha256-2ce5a87b2eac73c5b5ef2aaed52a11b7fba209c1bb1f1b7e5bceb332ae59e698 |
