# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2ed84d12aa80219c71f67ac4b4dc49c9c31220d644ee2203f557cfbb2718f653`
- fixture hash: `sha256-a5d98f20c022a45dcdc79196fa677af12fe3ae7a1d81ee01512e8a79553eb0a0`
- score hash: `sha256-1e2471fc74a8cec6aba9a289e63cd77954fac58ec057dc3ef791d539648f1ff8`
- bundle hash: `sha256-970e4164e04fba4fa33b851dc0d05bcd09375e44b914ae7e1afd225c8d724c70`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2d0401b09442b39c74248dfb10f1b77d9b52939def1349d2c685ecac4f520b39 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-212086c6c59d1db4026e9bbce58bd54e9f05a680f7511c56ea5798ef43d50410 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33409a809785704d044c7b20b0874b15067bdc8bf8228cee7fc1d98942be7177 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-13a10952f908b993842c0b285341f8c77ab34bfa3c21a088d705721f6ac3c02b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-98fde933 | sha256-4493949cbb796bb8789adc6f488713b4691cf7d3122c9f375629ef9793b00df2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-98fde933 | sha256-c0bfdb483658686881637ad770e5dd328c4c250fce8e0d04b84ced35e3035cae |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ad725d9a | sha256-c04798e9458ddf6a7dd562f323e5c362e55401bff709428114f14e603afff5dd |
