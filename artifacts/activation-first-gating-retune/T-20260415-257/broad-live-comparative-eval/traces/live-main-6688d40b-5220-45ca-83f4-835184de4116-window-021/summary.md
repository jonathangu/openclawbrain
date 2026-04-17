# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13cb1bca8722ed39c54b48c9d170af84a0229da5a1be3326ad569cdcb6c86e93`
- fixture hash: `sha256-3649ce5ca20580b372f2a2005a8164ef24eb19856bac1831bacfdfc2aeeebd5b`
- score hash: `sha256-19cd7428d29e2c124444434768ca5ce1598f1044ab42ac83490419bb01c581e2`
- bundle hash: `sha256-05f0fd2bafc57578efe3cc8f023a7e55941626a0cf915a4462b4f3610d6d6dba`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3fcc253b9510f29399fe22001359326c4d47b1fc87658fed51c53d2aa08bb9eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f3c16429d363accc383fbb251bb40495ef8da22bc6e062e50d57538cafbd9d0b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-af0540c21cf5b2791f96490456d722d79ea4ec1ec644e5365820e2e95065ecab |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5c1b6535a047afbe748a77aacf087425e9047857da75e1a95eb78f54c7272c05 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-40244dae | sha256-a5e7b99665390a97e90b857e705822c56ad9c9da25d6b043c38d8e7e1621eb69 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-40244dae | sha256-61bf51a3c86492011fded3129e139e324492fac2e95fbe7630991bf1d1c6cdaa |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-2d254cdb | sha256-d5275a012d04e4be9c26d080567b875cf13e9a1427133586d517edb81d8bbde4 |
