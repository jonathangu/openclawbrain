# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13cb1bca8722ed39c54b48c9d170af84a0229da5a1be3326ad569cdcb6c86e93`
- fixture hash: `sha256-3649ce5ca20580b372f2a2005a8164ef24eb19856bac1831bacfdfc2aeeebd5b`
- score hash: `sha256-b49538d67d87920c18cd9c1d149c89bdf6ce9925bb234f97949f7233344e1529`
- bundle hash: `sha256-ddb47f585737b01643edd6ac543d8042005fd3f7add4d72060d5b74c83ea6a8e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3fcc253b9510f29399fe22001359326c4d47b1fc87658fed51c53d2aa08bb9eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f991720354085f54210a2abfb0b23bb01a1d30d19c3b738af70bcc2312676744 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ca043ef0c5c5b50b759c8f86350531c5268a7ba6a4f655a1c4813446d6e9b7ee |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-bbc073a5ebedd2dce4761034e3d0e5500ce5c5eeed298ba4514ed7ed19e6a001 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-523b51d2 | sha256-46cf62eba13370159f2bb0ee73eb50e451399dd44e7d14580341122be7ed44a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-523b51d2 | sha256-daf5311ad25018e5748fd043b675e67d248f2e52f3a7a38195b6602b7cc850df |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3f3c50ff | sha256-fa1b7bdb5b58c928cb0d13e693ca8237fd6dee88ef026b0fecab043a1177346a |
