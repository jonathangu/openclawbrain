# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2ed84d12aa80219c71f67ac4b4dc49c9c31220d644ee2203f557cfbb2718f653`
- fixture hash: `sha256-a5d98f20c022a45dcdc79196fa677af12fe3ae7a1d81ee01512e8a79553eb0a0`
- score hash: `sha256-fa704b0dc067c9c58212d6373d78e4b486e673945c1de00da5b3deb4e15a2715`
- bundle hash: `sha256-dc0bdb27240760d342b78940e17430a87afb8c956da588caa9bcb54c45dd2ae9`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6ec46216d6dba9a8e65d3948d25451e1dd699d4ce22c309a9ca79efaeec707ed |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5162e355c3a947c2501213f93dbc48829d5b5e7731446df0548b54b84f146c02 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1bf5e7ea4fd88d1d25574fc243382fa7be462a748ab11c6cb85cc3ed6c5de8a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-04ca54b6 | sha256-c4ae856b9156cab4599abd8c6d3e33b25f685a3c52305ba4ed7363b4efc9cc03 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-04ca54b6 | sha256-1b42270378a83a07406e6146acf06c6c49d29758b79d63fe65878d5649e30ce2 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-193ec91d | sha256-ba3319a99040f8e1f2a8497cad636dd073d0e6d58962c793623cd2b011c4c525 |
