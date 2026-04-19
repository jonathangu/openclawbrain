# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67dcb8532f54fc5f6268aaf2cd959dca249a5c5832d88990a647435a45026ce8`
- fixture hash: `sha256-53a4c9afb3f28aa87aa1b17aad9db78e9f58b7b80cd2cde3904a19a0bb713c36`
- score hash: `sha256-d95def2921272a6b69cde444b0e918440cf808153351db5519db5d6353af0461`
- bundle hash: `sha256-7ca4baedd626265cf6ede0a790b6150f985401ad1953d0128fe1b13087f4950e`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e721a4b8ae2bb9ec3999c909e1329c35bf2b76bcb692645b1624780e9c7c3c31 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-ed8af1c49824017867b4ee60264c36994bd54346a89e7a2197edcad07ce9dae7 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-1a26d576d877cbee85a162d66174fa22aa0669634a99f4aba2dc19d5983e0dda |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0f62bfa7ae134a596e46a11ff3bc6dab809fe523c2be24ac8e23fc993715fe5c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-f827eca5 | sha256-2ec6500484720078fb961912a8d323a47a8a746acd058d4a83012d267bc49817 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-f827eca5 | sha256-b6d475316a5ab15f9c8f3b0d8c21bb45bb2482612df0ebb2900aea032fb00150 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f827eca5 | sha256-2ec6500484720078fb961912a8d323a47a8a746acd058d4a83012d267bc49817 |
