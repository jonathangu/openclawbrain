# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af95153d1f0a3be68251ed9ca1c6eec687f3524276f083ded8a5b5ed5deb8173`
- fixture hash: `sha256-e08d20ecb487c4dc497560c31f0ea6c918c59692d8f02eb17f1047383fc56246`
- score hash: `sha256-c87f8a42d28cb7191e232d3e77ce609547c6a4c940e697ac0e943ad6999bc1a6`
- bundle hash: `sha256-5b5dd020ed33a9c1b1f6bf4fa82bde8bb80e4cfa58af2b51c2439c65291535ca`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af28d7bfd3a7d04b34389f31647ac0f041f3d52e501bb74630c37fbf1936f421 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8c76033f6c5fbb7e96dabea54e34504b25e3a18d52eb5919b464330f36f8acd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-00f671d6595626e424ce311d8a4bae679949627609e30c238339ec6ca94653f6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-77e98bf2c175621898b7e7b4cbbc3c390a06f540e49a1c1660901f70688ea15b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b53a0509 | sha256-0ea7aee6c4be3791a16bec0b9f54ac98b8a0ca4b87b5a3f7960382535fa2791c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b53a0509 | sha256-276b86d547a9d5ae4f837260b582bc8dce77ca2b98eeb25a9444699273145b6b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a4a1b584 | sha256-00f9c585af569e0236f972ee777942da452c499a517b6c72ac2ed3d705c62b70 |
