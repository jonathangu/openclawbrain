# Recorded Session Replay Proof Bundle

- trace id: `live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-035806e6f9bcb3753f58456b70d56dc8a01f4abf60114aeaf359384806f6c24b`
- fixture hash: `sha256-74bbbcc2ba3e23b87dadc56cde438b46daa30c3743245ccd0b40d24de1249370`
- score hash: `sha256-804b7f101ba25bba7b56077d00820813d4b07736146931c78fb78ea4785e0a9f`
- bundle hash: `sha256-1d65816f9dc40935fd10f3a7f7142a307458ea32b1104569470594132ca47afd`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54ce8361598e1b4080ba115badc91e906441ece2076bc91bc1f9f28df2706034 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-df88ab1b32dc2c4a4fae116f533ee38ae37458d0bdf351898cbbd8aadca33f6e |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b96009499753d262e7d3bc1e6fd4b4143bda687f53120e42aadb543a3961562 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-fefdc257182ef3a0c379a0673da746abf5f5b641550c6f4b789ece227c43eb34 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-1c3f2d0b | sha256-bb3d1703f38766257caa84b85fe93be29a60d7ea3663dfa88ced9efe01851815 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-1c3f2d0b | sha256-c43dc2d1ded8df276b0e2554149bfb988ec64e2bee04328dceaee409c49032cf |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-ca00062a | sha256-c1e2bff18567b763cddc5e689926249076b586160fd6cc0193b3b9a1b80be9f4 |
