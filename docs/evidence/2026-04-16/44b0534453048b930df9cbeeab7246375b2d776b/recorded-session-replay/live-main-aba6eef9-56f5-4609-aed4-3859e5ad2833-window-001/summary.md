# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af95153d1f0a3be68251ed9ca1c6eec687f3524276f083ded8a5b5ed5deb8173`
- fixture hash: `sha256-e08d20ecb487c4dc497560c31f0ea6c918c59692d8f02eb17f1047383fc56246`
- score hash: `sha256-489254c87089e09ea51ab4c09cfda997f509b949fd29ce9ea3172f4ea748cb2c`
- bundle hash: `sha256-c685f5a9a77fb6bb7e8d81d3795bac45c1b8dfb8ff0686444e48cb604101b555`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af28d7bfd3a7d04b34389f31647ac0f041f3d52e501bb74630c37fbf1936f421 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3a4a88070127026e2e07c0da69fefdb5202048fc68dae53ab3e521b8c1e5108e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-182079ea7fabcc7b82149406e9f559b3a153be5d9a29f0cbf94a897a641a7994 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-204ac067b4bca117df6fbb4de4591f0cf02709ad8e635921924020d33b8a4c98 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b28d1dc7 | sha256-f298b727f98c9d17c5103c76d2a4a17d50d526001d1d7b2fbf4b15077990958a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b28d1dc7 | sha256-e073dee0c055aa710566e900d107233adca209c8110306243c738fd440dde0d9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a1f4ce42 | sha256-e8c39c8cbc4c7f259181a5cb0f5c918777588123751f9f910f85bdd44939501a |
