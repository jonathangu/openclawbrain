# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae32f07fbe5a45648ccbd2d0869190b2cb3596e4fc7c3e1299ef7f3819e0b838`
- fixture hash: `sha256-b830296ad0e542a07399e1e822eb8c0691a725d5f9135851e63c87d0c1b12ee0`
- score hash: `sha256-865b1faa0f4cb99640a3c68ea132ab3e9cbb0c72d9b634cc017acce8dfa8f575`
- bundle hash: `sha256-ee5342dbbe269d998b5c08fe7637e80853b7780ee8bd01559419b377e399007b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f883a174ba2c6d9b8e46baf2069a63ced4f1f39ba1f842535f04648f9481662 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fbfc59063f2d910023c816fbee76b9d85d2a26bb76f09a95c08d7f4b55e556a3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eae8c86a980c511e32f13031614c1407d7563f45dc217744646f5284ff3d4601 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a6627489f962077f57b5225caee22b75f7a239eeea7a5dd25f921ec8de14b6d3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c520fee1 | sha256-1c1c8ed301faaaf4d2d1b9317eebb9848777bcdef7284a3233ebc409aac4df61 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c520fee1 | sha256-d0dc7c5cc090c049b545ddc56fd986234a6682ddfcab9dda1206d97e2669460b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6616755a | sha256-18dbcdf66e948247f0d1707188d95a1a716702bd1d59e7d7d50484c78c763957 |
