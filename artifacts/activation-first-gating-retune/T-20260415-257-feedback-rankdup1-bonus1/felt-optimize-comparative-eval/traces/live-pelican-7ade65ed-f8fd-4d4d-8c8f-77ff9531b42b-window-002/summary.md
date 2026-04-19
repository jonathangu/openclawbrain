# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1cc116d5a5a3e4268eee5081d6d597a83a2afaebb6c2529b01952ad2f45437c1`
- fixture hash: `sha256-ccd8a0f1240cc7f92941ab2c1ede0327e4ed0a420f6a51ec4c81e0437c7d59e2`
- score hash: `sha256-78eb837e4bbdd81567a59bac317fc70f33cb3237ea46ffd8291bb4dd016009c5`
- bundle hash: `sha256-090abe3d33a1d4dacf42aa64d5e317256c6ad0011a8bcaf25808f27caf274db1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a97ee439ef356b4483f5735f34054ec24021480ea2dadec6ac22262eafbebd17 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-c0b9bcc8ccae133e62c1755c4b8881914904f34bd2ed135c40da125d8c9b3652 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-98a8ce1572c14e5424c6bc50edba62d7669a0d2d778a2e4c9c2a2c6a03e88495 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3e6745c8dc874d9dcf70337dbbb7d970a3ba53f9a38107066c079c638680c059 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-f1d61caa | sha256-8ee53ac77c93b32a62a8542873edccf274d2f939c91b447bdfde7ca6a11b1889 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-f1d61caa | sha256-75ef67f52d7294ea3eec1e6b1d12f83347d777098af7530ee83158115f380cc2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f1d61caa | sha256-7b5cc2456593a9e21539b04ff2932f6adc24c7ff6278dcca47b95feff3cdb49b |
