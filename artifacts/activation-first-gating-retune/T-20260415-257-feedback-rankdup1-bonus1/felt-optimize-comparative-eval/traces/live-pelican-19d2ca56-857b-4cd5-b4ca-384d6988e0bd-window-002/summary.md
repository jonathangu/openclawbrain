# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3587f37a965ca48e3b14fe490f41619f4a64d9248201fd791da49328673f2fd`
- fixture hash: `sha256-72b30c69deb757e882e827610fa9efbae23b0f6f41cd081abb4ab731c8f4dc73`
- score hash: `sha256-3b1bc220c3108a515ec43e6a90b573bfa9801a2e78c9a27cfa9cb968ab9f7d9e`
- bundle hash: `sha256-dff7b3720286b1c61cf62788a95bcdaed26958dcdf60db7d2db873218641103b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9cbe4dccbf152c20742a7a0d9f6d7f345aa7c7916722159d8bcf4f7a084bf5a1 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a90df3acbc68b8de9a6f91acc99c16bf68f63256672b635a700959f34c7b5528 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-74a5605726473a507e5caa59f7062dd51199dc5a64d6acf56c3245e0dd75e885 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-54992a9e1dde62c75d85c9c050c7c6d25e6476e7d75096d7b1627538107cad76 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-5101e73e | sha256-318abff503223369a2528b7d4dd17b794295775b983de05fb9675d18be300f21 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-5101e73e | sha256-a483a9379a43694b2c94e419b771695311cb91e476123fbba47b93f9c402a9d4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5101e73e | sha256-38bbbd09b0316bcff37c99f2db027d07dc4e070bbda95a8461c23469f0529441 |
