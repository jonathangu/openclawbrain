# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-838b9295d0df32bf17309a7744670eaab3129f24a6dca2ca9110c4b4940f8ca0`
- fixture hash: `sha256-56f7d90cfb38f59327532bc9b6beae4801650c72b03cf0a3e492173ea24b06f6`
- score hash: `sha256-ee5e18a81289e34aa77601e86b393be065236bb0d4daad97b3862ad67e1f04c1`
- bundle hash: `sha256-8fc7fbf3281fd146f1f0af84ba0d2b0fc7b01db5837046a864c6472a7dd1e5d3`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d12703af710851e5a23d60b1d20c78b1a6044ead7e09a16f607df5e76e23db43 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ed7550db5541daa394d3ea521840b582eecf4a3901e18250524cb6b33330d4b8 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ab548a36ae0af8cdec483f64677910eb85b25a4e29a13c5fd4ff9215cad538b4 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-03e599634c0dbb5ee9a31c1c32e9a3484fc5c94fd70fb36f538cbf0aec87daf1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-a3684148 | sha256-10c1b4142f80b46390fe37726386a3af41bc6a54ceb349b1488cbb26252566da |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-a3684148 | sha256-165a885b9c70880bf8b249042d28b456a0a14f07ac5e55a75630dcdac0e4fb03 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-0eabb753 | sha256-9a5e69e6b768edbd0a1d879b8db2a875e0b04677063282ff00eb7f0b21d9a503 |
