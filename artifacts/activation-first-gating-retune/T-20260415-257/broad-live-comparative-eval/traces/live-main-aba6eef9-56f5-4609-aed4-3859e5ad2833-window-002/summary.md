# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ca1018694169cc3fe531485fc537c09a6239e84c0e4410a019dba97e2a66fe7e`
- fixture hash: `sha256-9d6b96efb0f7a7d48de55af286c816bef6a9a27fdc8a979e0eeba28c500d12da`
- score hash: `sha256-d178b54dd66d51e649614145e64f2bd03869769d4cf34c7b989c1fd2705d0bd0`
- bundle hash: `sha256-1fd2b605e7d2ec038c613c40c53b17d6d0dd322fe4caf9c041a32490bbbac992`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2225486ce356841ccd69a322b5b86cae51f3de0b57802b050f099b2bdb0a0f2e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb4403b309f611d08ce38f4e0b88eae643dfd9eeb8b48f5ab8407659ac467278 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-59a7eaa9b4e3a5d74a79cc98511d81c9f3d9d22ed34d6dcec5ac9d092ed03203 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1d6b29df144e5f0d0752b73c46209d1404a43a7b505986f60be91ba0751f209b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-18b9beeb | sha256-fb6915b6bfb59193ccaada0f9b0b618cde94ab6b86f0007c49be504329960b2b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-18b9beeb | sha256-bed19fb7bea70fce6b638ffd67b15404efbb6d1e7e807d59a384ab899f4818a8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1878075a | sha256-1d3f10bebd6185daa27875197157f7cb0fc6dda01a04479e302fd4f0654b6446 |
