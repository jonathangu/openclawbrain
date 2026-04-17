# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e9723616edb2ad54551f6589a0d45a8a37518248db0b8e108b75c34c56efda98`
- fixture hash: `sha256-05b8c1caa5037185047ead07b4f318668a0cb8dc8aebbf981972a18dc900efd9`
- score hash: `sha256-522a1b0974773037ebe1689caf5655d827fc4dafa6e644d50b1bf7718c7fd0c9`
- bundle hash: `sha256-514ea53e8f71b6b12dec306c9891c26b40291db505cfe77d2942884ebc5650e9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c5680042ae3bf747adf63e364ca5bf29ca561c697a87cac9ec59524d5a5a73c2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4754d1dae0fe37af9460f321a88846762b7d2e4ad03e2511da68f0d4df1f2960 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4aab0c19fa92b3bac3a4bd7a790fdba485b6f8c91b850ce564454c7d98dae256 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-775de1f1cdd5b63351ff588db8df38f94de28db19a4b06acbe369eb661381c29 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-568b516c | sha256-7eda0b17d307045b8f22bd72fe4d0e48569ac518a8383f3ad035377d7cac0500 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-568b516c | sha256-9c0c4028814edb804d545c6493bb4de034850af0bb1398e812414b82fd55efff |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b27c494b | sha256-3a582ef0b986a0d26ec322a25b517c0b0168ce4c692b525376bfd087bd78823b |
