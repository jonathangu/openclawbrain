# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae1abddec00632179423e5d665c773fa81ea75d92b306fc15251840d9f53ec48`
- fixture hash: `sha256-c2c90149661c99c58bd2b000a17d70b99f16ed3daba941c64a7e5c1b67ab99b9`
- score hash: `sha256-afc670e83aaa5b16c585ece3507a43f1a831be2a402442c5312dfe53b86e8ca6`
- bundle hash: `sha256-c2c7fa612ddc2dc49e2a2ada6b0be63e6fa4676f69511d9c4a10fc54144b7295`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-83c42e700005538dba5b3a6d69c6c5e443ab91af8b598837eb4ca6b5f8135237 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d411eddebd09f15a955d3c26a01e70e1888c216447ce789f724446ebd412203d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-de447283f720530bf852b5ffbe79bc443b2faf1a187e836e455d8a16f99b7e47 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c37743f0aed79d880632c3440c9ad4c033071e843d2c321a0d2789efb6ec0c55 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6292a00a | sha256-13d46e5d553c41762854dc9abae0c8243ab353be4f1ead99fd6badec822edf71 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6292a00a | sha256-5557c6131f55321bf4a5b12df55ec26f781bcdb8d28029b3e5eb3600f47ae5c9 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-6292a00a | sha256-13d46e5d553c41762854dc9abae0c8243ab353be4f1ead99fd6badec822edf71 |
