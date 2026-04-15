# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ac9290c0886604313b41285576a01cf7cf17f3dff1c7512072e0576b64e4b6a3`
- fixture hash: `sha256-e8553284ec9b39217012ceb74491f30f031830486fe62cf2b317a9220acc58ef`
- score hash: `sha256-2a6d68f458191f0dcc517a4b36d88d5d42c3a8acf35558c6e25ad11f0faf03b8`
- bundle hash: `sha256-5f29a7af3474e0610bac4c9e6c869f9e742d16b230b4e79ba9af3d406f968bb4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3c9506ef693eb127f9b82369f4aa560da295df39e9e3ce7446b93c59e3f245e5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10aa0e6bc799fd55461b6ccc156b56b06944660190a32747d41e675bb9013a3d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-07bb88a6bd5dcaee112c82f79d42e903fb3401303f93744c65bfe648ad33312a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-1cec1afdc930881613265d9091ff8b1cf7ff4dae04e683ed70009203b3c36c4c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3d29d1c9 | sha256-e83e01be413d0664917db3816eae9e6c972d21a682ae619e44e91e602177819e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3d29d1c9 | sha256-66ee0f29f855fd805b6e1cfd0fe912ab1bda4183734e07f0df30283b1da1b1c6 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3d29d1c9 | sha256-e83e01be413d0664917db3816eae9e6c972d21a682ae619e44e91e602177819e |
