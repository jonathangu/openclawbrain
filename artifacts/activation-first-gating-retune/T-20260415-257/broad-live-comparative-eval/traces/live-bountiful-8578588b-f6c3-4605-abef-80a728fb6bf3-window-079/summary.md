# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ac9290c0886604313b41285576a01cf7cf17f3dff1c7512072e0576b64e4b6a3`
- fixture hash: `sha256-e8553284ec9b39217012ceb74491f30f031830486fe62cf2b317a9220acc58ef`
- score hash: `sha256-5f97fde9f9bd7843ef3497a51ff36d823aca979971977a801049b250cbcab738`
- bundle hash: `sha256-25286b6c798d4ed700c204da72054d4f34c4f6c628035453ae16d2ac760e562b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3c9506ef693eb127f9b82369f4aa560da295df39e9e3ce7446b93c59e3f245e5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10aa0e6bc799fd55461b6ccc156b56b06944660190a32747d41e675bb9013a3d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-07bb88a6bd5dcaee112c82f79d42e903fb3401303f93744c65bfe648ad33312a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e33353ac1c62338e5fd552ab132f29c573f81fa5c5fe0ef7d2eaa72614708ab0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3d29d1c9 | sha256-e83e01be413d0664917db3816eae9e6c972d21a682ae619e44e91e602177819e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3d29d1c9 | sha256-66ee0f29f855fd805b6e1cfd0fe912ab1bda4183734e07f0df30283b1da1b1c6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e1b9e072 | sha256-eb0e97b9180cc2a1170f2ccdb4a8e2b7177924c4a0bf0c0c05e67b61be4877c4 |
