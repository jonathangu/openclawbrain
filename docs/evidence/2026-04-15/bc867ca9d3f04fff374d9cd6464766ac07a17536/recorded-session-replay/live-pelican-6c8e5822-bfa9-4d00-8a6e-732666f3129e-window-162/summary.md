# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b02e3d7c43b0542a9708c97a4decb5ab50a7fecdb19a413e8ba04a6c6f24587b`
- fixture hash: `sha256-fc0fa875ed0ba10ef61e5e8b6c1b783878d38dd1c5525b62b1d2717e4e66617b`
- score hash: `sha256-5d0fe42f192c7994252ad28b387affc671063ecbc0385a8f498488fa0a983a5e`
- bundle hash: `sha256-ba0fb47f20932a70eaab13b83bd12b8178ef87a28210a8dcad03cc1f71aa3824`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f8f3d7baac7ea624c59c2785d2ad8b5f8904cda6bfe17f914b150feacd473265 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16c048a9c82e5c5d12b7ea6d945e695d9f9e1f4642a87f39e023982801e52b34 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fc2e9eeacec77305d4b9917c635787651739418f5a23403cc3636ca597ccef08 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-99eefa622f80bd63a3d3a9432b80373b353ad077074caa0b7c90d1edbc5c1814 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9d2a2b0b | sha256-5d2a06cad32a571dfe380dddd009cbe679a9b4a785847a27eb50d22bd1286bbe |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9d2a2b0b | sha256-5a1c67a7445e09f1c31c2ab4aec0aa04cf656d48df2acf57a85475517bc8db2b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9d2a2b0b | sha256-5d2a06cad32a571dfe380dddd009cbe679a9b4a785847a27eb50d22bd1286bbe |
