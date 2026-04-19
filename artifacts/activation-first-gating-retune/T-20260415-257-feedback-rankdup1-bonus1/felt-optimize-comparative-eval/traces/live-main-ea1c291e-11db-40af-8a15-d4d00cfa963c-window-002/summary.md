# Recorded Session Replay Proof Bundle

- trace id: `live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d851cdfc065d530ff6a05cd12aae1453cc6c5cc252f286f05c63b39f7b7ea103`
- fixture hash: `sha256-add4e01555ea0b700f89e1179ee076e863d3216d180ce57f607f066d853c468e`
- score hash: `sha256-6f828a32be6f1c0e3b60c086e0a7bc1540a750f24c7e79e9c4bb3b20ffaf4c68`
- bundle hash: `sha256-5b4d4f56864087946f4c499a9c37586c987b520b999dfb1cdacf0b80bc0e3012`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e16f2c51fd8866c40ce249b661c20fa44d3a586d3c45a550284b22e35e90bd83 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-f674d2ab9c71178699ee6125872a32cf82a25c6ed4648be1ade0001f526ea258 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-1d09d86b13fd9e2ee49f3c03c0a70c652077c13f239bbf0198ba3e32058cfb0c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d911147ffaa7f4c350a029aad5e77a3760f34a6cf4184f828b8103d19daa2950 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-22896f94 | sha256-5ef386d35e05d6f1e7fa3c456d20aa60d6797243260461a828ce55b6fd370ee6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-22896f94 | sha256-2867c56cfc0844a437bc4e045d5efc2e979670dc9ebe654655bc3496db928368 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-22896f94 | sha256-2e6ea8a13f75f91673910ff8db0c4b8cbfc98bc0e31fcc928c8fc3ef872f20d7 |
