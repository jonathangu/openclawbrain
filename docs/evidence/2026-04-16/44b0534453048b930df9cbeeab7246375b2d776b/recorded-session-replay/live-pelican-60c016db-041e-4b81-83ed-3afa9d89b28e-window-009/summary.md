# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6c49ca2fe441629d93695e938073dd41facc650cc9fd301e1fa807efab482f72`
- fixture hash: `sha256-a51ac08634e3c4803c3d3973ce1a7c858ffb1429844452de0ed6e3279b36730b`
- score hash: `sha256-5b8a4a98c9a998245d2d4a1f735fe2b465a77212e2309b7f5352e1314fcfa051`
- bundle hash: `sha256-37d1e5344c6c759ef8986664b5eabdd9e9042df32d9e12ee6719ef467ae37eaa`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c2e3cc9fd6ee1604d2c526310b1ddca47a11a50e7f9573ba696d4001f01dac |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf105de01e65293a10af424e02def6adbb4acb11447958b58bf78e2fcda69dbe |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9cc7ada5d45803fa49d4d05d8fa645017e010a90de5c70ec9519cc7129caaabd |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-5fd07ab5d55a9c8512d0febe8aa34e20b39e6b78c5f831c5dc06946bb9eaf1e0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-df00ef51 | sha256-179fb4a45b62619efcc18e1295329944e1d82ea819f1dc31177aa019993c2d92 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-df00ef51 | sha256-e04417a5bb64a96de5fb061afb866cf3f1fc6f9e20e9d4849e921c729eed0ecf |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-6270c56a | sha256-c7aa1ed829d41190d4a6070c7afe2dbeb1194136a5e01f33465f646e1afd6d7a |
