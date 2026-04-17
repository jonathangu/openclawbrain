# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c65984dab810fcd56a73ba24f7e48a3de3329e9e72c9abc055205970cf393432`
- fixture hash: `sha256-6edadb4cb34df6bab57971cb77cafbb8b923e3e92f73e144950ce412708011f4`
- score hash: `sha256-5235af6c877a4fff324b3430c6ae45d6a7883ade45223bd4b13cd52b24ae7c0e`
- bundle hash: `sha256-c32ad48bd360a0bb2cf1b66e677b94227f29b175431ce8d3b45b6538b550422b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b98041498153f3fab8845179ecda7c5ad292ef71a993f916db2031745eb7d0a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bd52abca703b642cc39fa471250da918e487b530ed7516c53b66061570a33e53 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eb18a6f18dba190988d8d6af206ad3f5c7673bfa8b75a97375acc7afce6dcd95 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ddb394d6b8e2091badc3c4c40a84ec5f97b1e5ea808acaacbfef85e817d44913 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e1775aae | sha256-cae61cdead4e82dfb6bada83f791af8c72a873e3b6a3bc3271e37afa54a3be41 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e1775aae | sha256-a87eb6dd0c68915d7a68b3b03c01c8fe7519486aa1341b03504d4e1538c4769e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9137fac1 | sha256-1218d4e7c61b6fff48b774af109bedf804b513a37210a61ddd18b5cd45cf2455 |
