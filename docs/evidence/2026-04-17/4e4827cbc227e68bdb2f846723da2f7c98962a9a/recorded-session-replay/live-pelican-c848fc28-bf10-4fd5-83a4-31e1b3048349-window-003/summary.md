# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c65984dab810fcd56a73ba24f7e48a3de3329e9e72c9abc055205970cf393432`
- fixture hash: `sha256-6edadb4cb34df6bab57971cb77cafbb8b923e3e92f73e144950ce412708011f4`
- score hash: `sha256-56c99b0a9a49ee42f1b35fe7e127efc0ebea97db6f3cfa65f84b2c6230df0057`
- bundle hash: `sha256-fe11c1862e901b8caa487562c60248432bc48b169e384188e71cf9b649253abc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b98041498153f3fab8845179ecda7c5ad292ef71a993f916db2031745eb7d0a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-217540063b9f1e11f758d63abf44b4c6ab7667f5c2a5df6adedd6dd3d04a0111 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cfcb2b63118dadfac9b643c9297558743e233c969561094bb43f4783576be2e9 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-85d4f04ce87cf4272315466b9dcf6c8da58f93b75dcca26e133a8f9c2703df9f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-061201ff | sha256-c49d53d1d537115824a73eb2756e07bf1ca5e64b7aa275218d83e94781308386 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-061201ff | sha256-c118d826b0d150edb9fd4a85f728943cb6b48b315866f130688569868e22ea2c |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b5d2a212 | sha256-3218985f4f218e80e08a8d48167ecb1b998f654350ce84b81974ac1c364ba0c7 |
