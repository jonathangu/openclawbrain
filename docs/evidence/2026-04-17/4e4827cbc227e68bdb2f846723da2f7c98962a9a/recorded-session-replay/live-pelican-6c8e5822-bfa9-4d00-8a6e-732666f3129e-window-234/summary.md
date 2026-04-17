# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80d3477f10050166bf08a79ad115cc0623875c77edbf3489b3449d2e77618193`
- fixture hash: `sha256-550621052f6f6f4dedd32e7dd1966df3bdae13f0842e74ffdcfed29aa308dfb9`
- score hash: `sha256-c5a83345879386329bbe03388b4d0769c7e65a7d2941aedfdcaa60b9c2ce31fa`
- bundle hash: `sha256-1500957be1d56ef9b684d4a84e308adfdc23b8586cef37ec3831d8188af58367`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b053d8d2940510defb6223852e2cebf21b6ccd631a727caf5859e48b2c5a0baf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c6b883137d59da35833d9435c3dd6a45025fb28d2428451f2ee2dfc05dcb0c43 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c8a9edacc977d4bddd6515122f533a923b8bb230b248d917a763aac77f7905f3 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-354e30f623d6fa76b125de70cc552324a90e2fd03487c31e3f664e9dac4134ee |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fbb94959 | sha256-8a75ef33c313d8813a7b9cc5d8763e1571d3bc19a664bc770be17a8a401d9e25 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fbb94959 | sha256-50cd225c107c615ffecf9ab65a7440997fa73a16b0f38c156682f8268c362f6d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f0dd9d1c | sha256-c21854ca6fb5b6e36cadb7fcdc7eeaa22bcc747ba26f4ea9908ae1558b4c9e92 |
