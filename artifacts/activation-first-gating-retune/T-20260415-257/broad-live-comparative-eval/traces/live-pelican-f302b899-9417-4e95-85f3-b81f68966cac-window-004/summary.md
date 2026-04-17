# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-08b73891909d4362ec26f6fa9db500532bf1bc8c805c846177530e06134e3890`
- fixture hash: `sha256-8b9b0ca98fc7faf65751139ae1faf124a5228fc02a0f5bb6427265ff145c7a87`
- score hash: `sha256-26dd94e02a39257b43b9774a04997166ff6bb9ec9ff9e98abffb90dc76bc99c2`
- bundle hash: `sha256-bca81685eca6545024799ca87f9c03f4eb1c12a45d66e93f902b4f3c23dc4f70`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f871e8b1f54b24b9e075d5f4db6f8b41f6cb53e929f6d747d42ccbb2426d8d7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a4f3c96c4f68d103e74c4fbcaf722cd490aa2a22f8f71e0fd088a3ec4705bb4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bd4a6e03655b92043380dd2469ba82ee520e32952352b469c2a3b61032fa718e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5c93c0bbb5d70b53a49a889f55175acf47816322d07b40e0e63b4ab339084c25 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b5483eec | sha256-1465f71ed7d59659517cb8ea4e219f69813353541280b81bbe3e579f5848477b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b5483eec | sha256-a6fde61caed1118e97b8a2b366ed0c56b975445a82695e974a0430897b2f682e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-334b4beb | sha256-aea351bcc74a4f6a0922f2e4ecbf5f20dbc823e37167ffd7aaa6b75e9dece8a6 |
