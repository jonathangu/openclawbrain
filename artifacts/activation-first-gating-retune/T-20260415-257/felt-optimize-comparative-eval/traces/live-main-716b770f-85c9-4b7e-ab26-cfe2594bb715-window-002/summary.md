# Recorded Session Replay Proof Bundle

- trace id: `live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e321442dc8033dd76db95133894d776ec05ebee5a5a98eec612f6b420b907658`
- fixture hash: `sha256-742118fbdeeb061b08c45664c524844d158f1b6be0af589fa277c4ab60f660e2`
- score hash: `sha256-247b6060342d7a64eeae7c9aeee910297cb4b6a33f1c1ff99007f00b6f806dc1`
- bundle hash: `sha256-c90d06d9bfb222a56d7b2a1a197e179d267aad08a5a7dcb710e4a7b1121a90a0`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-05b0912208f70d1fd8d2baa8f914bf08175b3f38b8f85e68cab4f50d835557ec |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2ccbe1c493fdb33354e4a3f29e03ba4d79b64beda0258997be5056e716286be9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-623500c73b26fe62e57c34c50d8220ff09bb5affc462a333678bcbed097f6fb4 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-6470feab8a5cf866e385673300ebf60ecd6d73df1717fd9ddc9d2ba2b4eec8ea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-fc2df628 | sha256-9b76abc2ab44ba4ec51346b1b8fbc43e6a360bf27ffc80e5db6ee5b6d9639fa2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-fc2df628 | sha256-c8e175f5d887325770b62488b6cdb4fb2a9d4c6902a1609f0bd47b078da81249 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-eb7044bd | sha256-d82019d753f9abac0d6fd809249b192ac119b87312352a1b61c62cf49d0b04e2 |
