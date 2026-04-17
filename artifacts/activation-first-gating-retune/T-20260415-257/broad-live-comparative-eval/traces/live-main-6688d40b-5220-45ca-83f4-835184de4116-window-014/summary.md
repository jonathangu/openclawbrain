# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ed82473cf8a44c6f378cb688937e33b3ea351a6801142726a4915ee5fb6d88a`
- fixture hash: `sha256-4909ff1896b085966400449aca0e9ac319b4cd9d22c11198c9e8e1d61fedcf2c`
- score hash: `sha256-024c5a1e6dcaf8b53c07d1c0a52426c2e316b2f8e087f8c638cb27bc36cec7f1`
- bundle hash: `sha256-c4d20f1a9a1a9823a31c0469e024ce62cd4c632c3d61c4a5aa59c3bfc5404e8b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46b89f268ea78eb5e49f9755003f2aa744b81e1b854e2ac1c9e8f1a95cc59955 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fba3a3990850f904cce87fdf380a30353d9d8143ecf9890b43b2b9bb6d9ce15b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c243e7d28e743bea03c6cd1048be0f3da5e2595a80320702fb69b32a3a1afe5e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-db225ff1576da3ab26b5e5f1991a462e04fb093a8cd80a6041f2b5141b0537c2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b6eff032 | sha256-a8c197fb5c9319d72b16c6c459c6b25cb102e876193142472713583e8821a1a2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b6eff032 | sha256-b12702cb23864a736f2329465e676489aefc76e5acb1a3d2df05f8dadc75d653 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0a1aac31 | sha256-e77cfe97f9c150be9f6fd46ffef81bf659bc0ee574d2675ca6668e14629e1adf |
