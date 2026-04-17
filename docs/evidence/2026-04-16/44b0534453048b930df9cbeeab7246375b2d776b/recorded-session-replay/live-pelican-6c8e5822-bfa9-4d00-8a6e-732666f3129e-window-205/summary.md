# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f579c26265d9087760a95275a1ed5d3c29a7fa2a5745f0cd5985ac21a42da923`
- fixture hash: `sha256-344c0f8fa42bcaf494090e8fb4c4629c475783bea29bc527602ae9b6d23e9791`
- score hash: `sha256-76387b2ff66ce81965fbc3636e322df8712ba98f6dc1dc8fd49941a291950504`
- bundle hash: `sha256-402a9c4644cc689b64617b07eb008bdee742dfbf389a7ab02e37739c73f9197d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c8dbb3069efa791ffae92b155399c68bb15a7550a719f9b3772c99bdfa5fdc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee78f18c4d17dd0d47b28ef7d774bd51af1c189ad42e77c435699bf789b59f2a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5457dd1a20ee4377e4dc8b235e006e21e54c5672ffc7489c42a29f2463144d9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6397ee54eafcc8bbee596cd6cfc087697cb171d6745bcf9d3f08db99353fea06 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ba83604b | sha256-f439431ebf80efec206e72be94460fde82eef3f8c815215244e08eeffb4d9949 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ba83604b | sha256-ed3c25a15debedbafaeecbab1a9ccb834dbedc4067fcdf1bd178bd1108ead534 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fd88bc9a | sha256-102f3e0889c1316462cac71dcb0c56bc1b8043f5ff8fd6adf6235c290dead454 |
