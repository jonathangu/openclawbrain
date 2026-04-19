# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d88b28d725d828ce19828c993023f4bb29218bcb84a66644da59b76b2ba63c9d`
- fixture hash: `sha256-87338b3d5a752854c0bedd7b04604892d7d56176980cbe837350f3ba996c423a`
- score hash: `sha256-f3ca94287f6bb6363a589bc965ea87ffdf7ac9713c4d51affb605ae917132fa5`
- bundle hash: `sha256-6f63fd27975cf845cef88f39c69864f7a7c99394b8433fc7733342bee50fb883`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8865b280477f916340cca3eb893b61dc8525b802f5a2d26079af29780e8fa757 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-05d2055eb629743eb86838fa12943c958344ef206d8371a3c626cfb0a11107bc |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-7f58e99cf507cf0f43f6dcd35a15ffd31cf2f2f918d7ee8c65c32a6321934679 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-956e3ca31b39cf05454cdbe25eb7ad4f72673f6d654fb520aa97f3969f68bd0f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ca0aa200 | sha256-fe0aeca64d769e68456a48a780aecf6a9b4a0247a734ed2bba78df0753572cc5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ca0aa200 | sha256-957f344ff136ff40694d73af14d0b04cf1fb80ad7cbc74b9b8936cc328f3fdb4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ca0aa200 | sha256-fe0aeca64d769e68456a48a780aecf6a9b4a0247a734ed2bba78df0753572cc5 |
