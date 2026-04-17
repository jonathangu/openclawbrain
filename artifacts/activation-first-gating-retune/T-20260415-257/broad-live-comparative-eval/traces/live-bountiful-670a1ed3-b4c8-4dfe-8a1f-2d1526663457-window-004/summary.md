# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cf10cae0f36c648f32a6c50dd8217f4092591b4c33fa07516441723d723d101`
- fixture hash: `sha256-74ccb99a45cbecfbd0675ba926480f518b6d9257f4cbecb8a7eccfb5e3bc826f`
- score hash: `sha256-09fb58f6d6031da0ea7fa01217a6ecc051066db7280caa63e1fafbf38d1222a7`
- bundle hash: `sha256-a93961aeacf3478878db8b3a78f6fb405c24249385999d6feb7edf6082e44b1a`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-deab881832958a8bd935ee6c81daddd68f45a0ca219749d213e1a30ab0bb8c14 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e43e2c6ee469b40d174ed03e9039a52df1e9f4efcaa15c095c0d99d32e161afa |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3e07a88402a5abf61000779a36511bfa3e42259ec21ff0b9866ddee0b9ac6f9c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-61fac13d48bd69f4bc066b5e71cc33d794132a08eea6ae2339e974939815fe7a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a4356446 | sha256-0284cb479c899316c67a87cfc16d03cd142d07e63cb6294ecf42339322b45439 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a4356446 | sha256-759ae8bb39c17b82abcc839cf9baf4553e0b1618bf4b87b88d1fc10ff3e4d374 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-8504f337 | sha256-f4fcc4d5fc38deb8b54c0e1b28a0f2df4701e3d23734aa46491d44c5bbf080cc |
