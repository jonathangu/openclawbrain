# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f39bc7349702eda18b0d056342226b1aabb41caee42927ea480ba26a62daf2f`
- fixture hash: `sha256-841eefc5eecc02fe972ed7cac8e3716da5b289fe7edcf8c461503d651db37931`
- score hash: `sha256-949eabef596d520f6295a5ddebae671793c3604c97d0606693ca9a2019c46e58`
- bundle hash: `sha256-4eeb7c24d7c230865e5a37cb9020ded93ca1903adea64a232be0ca4a2171f844`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f5559e04ec3c75dd16ee057dcaef2391dd2363ce8cb9ccfcfa727aea97487dcd |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7b0496de9b9d51c212212ab19031473250517a9710f65e2472c2c6f48d946870 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-806f48b076e75316e419992b1172c5fdf6e573d6ee2a869784ee9bad4433b9b8 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-23141e98be8115d02406aa327864dce124c4f070d07958f256df0169710edce8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-165aa629 | sha256-04e9ab95ae26fce331de923ce966157ce8e711711308824f49a6d7d7944b00c5 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-165aa629 | sha256-0164be415f2159b53a2e9563a6a72ad309f3513e2bec5d54be2e27ce897f3a74 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-af5d6de4 | sha256-e67c4ecd89e62b4e909236103972eb2a171d210e4e7462064ada781fcc4d4d73 |
