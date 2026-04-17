# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1d9740289fc2adbace7590e78dff24d1d94c6a419d6474e3af27754996da05a`
- fixture hash: `sha256-b9420b72d3a2c2c9c62adbc0b7f3ef24407bf200cf73b9c382cce44e2d33fe6a`
- score hash: `sha256-0881e580576b16da0ece32a14e92691096175c14d05a2c5d94e864bb8cd025b2`
- bundle hash: `sha256-7f89ce9daaab5470eaa65c8b07ef0beee6fca523826b22368b68ef9a04138329`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a36b1da37ad1b5e7a8a6bf9a89b082e6da9affb9cefe62c4630aaf0bc52cbd76 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-96f3bd64ff447216bf45df3af9fe55772452094a82fd940b98b1c3ce0583d281 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b930891bb4419bf9a2f3e8f23225295d8e8b7f0de4c29937f604ab9eaf2d3e4b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7eb435f60aae0008778c9b802bc7025f7c3953ed1fbd423693e268a514d671fc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-40b2557c | sha256-250f9ef528f7fedbbcf007051cb56c8708e719c96f4194af296b0b7bd4b923f4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-40b2557c | sha256-20a87e2c1303bfb380409e8a3bcf70d1dae657cc660ab94a7e7f619f0951291f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-302df0e3 | sha256-187b830119c24a31e8cbd4555fe738400e2809efc7b0352b96026cd775505127 |
