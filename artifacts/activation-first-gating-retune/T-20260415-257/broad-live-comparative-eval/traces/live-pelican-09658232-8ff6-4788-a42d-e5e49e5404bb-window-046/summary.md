# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3c2d2ba443dbef189a04f697781c21859dae784757f070f6d624a5c22c1fd87e`
- fixture hash: `sha256-d5090234178376a892e6c521a05dfe5104bf688b9e6c7c68cfaf8797d0e0e324`
- score hash: `sha256-8027409513e4fcf98ef515d2653eb8d3534bd35d1afe555f6c54bde93ab36d82`
- bundle hash: `sha256-674f898c8a31858ea2017934ad15a5d52e8aa53805519cf68d2cdeeae5d62450`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30cb0b5562af7757589ca1411482395b52af039eced1208652e3e0610a2b0728 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-01250a0bea7162139e046801b7cd608cab6cbbcb1c1eedc0c456e0a39f4e0932 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b9a6980c5b095443618e0999039bf6f5e553ffa323faa872aed3ff60f1cd427f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a9c8d67883b23110eeeb5edcafa03ac0d285b36c60d2817022f3b7267ee92100 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e0839f99 | sha256-764749e60d6c6f259f5698b0af0ad0810a8adcb59b11a3e4acb092d456a9b51a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e0839f99 | sha256-524b60b8833d1691d43cec480fb0488a2f50e079513a15dad1b5017230c2aea2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d2745c64 | sha256-67ce626fd4435f3c744128da0d2d2438412706e6c5deb9f5eb903dbbd9eec0d7 |
