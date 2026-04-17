# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cf10cae0f36c648f32a6c50dd8217f4092591b4c33fa07516441723d723d101`
- fixture hash: `sha256-74ccb99a45cbecfbd0675ba926480f518b6d9257f4cbecb8a7eccfb5e3bc826f`
- score hash: `sha256-a601d092837c4189428e81de0573b034ad850a7182940011ca529a3404a529d4`
- bundle hash: `sha256-43598475d878ae349ddeed6d6ba44290f7037bf0d777925d81207fb188f599dd`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-deab881832958a8bd935ee6c81daddd68f45a0ca219749d213e1a30ab0bb8c14 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-36cf382dc0decb766ce0965a968bf5d133b939f06bf651536689f5fdebaf4c9d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f92f89bb973b6e1782dee340f4baecf7c11ed97ffa1721171d7bd3450b0b8f15 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-b1647df2324763509929fcf5ac7031a7065df6444ce275deb33d4a17c8f2effb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-860b42b8 | sha256-744e237d88822186658cd3ae80cd75b78bacadef2e4827b1c9970cd0cc8b6e76 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-860b42b8 | sha256-b7e39dec887736b092a2236a9f44a915b2014167f6a0ad67131d545b6b5a5824 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-66dad1a9 | sha256-98944a4bd135ae80a9913c2381a7bec09de731a7bfaf11699370c1eaf459a9d3 |
