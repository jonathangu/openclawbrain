# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2d785c91c6c2597c88bfdefe91898000c30733ecf3cca8e1fa5fd2d6621049e4`
- fixture hash: `sha256-63b7942b83cea800c5fc9cb957ce0307322538d9d8e1a745ea7ab80b74e65911`
- score hash: `sha256-f7f962093cca9f45a6c7f3f65b6cda7dbf82b6a64ea52d78eb9a0e1e0b020fab`
- bundle hash: `sha256-89246c70e00ec14ee4c1af86216d49e416107ef86d7eca8b69a9172f4527e39a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4648c104e20ab98d8928f41590949536cf65a6240f7fac95811ce6126bd169f5 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-41aa6d9f3ca430e576e184f020adc9cf3148e1114f00565ebc84e669e0e5657b |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-cc03de20cdac5f581d34fce293610fea9de8eb744fe33720ca4e8af4eb559750 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-c69c8812cfd13bc9a3c4057e778df1b76f1b00a655fad220d1d6e18e3ced55ee |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-ab1a7c51 | sha256-173c05a82df78cc6a03b3e18d46099da2fa7e3a463be237d31472859780d9072 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-ab1a7c51 | sha256-e5d70ae0f4fb5d7ea723b25b7a472e55a9f3f084d08150f31a6ddf745c01dbe6 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-0d6cd86c | sha256-e4caa383af65c22e2acb1e86b963c3dd12b5ce93aca534e71ae2c006ce322cba |
