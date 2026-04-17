# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-07e10f6820bf810e1999011ea58316d9d53ec99aa0ac7473d30a2c9a79d153ae`
- fixture hash: `sha256-54c4d68b5e528e2dc7ad50c599fd75e1b659a972d8d4c97376e292a3ef62dcc8`
- score hash: `sha256-aa9a6effb54d9862ca271ab93882a62bd1ea1208d8a2cc5d246a42fd79005bc9`
- bundle hash: `sha256-62a9dd9c9a929648dbd20e95169f0afc362523eed7ec16e6fdb207a54808281d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-430dcaca40205cc8d42bfba95521d8acee2a6e6c074b542cd0b9a2d9f1547939 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-083c68256e6c93072892d5c9558cd7ebc0123e9071f516626563454931876801 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f5329004bd75294fb338aae2f5c7a6553c61b122e54c64cf91d7e222e71a20fc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2ab09b726121ccffd046f3346d7a1f5b1a76df9b977fdbb306879b7a9d4ca711 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ceceedc4 | sha256-441d74682d82c911bdca828845ee13129c6bc23a8110b564c0bb448235779d63 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ceceedc4 | sha256-5a7a919e70ab31c6988c377371efd8a7d997e9cd0f7c494bef7def8a341acf06 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-93fa119f | sha256-445c8987164139275155a0ac8a798874868c51e86afe14463ccfd245f5e5a814 |
