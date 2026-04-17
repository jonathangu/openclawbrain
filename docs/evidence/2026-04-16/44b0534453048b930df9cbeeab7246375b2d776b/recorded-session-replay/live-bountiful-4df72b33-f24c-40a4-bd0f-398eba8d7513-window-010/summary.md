# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f896b3d7889710642e066f81d9ef38f09a0375e7c4550a3de44bd42b8be0c728`
- fixture hash: `sha256-02b10f8d55f27089a7a2cdde95f78ea9472dddb1b95943a1431bda089a73cd5e`
- score hash: `sha256-338e0218c444347daf094ca0c4bcdaa0ffb7b5973e14d647e8adfdaf1715e524`
- bundle hash: `sha256-bd6ebb282e1a85f94ead2c1e5a177314015cdec1e0ef235f9e5ced8c948346a6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6e466bcaf85a4528dcaf1f22f57a3cde69a22135dbdc628862617cea9e4f77f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6124f80d349c27760bc0877248f9646c9e962f0d137d15c59987d2cac9403b10 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7b30a756fa85843bdd6c0636b041ec6d26dee199d07ff1f7038d606f2b8c8e34 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-457a3bfa7488796ac43a52a0985cf65ad858064abf1ec09dd1565f976d567b38 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4246ce5c | sha256-27f9fb1bfc75718c2061c6c9d3539a1e8bdddc068551979de2bd02fe3d431c14 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4246ce5c | sha256-c1913d8db01c5a2912c8946a646949b088df04fabf6a46048d6f1b1947c5ef5b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-972d7c37 | sha256-41518275f15901d14de42c64d36acd4216cee126abb27d5afe92c10c37ff8745 |
