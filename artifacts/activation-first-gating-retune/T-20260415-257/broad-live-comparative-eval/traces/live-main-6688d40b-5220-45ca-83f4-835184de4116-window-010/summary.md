# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9af4d3068fe0abcd8b0d002d37c1f3cf1f47d195e7f9302f6c99d8ac1c1ba8d0`
- fixture hash: `sha256-b1358b7f23f888234738e0f7490e996569d9c09e6a59451858fe36290e0374c4`
- score hash: `sha256-bf56d9123f42955563e55ac5abbef6d646fd034df396e29ec998575063726562`
- bundle hash: `sha256-26870769ca40ed8fa30f3196fbc4fcd127ddf66e34914569f6f0231aa29dc555`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-38fcd5e204f9ce44394f224032307e552dcb6c83cc2ff3a9c8d07b3df48aab19 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-de6ae72296944fa6803cca0321db093fe144f3c45f275130ba97845184bc1b31 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1db0acf57fe1584cfa72dc9c33afebba89a754e3de0605fc81b11e3ab3fe92b1 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e2a92b56bdc7135d68ac0cf563b996830c83c5e403a0ef3007808e568ad3ea67 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d1eca3a0 | sha256-5814effac7719876670bba0b032d90aced9de336373e246b5e5fb5f0bcea9fc9 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d1eca3a0 | sha256-89da9ceaf59446a9d786684a8a17be423cde40f5329f64f3cd98ece3504b92fb |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3b6369a9 | sha256-2fff72cb5b6b5524629a79deb04fe4f7e3b41645c25d83b0fc652fc5830e3d18 |
