# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f80bd1bbdcfa166ddf0a470afc83601b492c4563388d58e45f1a2c05fc2e95de`
- fixture hash: `sha256-6351998b89e93fb758f480838cac801b256b59b498a256a0fe6d16fd14c2a7f1`
- score hash: `sha256-73d107e11b4722f3b9d85b8a3ed666eda54fd325ffd7f10df44ace3f3e8d6615`
- bundle hash: `sha256-03e4ef729a6e633ac01d753f87e9f53938dd41908b819b8cba4838c53142b090`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc854933a918fb32be2808e49cd62ec70012cfa090e09597f270649ed2f5446a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2145ef0846918bdfd539981c17a47e42b3759a1288290a9f442bf703aa87046d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a95a80a9b6c490c895ddbc759782f32ad94b871417fecdf9d53ac58de27ad4a7 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-49b2ec2d21dd8274be2e6fccd0399e5ed81ab95e9a76ad761d76b015825f3a53 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5e9e23f9 | sha256-fd07311884bc7f7da1e64aaa07f962fddc1c378c80c4b66c78db796c0148f2f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5e9e23f9 | sha256-0e2bbac72913c44c248587606e914dc71914a060c20de54e963320aa861837ac |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-eb66de06 | sha256-aaacbd7717b9c122bd382c2775ef28b444573294e5decc28e92c3180d783e630 |
