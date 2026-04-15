# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f9aa9aeb2a008ffbbef66937498f659450de790103271a3013e9525a14c6fe94`
- fixture hash: `sha256-5a27682864273526a5ef1ec747be28d22cb7ff7f18b59d5b0629943c5f759e11`
- score hash: `sha256-7a1964f710b6d85b6149c3c6016590a2c2612f416bb0be8805873a7dc4c6414c`
- bundle hash: `sha256-faceb0b58d9405d62defd936afa31e71e9e9b60a5866a875117a7da4141b6ec4`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b2d51f37ef17f0ed82a2f36897126b205c47228efe0e37855cb029004034490 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-78ff1bcca61d263f30231714bdb6403e3e94f1d9c432ed327772ef97ea34574a |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ab58cde9a54a65d9d2114ecc17b461896e0c562ba4e3de2d070a29ad10cb498a |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-13787a0eff6f932bf8ef8681e3edf26c05dc03f101d60ed7e12762c061834b84 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-7955985c | sha256-07de8ad5552ec8ba5f1b9ceac8ea3438e74efff05ddfeaae897b4f9390953a21 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-7955985c | sha256-292f106aa6f9d31b2d976371e0a4693cb9d07a110ff56ab790b4040998db1248 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-7955985c | sha256-07de8ad5552ec8ba5f1b9ceac8ea3438e74efff05ddfeaae897b4f9390953a21 |
