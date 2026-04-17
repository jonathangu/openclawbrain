# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f9aa9aeb2a008ffbbef66937498f659450de790103271a3013e9525a14c6fe94`
- fixture hash: `sha256-5a27682864273526a5ef1ec747be28d22cb7ff7f18b59d5b0629943c5f759e11`
- score hash: `sha256-abb491f6f633ffb250530fa78058441716dfcbccd6e5d60e8810112d8ac1cd7e`
- bundle hash: `sha256-41b64dcab6d857f8e1cf5faeff120e273fc936bb7183eb447e410d8613df2903`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ce2d9b6c9e8a2a1719024054fa61aed786ea8e6db55d697c37659c0fc7a2fbd4 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-81b54617c007d2ba837a9d3023851e605a97d1b0c5b9744091f237b811194017 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-8e0e36c1d9b6177b38f73909ba499fb8f5b72f3c72cb88962394cbcd49f2a5cb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-abc6b5db | sha256-2451c7c2530745d6389b39039c3232716bd726af792996483b55bd43a8e89864 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-abc6b5db | sha256-40a6efaaa678aecf692ecac03e6f18b8f61fdb2c7fc8a5892b280a2538e624c0 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-10c76f70 | sha256-be2801fa7b4e5ca9410e4cbc3d010fc4bd61b2d863b343308543e6f719108ec9 |
