# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-893bbae402dfd268ed32b5d7137ff140cf694f0744998bbe068d3b879f9ca62f`
- fixture hash: `sha256-fcb57e82ae27e8603221264cddf33ddbfc96e5fc7bee09bcaaabd6c496832873`
- score hash: `sha256-090d8c7126160aad2c3d2153f4905b6ad086d44ee295162738071ab61dc4dad9`
- bundle hash: `sha256-4f2a8ac77aab21012c34defd685f8f139f49f3036405cdf345072bb8e64229e8`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-593942095b3dca97714f70191bc3ba85569cc1817c4fcdf560f63906b71b3cd1 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a655f972c4d21f3c13f95d63d4a13398e8410e9471a5195e318e414219e6cdda |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-bdce91fdc6d8cc45d0e899579796e2d40d0005faecc6a7cdc1b90dddf7e3e608 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-00f7eb45c02064ca9c45eedfd64711a4b73d0101c8e440e0efdc291db280ba76 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d3a46040 | sha256-a959330bb8d2c2ffd4d4c46f522668e445fe95bce60bfa88eb18b0988479a459 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-d3a46040 | sha256-df447c88e0ec722e95ed6e40db84aa0b0a275e26e8f56650837b719631048a26 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d3a46040 | sha256-8447ef871e3949a7eea41a4f00d9dbb89cd8546339d83e163b25671f896b89d7 |
