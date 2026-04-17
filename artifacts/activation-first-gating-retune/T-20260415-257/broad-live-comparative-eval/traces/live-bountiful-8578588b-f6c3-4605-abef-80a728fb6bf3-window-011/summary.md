# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-893bbae402dfd268ed32b5d7137ff140cf694f0744998bbe068d3b879f9ca62f`
- fixture hash: `sha256-fcb57e82ae27e8603221264cddf33ddbfc96e5fc7bee09bcaaabd6c496832873`
- score hash: `sha256-3ffe2bc074b6d3ace07a1ebc12032eb960420797832fc7d57f99212d5568ed38`
- bundle hash: `sha256-c20466ba4a839a384e4a442fcf251d64146ccbae5ea31a47ec6a4bb1ab8b345c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-593942095b3dca97714f70191bc3ba85569cc1817c4fcdf560f63906b71b3cd1 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ca4a19dae8c87ccbc80e72b74e9cda1f85b81691ca7ffca41f8791d2c0b62a18 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0453c15142d45afd9bddef5f21608d1ab6b3131f42f24829b15fa861de3c8e9c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-af38636ba7e015f683d5b66f31e186480ff42f4901b75c6014aaa5a3a4bf24e3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-15f40f99 | sha256-73248df1bc6aa7d2dbe64a5df8192df232035eb610f5d3e6b2c0a1f5e49c7de6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-15f40f99 | sha256-3b4bc7f4225c63b1cc0ae361466105399ecf1c7cd83d8771f7317886f2ea19ca |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d3a46040 | sha256-a959330bb8d2c2ffd4d4c46f522668e445fe95bce60bfa88eb18b0988479a459 |
