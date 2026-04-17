# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050`
- winner mode: `graph_prior_only`
- trace hash: `sha256-11c3fc8419968b3646c9393b68a3489cc8f59dd899a6b693d7a0be2f87ecb9f4`
- fixture hash: `sha256-de224889eeb5e399123549dc0a76f80a745378d0b306a7fbf4d142a78dbb77d3`
- score hash: `sha256-7bb7131d299eb1f87e41b49c3f84b876f3b70ea09e929d1f92b0391d57f5f4cf`
- bundle hash: `sha256-aea6dc4d215f59348d538d588133cf2d615fc095f093a05a544a2014a0431b67`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6ec78224bfabb767d014be29075a2dce7e842b42fd97bfcab45b7a968e220fef |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8bc7a48b7dd21698073aca7c9c2cfb949831067c2f9c5884af832af746967d5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ac21b82a76b8ccb4599630f8f0f17e4eea400299b3b2ed34e8bab6b137d2a106 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3b5f108950b3ecd4f50216fb3728f3943a8fb42c8451ecdf914c0d16706a2576 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-65528b17 | sha256-4e1d686eb161be6f4b391e731003a630b49dd2413fb08e7bdde4bf60f3c8b2a7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-65528b17 | sha256-77b0c745ac946779dff38ed20422e304372de9e90692a9cc59f4cbc4c4a8018b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-627280e2 | sha256-b636bf21a11ca6c3604f7a4ea12a9fa49358e944aaeef74e45aa9f78c65486e8 |
