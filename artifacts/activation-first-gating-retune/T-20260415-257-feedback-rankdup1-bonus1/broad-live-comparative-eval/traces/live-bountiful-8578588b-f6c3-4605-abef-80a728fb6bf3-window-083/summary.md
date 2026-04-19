# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083`
- winner mode: `graph_prior_only`
- trace hash: `sha256-57ccc92788c2790c0c3abd5be28501a9f0f77e46149a354e0d4dac5d8146bf48`
- fixture hash: `sha256-49b325de5ef5c4cbab453f7084ce8035e6fbd63068087fd5530b66fcb0390183`
- score hash: `sha256-56fd5f6dd42806547d043fd2ebb3b4f88344eaf08f94ca5a7c3fca7d38fc1f24`
- bundle hash: `sha256-0f6e167ca80070d034249487a870a0278575dcf9942682029249746c47ae63ca`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b5cfa06ec873c57458d2a0e78a0e3bbb2620ec1455300c12f88247e370c3b4a6 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-3c9becba9e2d5d9313cb1ace739e76c8d4219a43e31e90696973cdae55bed233 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-52b8ad80695efe0c76ab5b3362b3ab26231e6b2e229dcccef82f4aa05ba8d713 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-eb49b13719129f523bced2491e39e9407a163bfd0f3eb61fc42971b2d2648c00 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ae15fdb3 | sha256-23ee0c549427be291fef9bd2894bd4b041c94d0df9f1e5b661da60a14e3bab8a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ae15fdb3 | sha256-5696a4b09af09d4f2a1b48c6662da32e4ab94e20a4a6fb73b2864ba75f2129c4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ae15fdb3 | sha256-23ee0c549427be291fef9bd2894bd4b041c94d0df9f1e5b661da60a14e3bab8a |
