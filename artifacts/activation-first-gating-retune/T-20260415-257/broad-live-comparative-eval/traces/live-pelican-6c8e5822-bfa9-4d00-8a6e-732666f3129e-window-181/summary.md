# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50cbfb4de0d67c0910ccd1f15acc9132454b767d6a9ef6092fa51c701d086751`
- fixture hash: `sha256-ed982aae33c06dfcffb629c09975a63d396b69570ab9ad349366a4a66aa757f2`
- score hash: `sha256-1c48f17d9fe63c1bba6ddce943858072146fb1cf1c1defd955631598d5dfd715`
- bundle hash: `sha256-1c5ae0f7f07d512350077ed28d333e85d4fbb390b5c2aaa44c41fbca6597bad0`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9ce8acab11edc00f581b930ddd46ccaeed311548b8f75f0398d0e21fa5078567 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-8aabf990836cb48b216349b261acd02cbdf09e6a06880ad1859c14c8d2ab9d6b |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-089a7d698aa25710bd35f9ae288ade9d9bd4200741fbf0d85d059ff78a690dea |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-9a3cc6d346534eb9f5e52efdd0fbf9624d3540cc164527881a9f6a0edc81d322 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-99624f63 | sha256-2754b60bd73bceb9a7766c80afe796531ce0a09361aeb51a8282ee8937cf545d |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-99624f63 | sha256-433d45d5d8b14d37712750c2dcdd9d189dcec7e171b2cfb104cba48e81acff79 |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-eb1b7010 | sha256-c10933a01112bbd932c028bd9eb4e26158e3eb5133c191192feb32d3977a6c8b |
