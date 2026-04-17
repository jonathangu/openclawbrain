# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e1899273f160788957979a298d976827dfb7d2c8980b1c161a6e0c69b405f12f`
- fixture hash: `sha256-e3a4578dceff89673c40bbf12c9b294dd97be3ba2d82b9f266209970182a5648`
- score hash: `sha256-a037c5b4e02e365f6c3f52cbd9b5908089a9ea0e80b5e5afce36f5a0c1638e53`
- bundle hash: `sha256-bef26eb1ff3f4a661db6d2bc8ac85a73cdec4a168058344d8c77f739f047ef25`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ecf5a06a6508fbef20c40ee36944ffad441534c7ec83a389bf5c81a0f73bcb66 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2de61f9c455b7e5e43db8efaf8ec2798066c76d5f5fa6f608e5087945351c0cd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ce1b3c0c1923cf5ed1085a34f8ed1b35df02ec837399cae7791959c2a7fb7688 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-faa50f59094effe5cdf4cc598cc2855b0809b57342f84c02dbdb14b59ec921d2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d7bb0f42 | sha256-ade0740aecde1a8f7e2e5664aafa020f5550e5cfd86c714a8c75c0f430accef1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d7bb0f42 | sha256-46935fd6678bf17d967e3bb81f0a75bc00100e6942cdb5b7620aad19fb71aa60 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d79ca725 | sha256-692449b61260fcb4bf27df6650b3927a3361d7e7a3a7294a5b1f75f4fdf3ea40 |
