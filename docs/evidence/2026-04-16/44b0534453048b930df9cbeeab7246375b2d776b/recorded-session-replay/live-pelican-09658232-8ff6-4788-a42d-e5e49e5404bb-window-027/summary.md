# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c46bcb381e0fc3efac0e09438f41834359285b619fa2a6877dd357e64e821071`
- fixture hash: `sha256-1ef85417b722b0f394a6f903af4947a78bca3d01432416a0cb17a206ec104c37`
- score hash: `sha256-8e4cf4fa5276c7391d6ac7351bcb0c8d220286ddd32897b3d97564e6344dccf4`
- bundle hash: `sha256-eddd10db35e18894bb43b192b62722192622fb0ef60e85da6c207f7d3ba73d95`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d30e174da168a75e39ffd3536c03dfe75f2623e328eaa1807c5de3d00819572a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8c804115047da0e7eafa95a32a6cf77340e70aba380703b9b1eac84e0f5df2bc |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-27f0313fbeac9e42a8abae09b6de7554dc13cf07f4faa58b4384e3186b218355 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-955e8116697989bc49bf06791f0cf789ee108fb74b69fd05cb6c85f5564d2e39 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2b76b59d | sha256-72c14ebe7835061969023c649731a9c055b7600a672352bbdbe654eb237f3038 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2b76b59d | sha256-804ae7b02b86103bae4363c5690adff711da3fcde8687d4705adff9964a69946 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-1a477a44 | sha256-ee28f704a96267496b79dfcccaf3a2bdabbcb113cb3c5258a7a6b980c6b179cc |
