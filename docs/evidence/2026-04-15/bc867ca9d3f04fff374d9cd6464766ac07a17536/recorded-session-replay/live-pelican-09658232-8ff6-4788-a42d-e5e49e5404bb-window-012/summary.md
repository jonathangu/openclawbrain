# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50bf6e94091f3556fa81577b5a708e4425c4e417b6705bd10df603b7966e593f`
- fixture hash: `sha256-18f79e8bc777ea9555de29a01a8501d21c8ddf1c9ea32bbf589d49b4f4a3aaeb`
- score hash: `sha256-24a2e7889f0dffce8411a88b4ed587f1290d2584d42baf1b7629885ea456b203`
- bundle hash: `sha256-16e19d3f651489f9b7905757ea058b0ba89f13f9983a93cd2ec234f91017a6bb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12a31c86174cd8cb94081745ed9b01e9e8efd75760d60dfa60b0b81778821ef5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1e716ac099149266b1c4f936ef1711995148e527183f2928243105d49039281d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54c7863bdb1846d873ce6f162456cb6ae5693b4328eb8fbf559fc9eefcfb5f24 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d2858865540b9f3d02da145f5e77ff1a3b0c6e4782940343f263533f2ce67cf8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c982828b | sha256-f65f584c8d0cb77659f9fb5d00a847c5c7039feb0608a1c864c5fd92e44b03fd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c982828b | sha256-b47b925bb0f8ce6eaa23790263613bc0a5707430eced7826fb881d86d1637a9e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c982828b | sha256-f65f584c8d0cb77659f9fb5d00a847c5c7039feb0608a1c864c5fd92e44b03fd |
