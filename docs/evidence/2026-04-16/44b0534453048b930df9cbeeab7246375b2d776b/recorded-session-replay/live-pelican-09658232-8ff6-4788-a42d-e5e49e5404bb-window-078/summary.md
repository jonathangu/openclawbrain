# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8fecd38f3aa3470c67016a58c02da538613366240f311d73e765e2e999bfc5e1`
- fixture hash: `sha256-9a635fc4466dcd1f01d2e94228a353c7c6a97d36b77eaea2bf2676d0c4e0cb26`
- score hash: `sha256-1798cb09d69ebcaed44b22fdc166f9e4954d00a124b05f4dbd7e54e33a9e2605`
- bundle hash: `sha256-d6ef28f0561d2c104c498bbff21ab5a8526bdb63e0b3d3bcc876001cd51f14ac`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32d500730de7d73f2a2bf38e8b78d2d6ad04a3a58dd8029622c951f7ddee70 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a53bfad32aa86b2816d395c3b2bdc0e881a78fa722ea2768442afa08c1e0358e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fc13aed10107047c1685567f64c11d168a1c7d8630fbbeb3928d28cd35202710 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8cbb3ed517bdaae3833895cb7ff9076d8e2310424e9d7b83253cb220c8cfba0a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2d1fdf00 | sha256-44ee50736ae463a7dbba77684f048371467f055799781e1c565496eea8ed455d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2d1fdf00 | sha256-3ef9131fd64acd21857517f6d5645bc9c8bceec3356f6b3fe57dea11d1ba7a65 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-19c50caf | sha256-38b3655dcd018e0fd2802e0e1129eb4c127d405c5acb98bdc1608ed3de03829a |
