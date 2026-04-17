# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8fecd38f3aa3470c67016a58c02da538613366240f311d73e765e2e999bfc5e1`
- fixture hash: `sha256-9a635fc4466dcd1f01d2e94228a353c7c6a97d36b77eaea2bf2676d0c4e0cb26`
- score hash: `sha256-ea8691992b37f80ee5e21211c34aac1cb50fc289f6611b2cc98b23284ca822d7`
- bundle hash: `sha256-02d4cee4318ef443a8da4a1a25d0ab47560b9dc8552f3be1b1f8e2bd86f539fc`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aaeef480a4d3704484f366d2ff6ec7b8fa4c7b3862f3a1656c04214bee457ca0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-56e919cb7a91fedd6ebd136978289dcac1843bb689d8f72b44cc9c433ef3e3c2 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a13e19c01fee75daae13bfd4faad56d203deaefe93f28e9a82be1de26c9d990e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1fd69a95 | sha256-63f7c0e1dacb0922229254745285eb9f2eb37666a2b45c41a72e95fe83eb1f77 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1fd69a95 | sha256-843d6871bb0a963e176d302040d3454e30e624923eb99134404faeb0ea7b6f26 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-0c7bc844 | sha256-d4909ff7aadcfc4a3ae3127bfa2475f3447eb3f87afd4f4acc66d0eddc7a9a3d |
