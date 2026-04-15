# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bb64d18c7380c29a36adda7f18b9d94028bd2ec79c3f043249c311ff96079b77`
- fixture hash: `sha256-4d9c945d16c80ffc64625c9921a10c1aa73d0e2d0d7dc96750c287fa87ef0a3c`
- score hash: `sha256-1f781eb2879a6dd0cf813cc9732a20787a807f825b230184b8f7ed6f076987b8`
- bundle hash: `sha256-e5799ae0b28e99201e1551da099d3653c27d311355e3646ce870c925c29611ca`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2057eeb74350d5472d7a207a6cd23d83fdfc1cbff7a9da70502d2c9709cf85fb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4295061fe6226672d111aeabeebffd33811ce3696a8e7ed89949d0633412b71a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12b347cb3c7152a414ded74e4593cae7687673158bb9ec557f0b3a641f73eb93 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6bb73c7199754ae90afe396eb185a008ab991d9b54b39706c77ad5b6951bdf71 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e4f22a2e | sha256-2a9dc5f69c8db54b78eda4bb4de8041282bb21b41fbb5354c9866a8137f7fb21 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e4f22a2e | sha256-4fcbfdf7fa90e6b27ed8dc81d735fb7bc6d297ea2874e202678c63726376bafe |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e4f22a2e | sha256-2a9dc5f69c8db54b78eda4bb4de8041282bb21b41fbb5354c9866a8137f7fb21 |
