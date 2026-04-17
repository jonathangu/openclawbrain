# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bc492042fe348d82c21faa673d938d0577346ee06614b38f34d614d883fe125`
- fixture hash: `sha256-166379a9c9e98e60de3e148d45fed20846d7dac8b779bfce9e0299ba405d4f98`
- score hash: `sha256-0a8f53232c0f18d6738b94520bb40dcd73246cf473a17d03f2b441e38d064cce`
- bundle hash: `sha256-51f2b00cd87b63d022ad32b5e6fc69d7e928eb4beebc4e85541f3ae74449436e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df7544610c7c12f9cdd0d8aad84f983991755b60031939cec1112c0295581782 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-edb58ba12c6beb3f48cf79d3769a5f06e527918b2be78bd0daf554a9c21961aa |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9ff72eb683d7be26b4a7a7a92aff9523d3e5dec10c19c2768490fc82411bd380 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8daa34289052d01508c8d90128f31e66907279ded4ec64c08a2bc2c6d342ec33 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-98fa1940 | sha256-0923d12beb847483cf2f4dd75d3188bf52fc22c8e950365a0ac530d9cd717e34 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-98fa1940 | sha256-b186b2c4902e207dd416a7cc9276efae517977369b59adeeb64d1514e04f539f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5c4fcb9d | sha256-dcab9cf4baae8e08c60449b12e50dc33100aab2b67757efc2ed4d43c92af12b8 |
