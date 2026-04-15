# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97f203f8cf6e54b5353937fa7c2a9de19dc80e0c9cd1f7a7efe923af5d952db1`
- fixture hash: `sha256-125dd958dc76d20d11fb8d8f175ae0fed91a68b28d91d5171d2943217403837a`
- score hash: `sha256-95860c9276440a4abba22d5b6fda37ae0e1ce03585d11cf8424bd8cad7480be1`
- bundle hash: `sha256-8c8c69507588ee0424e82986a7b628beaee121c3abca153ce9e573cb4338aed4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-511d0fcc64563746c6c18b192b94f28492b7e276c306dddc6df1e62d381fef89 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dfd8bf3d753e16b2707e8a01aab630d7d11ea93ea0f86356df0311f0edc98776 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81ef224e5cdd80fc9dbfbf3dca9e1749b810e7a81b6737eb77a662102978905f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-33051a9fefba71cbd6906d0105e9485725e569293e27d64d000ccce39b17e006 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b15897f0 | sha256-7e901f0870c0f0d40eaec02dc08a5a123f99f7574060b4c13a42dfd46a86727f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b15897f0 | sha256-b55497d53ebe0411ad773fc94280412ad6e2bf95777e38d5f3d326315eb27830 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b15897f0 | sha256-7e901f0870c0f0d40eaec02dc08a5a123f99f7574060b4c13a42dfd46a86727f |
