# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0ad01cba10197b8b05f27f973f11a9d5ae36ed08ae5a28bf29003ec749435fdb`
- fixture hash: `sha256-6830cd222cfba386b49de3a4d46620d84d8c333ee746eafd9c6c6f8ee2dfa95c`
- score hash: `sha256-af4409b33951f1f992b155f6840ca20773daf7b1221a4fb2dc912b708b2a93d0`
- bundle hash: `sha256-6005c7e0f8ed8b672a18aa35b518f4c2106d08db1a5916541b4a74b9b3dc14af`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-490c3e2fb439b8787ca9fe3cb573a9c587cc3abe414e2faf637ea2b84d91d268 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5277458c0ad4ff0c64fb3613f31b6080a3b9b498803fcd4bf97169393cb8ec6b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8bef88c48bd985b3e70f09894c6e82a0200d094f5365d5a4706ae14f0a4c198d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6db38af74b699672c3347d2c7a06eb7248a30e2b455e2c80a394922cc3f59d68 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0784d9ca | sha256-9112a083d32003b22fa83111ce7bae6651052562e646ab643310090ccaf21831 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0784d9ca | sha256-0d9ce1498e90b9a87892e0945dcd6d3955e8326fcba9e3926871a912b5284143 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-28157d93 | sha256-d585794ee5390eb64456095e6b5ebb41c4f2bb0db928d14263e3da3e3cfb5e7a |
