# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3c2d2ba443dbef189a04f697781c21859dae784757f070f6d624a5c22c1fd87e`
- fixture hash: `sha256-d5090234178376a892e6c521a05dfe5104bf688b9e6c7c68cfaf8797d0e0e324`
- score hash: `sha256-51526d1c50ff19ea2863f5a7740a7ed769b65f20ba4e5975f347cbc82f36b8a9`
- bundle hash: `sha256-d3b060468b3f680380ebaf7edd250b146fe25ca846eb16f7dd5dd5526df960c0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30cb0b5562af7757589ca1411482395b52af039eced1208652e3e0610a2b0728 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-551f6ebc64cb489d8e41833dc7a571e7d640be86518fc869d1bfb8c75b7d0926 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-69a47f8c4fbfcc8d39579565c637d76f50303d5bb19967f53984ead734095e9e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-76c222e8d62c6ec03f2283d23062a86331f06738a46d19d57c3c9413daacac57 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2fc4d48e | sha256-bc2bc25423e36301658286fb7c18ce8c435bec4c6a1a8678be93f0a5f42cbea5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2fc4d48e | sha256-5838aa8fd635188a3c23360f55ce7fcf57b28dddacab9f485f4efcaa0e357761 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2fc4d48e | sha256-bc2bc25423e36301658286fb7c18ce8c435bec4c6a1a8678be93f0a5f42cbea5 |
