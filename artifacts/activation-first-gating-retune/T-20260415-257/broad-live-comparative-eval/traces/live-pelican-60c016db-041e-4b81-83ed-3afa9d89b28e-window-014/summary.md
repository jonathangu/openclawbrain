# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adbfb582784ce9c57067bcd682b42040f9ff5a4fc2a41a6b215fa1e5e63926e2`
- fixture hash: `sha256-1b81b9ebc5b6e57a68ac36d63b63963fa7e0e03c9b05269658a97fc89e8025b0`
- score hash: `sha256-26b275b9369d06515b2f5ed115390a45e36b1c906003695e5caa1fbc8088e71e`
- bundle hash: `sha256-6b09bf62a5eebd6c8912fb803a4273d81750b3b00f817f2de8687ebb3463919f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d918c89a43fc84e7a627af305c1d796a487842c9f1cf040b6474472ff6068ba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe90c785c4c67ce9a258b6339d3514f371594845216fabe4073e8f11240e4124 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-133a5a2b9f4d8b0ba1abc17616b9cbef2fa4eade77ee98098c4a13ef51c0fec3 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3d0b1d142712047ddddf77d6917878b0678b370fc9c72253db429bcb9500cc87 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32041aa3 | sha256-2a0130f6cdc5896e342770aef99bba69cb337bfd9e640d354a5f88bb4d69086c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32041aa3 | sha256-545cb8c06bff5dc12f628d77123bdb107e24fe6e86d16b2e8f462a537262a736 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-68d4d5ea | sha256-9a1590f83d86f2eb1876faa38ed0a735edd66dd12872a22cf795b5f45c56d9eb |
