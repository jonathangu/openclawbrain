# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b95b3f2d8b9d73f4ca18258a6e6b859d4663ae6e1dcb8adbc358ccddee06f53`
- fixture hash: `sha256-e947103eb16b8507ee81b448d2eaccccd37ec8449f02f552042f6857d04bf6f2`
- score hash: `sha256-334285f301f0b776f230182a26a3de936ec7d3711d9753c84716c27bb4f6ba76`
- bundle hash: `sha256-805ffa9dd9a72ea9b8056f54a00e8ec6429643fb70fa8d4e980355ab41270e32`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-45ceb0258ee7c7f1a993bb3ba076165647abb9bb280ad94965584a6223415962 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-1e76655309c848c668421c3bdc7f0649c11404f618331b7121e21a625b9755dd |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-0ed842970119e71e6cff1779472b39743ae0527a3e30ba916d3fcbe8fb67ede4 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5e65d778b56940d750b8574f46673f65a98e9ed3b234a928a0fc7c7ca9a61392 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-959a2db5 | sha256-5245ce87760cc009f997fbc1fe2c354d3a37fa1cbaa36db2d9a1589ed0ffde02 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-959a2db5 | sha256-3127c05d6cf4b637349f1a4a14d5f38a62f98740b4087947ab860cc09f16cbfb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-959a2db5 | sha256-c29f7fe7ed34347dfd757790026af6ee5b99f2522025e591cfdf115afa119e5b |
