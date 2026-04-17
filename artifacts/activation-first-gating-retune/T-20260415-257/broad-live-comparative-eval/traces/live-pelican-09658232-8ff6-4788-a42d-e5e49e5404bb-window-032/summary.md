# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3d634932f1bce6d4693c067f779badfb747407b8de4f4dc108015f5fd2e78b`
- fixture hash: `sha256-178c9c975a3f9bee04b778ee3424e4eb908e1106cd7f867502edc61b1de425cc`
- score hash: `sha256-5e850318f431489b287e604641aecb313af5707cd731d4977bf7075f65cd0b35`
- bundle hash: `sha256-7e005db8d58fc6fbbfde8542a8d527a4f922e9d685f73d26f246e6c4676421a0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-252786d108eae05056482d31aaad41cb1fd7abe9a8bca72a4a7a00c78ba84b59 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d71cb76a52108676a3420fbf5a5dc2b7c174fe0482e9582fb675c4d2fc32656a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-47df2cdc6f082e60807c9615c3e67619c3ec87f19507ae9015a69d1eba0e345d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c2cb36eac0d52c2dbcf31a0788666cb0a3d554499a8bcb3a2b48f27ce14e61d0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83a4f530 | sha256-17976d179edbe9f30c833585ead8314c397bb9c7e0aa9bcf253e724e43057244 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83a4f530 | sha256-9720b2fc5f912e7e05b2f9a3cb685cca296aabb43de95d92d91c17028e7d66f8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5b323c0f | sha256-7d378922f0ed5b085c7d7559fced64b1f7ce95589e52d3a9f7959f27baaf0cbe |
