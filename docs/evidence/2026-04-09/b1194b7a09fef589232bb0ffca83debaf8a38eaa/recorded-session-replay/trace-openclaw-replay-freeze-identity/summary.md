# Recorded Session Replay Proof Bundle

- trace id: `trace-openclaw-replay-freeze-identity`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f69fc254681ee124b61d343539bc571afb510dbef06d8cc553fca8f4a3781603`
- fixture hash: `sha256-7a70449faa887e7ec02d5a8115792adc662e98498339bea4d10387f0ea078086`
- score hash: `sha256-c50b91bef701e0a99ea6ebb5ea333242825b38d6e4f0310c6c5fb28166ae8dd0`
- bundle hash: `sha256-5caa6cb3d3194941f2d818d166da1975ee7df09858f9c2c898ec03c99df63257`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 1 | 0 | 1 |
| graph_prior_only | 3 | 1 | 1 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 1 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/3 | 0 | 0 | 3 | 1 | 0 | sha256-d0d7ec5bc630c9366bf082fe9e3df18c7cc3b50bc509aba5c3b112c7f54804ea |
| vector_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-e2baf5d6eb77427f9859a5651624fbac23ccb5153acf77fda32fe3c109729dff |
| graph_prior_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-dec219a4d91ebc1fd33aa6a44db6bfcc2f4f43faecf6e2d38132f2dd6f9f9827 |
| learned_route | 3 | 3 | 3/3 | 2 | 1 | 3 | 1 | 0 | sha256-1c449dbe5b2a4bcc2bc56fbd76b1fb2412146e67b8a9f96f8b1abb71b40e3991 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-3 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-73daf709 | sha256-5c28678a8abc35e5fc7687bd89ffe2168f77e74454876d9f6547f4434df0a46c |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-73daf709 | sha256-d2a7fe080941ed9e5084dca9e58118cc1e1fb2365056c9d98e2b02de2870b840 |
| vector_only | turn-3 | 100 | yes | 1/1 | no | no | pack-73daf709 | sha256-5c28678a8abc35e5fc7687bd89ffe2168f77e74454876d9f6547f4434df0a46c |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-73daf709 | sha256-5c28678a8abc35e5fc7687bd89ffe2168f77e74454876d9f6547f4434df0a46c |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-73daf709 | sha256-d2a7fe080941ed9e5084dca9e58118cc1e1fb2365056c9d98e2b02de2870b840 |
| graph_prior_only | turn-3 | 100 | yes | 1/1 | no | no | pack-73daf709 | sha256-5c28678a8abc35e5fc7687bd89ffe2168f77e74454876d9f6547f4434df0a46c |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-73daf709 | sha256-5c28678a8abc35e5fc7687bd89ffe2168f77e74454876d9f6547f4434df0a46c |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-0a60302d | sha256-994d27715193225bb73086b6a46a3d48945b6447a87401f5d7b7c1571d97ff08 |
| learned_route | turn-3 | 100 | yes | 1/1 | yes | no | pack-0a60302d | sha256-dd48e45d6598eb3cc5acd1d1f3f50a90d764ec3a8e2781e0efb74b44ea3f7789 |
