# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f10c00dc4efb180b5273900a9e561d1c614344a77050359aaa2d54aa27cc20d2`
- fixture hash: `sha256-43065829df1e95ca79dff07d99e5773679b5561b6bbdd3945d317201ab2cca51`
- score hash: `sha256-8af3243b1eb2dc8e8bff9a7e93b746dfe724aac85c319127bf6d7e1a35c16630`
- bundle hash: `sha256-28ed87c780c1c5825c08c1f12538a3a56edd5231e251f63260eeb6b2ce405fe8`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f1460fd13a644dccb389d5e4bb97bb20a28fa61d221da193a36a1bd2b7379c0d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fabfc8432b41b7a0f5219fd7c89ad4038fb734f42e449b55040cda4f3cf63c65 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5a0d05af5e2778a7f6483833835a31ac7fb8afd28d0a0c30a7985ddcd1630761 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-de97910108b0ca9b09462b3b7becf308700303e009def85cc620522701cdb6fb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d4b3f4ed | sha256-8df3f6ac69aae03171d0a66887bf00e5b03300795fab5ee1ede748b9763172cc |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d4b3f4ed | sha256-8df3f6ac69aae03171d0a66887bf00e5b03300795fab5ee1ede748b9763172cc |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-90f386a2 | sha256-dbd7c78a7d5036f0c82808372343ac655063b0e5f66cfdcd8bef8029a47a781d |
