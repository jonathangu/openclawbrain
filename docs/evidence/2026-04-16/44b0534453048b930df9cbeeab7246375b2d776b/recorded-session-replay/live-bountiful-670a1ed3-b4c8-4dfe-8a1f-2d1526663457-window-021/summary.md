# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-512c430e649faf76044870db1348b61987384f6c4b42eb2624038c368ab6a4bd`
- fixture hash: `sha256-6f2c5641408f7a03798669e19a288492bcf8f6f0b8043e459e2c72b4bc2ef9f6`
- score hash: `sha256-615f8255b5aeca3b5cf3ceb676de0f1f4bed337a02d226653ea2d16fbbaf5b94`
- bundle hash: `sha256-9d9ce380042a9ee383c6df0efd91534fbc87234039b23f5cc7282d05e04d9829`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fc8b4083586f10fbbdda0686c1eb4cc964fe1c89c35a3824fb52431cfb03e36 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3d2c2e67f52c9275d61db8aeb33a3777344c469fbcf2bdfd7e9537711d8bd02 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-43298d4da393ace5087c53915afe7805a0b25f3a7cb78c742a4c7188db90fc80 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-bc50f124fd4160da8c9cbbf0c69aaf4ecd549ca2f3cff76b4a111269e9ff400c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1d26da2c | sha256-4aa8b56e14c587a86fdcc4eb7dc0caef57ec9eedeac213c8c72a3e4a4ece4ba4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1d26da2c | sha256-f5e0f9b2f8a7348f08d43ea109479cd5d05ae9115123a8ebbba29e0ae433e90f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b9d4d647 | sha256-8faa3440a53d6ffb660ba80e853b77c6b9c42b742a9ee02c6817a5bf4a7b66db |
