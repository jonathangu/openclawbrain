# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-609a8ef08a0005ac8c6d28613bff0743081fda2af2229951c1bce5c2a71dd05c`
- fixture hash: `sha256-6a13457eafa6a8dea8911b77d2fb44eb3c714588ecda4ba2d46120f25504eae3`
- score hash: `sha256-de153b430b5aae9848405d6d2024ae96f0ee86b94e4852c87117ca199decf7c4`
- bundle hash: `sha256-421e48c95a9b7ec45b9f4c275ca55358e2f5b923659f960a75e586a408e318bc`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-529cbc550c979df64d974591190e0c1d456cbd8f7265be9e27a0fc5cbc417683 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-b06010dec2eac4ab97a93abe1ff7f946e74cf2934729aad37b8d26e3b12f163e |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-977afe502c3462b9109a1ecee502e599d9345b08789d49c2b8e343fd5346f53d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ae4b0a06984d3ef0cf00aef01faf19134cd88a133d52f431f9de1ed9be005c05 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-9589564a | sha256-45887e5c3471b2567572e1f42c109b8f99a5b3741dda9ecb55cb2182c5e03ced |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-9589564a | sha256-a51c47d3503498a2c769f56e896747dbecc1aa57db07e02c5bb9045cfe243d1c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ef29e96b | sha256-6af1f2fdb95abf2b7e893eb635fbb124642e51271d2eeab49df62142dcc2d5c8 |
