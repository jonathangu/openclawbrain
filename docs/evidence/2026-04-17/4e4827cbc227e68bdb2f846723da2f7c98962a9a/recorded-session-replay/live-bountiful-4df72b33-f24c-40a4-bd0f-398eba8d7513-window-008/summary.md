# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70911ec1e7805ccec970087d6c2246db12da18117e08ef4135c17a78ab963e90`
- fixture hash: `sha256-5f3ce437bc5a34220be72a905054c7058ccdfb9aee9afb407a944b39db8e43dd`
- score hash: `sha256-767359f309eefa20c4dc1444feec7c1dcd6b8e114a95ee7d257485ef51c849bc`
- bundle hash: `sha256-11b71e1d1cd4e90fba4f5d266d29a8a880c4af39aad584be5b5769f1cf25b693`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5564d3b460f1097de136a88547e9b2bb9e15503e1a0ceed301551bb8e7b5353 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-900cb956d289e414d37ca8ca576b8f61cfb1a1b0ee97382078c1285c69837d8c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e61a99cfce57baa158cb75522a34ed8c0ad8fdfc34fa428be1b1c9583ba46782 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-fb96d84020a80c411082e12331eef6cdcfeef1316a691ad61808ae574b7bc0ad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-856b6c78 | sha256-5c92f9d8ffd73a7d9a27e1e0e4143fbcaa3ab1a08285c16e33a7c3b622419c22 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-856b6c78 | sha256-d0c1606e08859ff6ce161980d9d297645a348c7edc5efd1b864c215013bad6f7 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-b52ef1e1 | sha256-712269f1e9488b23c108c5ebd17430d2de66c397b48f0eb83f280da60a6f0c72 |
