# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c46bcb381e0fc3efac0e09438f41834359285b619fa2a6877dd357e64e821071`
- fixture hash: `sha256-1ef85417b722b0f394a6f903af4947a78bca3d01432416a0cb17a206ec104c37`
- score hash: `sha256-a848f5276a7c50812077aa6d60869d668f77607060e531f8cb9477952d2440de`
- bundle hash: `sha256-05bff8693f9ca0f4f130cc5f103e248a94e0987ff0a45752641b584a9a6e3725`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d30e174da168a75e39ffd3536c03dfe75f2623e328eaa1807c5de3d00819572a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-99e186e7f328485081ef44496071c1bd5a7ec5c18b9ebaee9be0188ffd827c5a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-be6dc5f35c7d75c7880152080a29a586b1d659e096fdfeb191369d5384a86abe |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-cbb5c03fdd685603c6cfdca394583f82bc4ef819baedd6e267bc6ff7667fc837 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d27a4e50 | sha256-1e517ebff9b6c1ee28eee486c5753a9b9cd7184386f4c40fdfe4d313c518e2ac |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d27a4e50 | sha256-083cbd709f6bffee9c8c38cf91828423767dc720ed6db9e3fc7496f13d825cc4 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d27a4e50 | sha256-1e517ebff9b6c1ee28eee486c5753a9b9cd7184386f4c40fdfe4d313c518e2ac |
