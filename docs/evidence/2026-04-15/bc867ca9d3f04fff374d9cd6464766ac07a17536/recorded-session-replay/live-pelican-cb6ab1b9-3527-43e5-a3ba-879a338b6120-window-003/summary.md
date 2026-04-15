# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8226e38f2d583af41a4327f3b8df4e5b434ae18ebbdb89d67531a4a854359a44`
- fixture hash: `sha256-e3733e9aa09beb01fe43936408b2069d985913ff1742752483045d9debec0829`
- score hash: `sha256-fca3718743d8585b25dd5cbf4bc42911680d61153551a8eb0279cf9083add736`
- bundle hash: `sha256-4320d93d939a8f7f0a5f4520a7e5bc36533b2fdfd3255d3f2c6798f3f08cfa72`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f1f95cc8e218fff5d5905cf899fc04d3d3c62a98c1d684ae5ae4dffaa6f7bd10 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0ccc03fb265e297e84ed0e8e50d7d57d027f5418185a3ba9e64c6de84444e462 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-985cb89e5fbd7adf40c1bb98f6bc2ea9c2ead38050fcac16ee053695d0b4d156 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6715e92870d1fc17d589934209b9ac0790b2c3a5849fce0f04a4d2591c6abb20 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-86f07e2e | sha256-9cd6723ad103da3c6599b82204459d509ad74dc2c2d488f4e5b4bd7fde5daee7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-86f07e2e | sha256-b18308c53add67ef28c6c547edc04e05b0709723b48dc70f19934096176230ac |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-86f07e2e | sha256-9cd6723ad103da3c6599b82204459d509ad74dc2c2d488f4e5b4bd7fde5daee7 |
