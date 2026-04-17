# Recorded Session Replay Proof Bundle

- trace id: `live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4d327d918fbcc4a369abe7ef4d164f9a5cfa04faf7adb2505c432935e6de9ae6`
- fixture hash: `sha256-6153c43221a8c0bd8b8f42dc9046e70b2d1a03e5bc667d5e9fc62b4aa1f0fcb9`
- score hash: `sha256-5702f983ce3a096a07cf0582c799af33b10fc777553d672cd003c83e3593d90f`
- bundle hash: `sha256-e7770ad57698014e19cae77986fd83da3c29dd04b44020e6ded54c71a3ae1fbc`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d542a8f0c800204dee4f72d85787e8cb1b923c865594f3befa573eb5cd2d9388 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c010d83e3a7c1722e087d962d21bb2e6956329346dc37df864074782c0ec0e27 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4c66481af06e346035cd70ca2795c03234e89a45e83e460203e10f41720b84f4 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-60a8cc5edf9ad478ac41a909b40ff9eb9e5989b280b695a9825af842e1c6286a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a60be8f9 | sha256-4b233d7b56a5c776c1bf39afc6952839fe9e76ffb9e49432467bbbc1148e417b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a60be8f9 | sha256-79427a3e6077150e033ec89dfd49055bdda88325ea4af2fda17ba95b5512d0c2 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-dd7bd5aa | sha256-6fe6754860a921f76ef483a73c125ef5d11eb8cb3f98870a025c2d617b3aa80d |
