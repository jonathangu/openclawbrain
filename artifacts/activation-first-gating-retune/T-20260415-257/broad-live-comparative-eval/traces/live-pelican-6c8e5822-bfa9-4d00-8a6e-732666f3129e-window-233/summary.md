# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9d936165695614f904d36571a8a48065c182dddc8afd06f7b5a7de26e3d1a3da`
- fixture hash: `sha256-6ad09120c53334c8df0b9f19b852f07c2aa8ca071680e8461d1d0fad693137b2`
- score hash: `sha256-7e4463ff3d49bcae1d46a3540d344671ce4a3cb612f48c4ad48f6937d32b0122`
- bundle hash: `sha256-dfd198e7a254be6b1ab1a8ca42f8959838ab7f2c001d675752fde5fb0e4c29f9`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3329ac350a9048e47f1760a5c97b317667c0cdc04bb3d7fb2085cb6158792e13 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dff5b4ab11490f8cea7da1f847d96cbca7e4bea8f4d13bafc451ed3ae1025a0 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e8ef7443b5074510f4382117edd854ea5ab97ce7751fd61a8a83f0f7fa8fe344 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1be25c80919aaa3bffa780458e0fcf780051b051aafa72af7d9d961db822a74d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-36bebf9a | sha256-bfb8e1cf6e1bc1759cff96f1887214073b4ddd8bafce2175a6df62099288cdd0 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-36bebf9a | sha256-c1f3be0733f8774fd471f2a8605e34cee42b32395c345b477e09501a0f102d71 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a90a90c5 | sha256-f82866b1206e67a88c8b27acec74dad704edf6e9c7802808d2d7370528dcf6df |
