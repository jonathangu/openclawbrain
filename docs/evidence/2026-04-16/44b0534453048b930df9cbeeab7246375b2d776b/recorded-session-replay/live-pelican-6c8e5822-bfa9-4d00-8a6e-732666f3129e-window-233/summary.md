# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9d936165695614f904d36571a8a48065c182dddc8afd06f7b5a7de26e3d1a3da`
- fixture hash: `sha256-6ad09120c53334c8df0b9f19b852f07c2aa8ca071680e8461d1d0fad693137b2`
- score hash: `sha256-6dd28507d777a59428a40788edb7361f225b66ae29cee0cb6aa68f82fe5f50d4`
- bundle hash: `sha256-9595d08a512f5fa91ac4acf6411005c21ec307835e5a08073acbf18da1dee902`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-64c829bce610fcd103488b348592039f62df2ce867ae9d80d873a4b7e859608d |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-01064c3d05e4f1047a24ad05cf3d527c66d10811857168c1290af65fc2ec47fd |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-0d204a8b679eccf8b2abb1df5266d204a04718f50f6913d7b5e1d072149abc2d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-2ea57221 | sha256-d691137513b330ed024af875c64474e54a21079058a02be70cd3e687bea003ae |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-2ea57221 | sha256-391cffe0d63ab1912d2d7dc8478e87843ebb27dc4c38341bee6f97caf27d65aa |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-a0f1434c | sha256-c50ad0ac5aa6863fda0fffd838211b4f38fa43655a05d9725954fa2460c4215b |
