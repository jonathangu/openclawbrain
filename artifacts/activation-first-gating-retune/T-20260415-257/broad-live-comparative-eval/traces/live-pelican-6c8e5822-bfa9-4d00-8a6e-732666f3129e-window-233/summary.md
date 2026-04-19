# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9d936165695614f904d36571a8a48065c182dddc8afd06f7b5a7de26e3d1a3da`
- fixture hash: `sha256-6ad09120c53334c8df0b9f19b852f07c2aa8ca071680e8461d1d0fad693137b2`
- score hash: `sha256-8ad09962cc830b5f1234452bfe99d725e91e7eb0aa5d1a062bc01df4ae597058`
- bundle hash: `sha256-04b232933a36ab10ac26e8d47527b3fe6eec3b3b0b40e40d135b5aca0c4843fb`

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
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
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
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-d823e4a7b83fbc409e911965df3fc505c6825b46c1fe8391e5f12b1191712b27 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-0028539f89c22239cd4d0082810a8cdc146df2fefa6809d294ca003fb71bb693 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-711f192daa5ec86dc6e7d357099754e614ebc7ab1ad9e825c96562247194c8c9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-8f36cb26 | sha256-22297590514d650761fb2269f903ffdf60f25d10a3e9fdeb1b1758e1528968d1 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-8f36cb26 | sha256-7ff8ac932ee1b9c64af26229bd2a7690c42871a25f20805ec47603c8fc0e1622 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-8f36cb26 | sha256-22297590514d650761fb2269f903ffdf60f25d10a3e9fdeb1b1758e1528968d1 |
