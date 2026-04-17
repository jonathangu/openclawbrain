# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155`
- winner mode: `graph_prior_only`
- trace hash: `sha256-54c6a7f75aa98b64fa06de64444db8f288aa41bfaf9731cc070d54f577be960a`
- fixture hash: `sha256-899705aeb2321d03b6a0aee78d7cfb19ca0d976080db3e6a3f83db60267852fd`
- score hash: `sha256-a095338e676f6bf1d3a3a2f9d27a829cf8229ac0291ca25ae0735d34fb3e829b`
- bundle hash: `sha256-5c1cfe2a9828785649e3acd941663a37fd21d9b88f8aa53faf918d5d2866012b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee198590c5a4a8c84e2f8fe36017d040fb15fc92428b4d0396417de634b42329 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f291cf69bf17a16d472df7f3be0fbaaa58685aaa5de07e51bfaca0cd0d92773 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e76b26c5e15d822c5a4f23c290a395eae5cc1db76f2249a0147aa20d1fe8b856 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-07b63517b0aba8c784de22358b17fecfcf96b31f0d527da1bbbfff14dabda77b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4d26551f | sha256-964f2d3fe2c5eacce8b9fb7f14b9e2d2928d15e91f1d5d004e2243d40ec0ede1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4d26551f | sha256-8664b48d977a3b69bf7d7ce346efac0f5e416b8424e5d6ae9af9678a6e6daf8f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d2bc8f84 | sha256-00d9bd4969f34e2b8a9b4686fb544878700848b527ef8003e021a82913838545 |
