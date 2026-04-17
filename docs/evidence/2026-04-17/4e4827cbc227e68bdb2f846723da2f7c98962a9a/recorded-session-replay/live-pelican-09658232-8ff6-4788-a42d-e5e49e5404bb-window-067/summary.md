# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b511b4ac1b0719d0780ec2c9e18278c04d48d286fc6710bb079c3e2eba6029e7`
- fixture hash: `sha256-801021da403c0e7ffabb1f8a7bb11de0378ca6fc16e45764ab572505b0e2f302`
- score hash: `sha256-f9ecfa0e77b8e7739ae2b66e39ce7e735f01eda38b0f258d02c67e84573107c9`
- bundle hash: `sha256-aef596e5bbede69564342b09b189e9e9afc0f6c164e2e3261a5fbce1619bc687`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88537cfda5a24580056c21d94f97ea80249672c6a52c1a8bd0de62e2aead80ad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aabdbfd44ed386684515c91748c12c2799651f61de87b910218a9acbdc01f8e2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a257ce1c9ba315578d31ae261fd1800641fa6135bd3e5b680a3972e9fa226f69 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4978768f02e85500854a98d36e0ef5d60a9abfc6326a317ac28f9d5268fb6d4c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-933a9880 | sha256-6b168a3f8463a2ba7ec698fd0c532fec13286d3e485b3d772d04b690450cbecb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-933a9880 | sha256-eb4e4173f15c4bd914b32dac8c1344bc27b9c317238ccad1de4544f0ad1df06c |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e82429af | sha256-aa289116d2fd4fa46ebaaed9075c40b7f1d779eaf82cef1bc23a7d5c2ca02995 |
