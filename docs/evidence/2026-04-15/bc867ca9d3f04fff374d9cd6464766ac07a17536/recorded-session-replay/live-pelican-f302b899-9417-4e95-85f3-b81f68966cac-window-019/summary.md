# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a95260c17a69374ef7a9ff20490cb415b09868b4babdf035ba541b6d82beb5bf`
- fixture hash: `sha256-ff74599acd0d3d5ad2046fb7795a787fe8fa0e70837c98ae65f89838fc9f50e9`
- score hash: `sha256-510efd90a0cb9b977f7465225988183e5ee4a68cb9697975bbfd39cac0bbb58c`
- bundle hash: `sha256-0da0642d94927e64ebf6184d3576b9cd1d064b04f1a9b96b259141c5140194e6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30f22d5abba84d169a9ef0f72b28eb7bd4c2afa26a7910c928c371f416decf04 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-89773a31335bbcf139a3127fb77915a48acfbcb4e0213ab833f5b84b4010300b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8080bf54238466a0aaeec67da04546cac8b528e2a2ab9c974a5d347002527cb2 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a3e5d20b82fc1d2ad81fcf8366295598f643f285a09ac1a54052892776a1ef3e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9cc428f5 | sha256-8f140c1345fcfddcf0fa302ec10a33b7e12b1f2acfad595cf017bebaac327b04 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9cc428f5 | sha256-bbbab31bc26913c7913650776c2b20351f8700e152f9ee1f236a6a1210373aac |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9cc428f5 | sha256-8f140c1345fcfddcf0fa302ec10a33b7e12b1f2acfad595cf017bebaac327b04 |
