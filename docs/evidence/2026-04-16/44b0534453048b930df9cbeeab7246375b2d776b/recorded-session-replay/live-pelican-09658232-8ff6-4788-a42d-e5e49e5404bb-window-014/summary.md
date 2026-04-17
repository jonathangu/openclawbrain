# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a2e597e92fe22d4f55c094b8ed54b6a9af6fa4591283d89702e798da892600a7`
- fixture hash: `sha256-622fac4fb2f464038d17b973948a3daa701456585a35960e995213dcda72d3b1`
- score hash: `sha256-a33d4bd39661c57bbef51d35e5eae5352303acb52db476ea1fc8dd8873d7b9d1`
- bundle hash: `sha256-191b05c6788812e5635529e122a160cfa39cfe4eead957217cbf5a174e41c2f9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-937c7f8b3de6cb0ba567e2def00dcbf253af96c301f9c26d07a7c1aa6375230e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6cb797aaa462d4e92c636da66daf5c8edb0ff87ebfcd27128677b8c04bd694d4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e97492b95fbe0b134724d3873e2d11db6e7f4f50ca41fabfc97c58fe1df0c66 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-10ebfb1c51d1609434680e7643f0104f17859640d367103ff7033633fc3dcd43 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8b69bc5b | sha256-99a692297c9f315aaa75aa7f11b1e4325a47cbfe306414f799121dbe01ce0fdd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8b69bc5b | sha256-b057ca7f14f13c840d9f2c39769a2e23497afe92dde756ff14dfbcfe30b91f9b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4f859e24 | sha256-69220ac002ade2f411ef391eae0ab5333cb8aaf2d3e7d40138478d15828c9c83 |
