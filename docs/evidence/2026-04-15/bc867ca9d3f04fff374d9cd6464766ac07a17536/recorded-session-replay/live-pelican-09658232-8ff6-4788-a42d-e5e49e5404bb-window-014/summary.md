# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a2e597e92fe22d4f55c094b8ed54b6a9af6fa4591283d89702e798da892600a7`
- fixture hash: `sha256-622fac4fb2f464038d17b973948a3daa701456585a35960e995213dcda72d3b1`
- score hash: `sha256-b3f255d939e7d83edf77a0c773b7484899258cc6d4702a2cabcf7416382c2fdc`
- bundle hash: `sha256-454b4da027018c644da3bd5b3bf70f13ef7fe05cca5b0296022a13bbe7896f82`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-937c7f8b3de6cb0ba567e2def00dcbf253af96c301f9c26d07a7c1aa6375230e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0a6ae6103bdb611f59bbc96e15ad393fc14ad19ffea40f4c88664e40b32958d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c295ab1601e9b6468ffa40b861d496cb50571d4f616956b43431296c6d800e5 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ca5562d68ba92722f85851c8147f5c13171a1a2658490817f9baf58a804ac122 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-83db636f | sha256-8424f32f471c8f19cae72041a6f41377a906c3b9a3e6952973f077c42adf5bc9 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-83db636f | sha256-9a257f2eb5ab3537c0bf5834df8f2d4127884c47996347bddfc4e99634ba1043 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-83db636f | sha256-8424f32f471c8f19cae72041a6f41377a906c3b9a3e6952973f077c42adf5bc9 |
