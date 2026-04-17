# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-42dd5ae1fbc52ab37ab26b7eff707ccc814072dbaeb4cf80246f57beb5474c7c`
- fixture hash: `sha256-dae1773b38ede59c62f735546227926063dcc22433a680794834acb15197b82c`
- score hash: `sha256-bccaaf5edc34b6a164e5adaeb1f4eb33dd401c573bf9c8b408aefe55aa2ef085`
- bundle hash: `sha256-6a0335e4020b31203281dcacfb7c805b9923279f5ea6026e46355ef708fb44d3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97339b57eb3bff564bd492b91102981f9054b332cee78d9338b804ad8b646434 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d4c38206dd011ed2b9f4816776feb5e4515c8d073d21aafa8e855d675e7e6479 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-09fe2d91d57995ce987fff9ba091052ddc0e1a84dfbdc28c3ecb07bf2acebc8c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8dada51d57b88d21dfbff9ac5bdf21865c0ecbcb7a4670a5b560118c3d925240 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-04becdba | sha256-7d9bd3f8915db79c8cba1e72e6a0c20ccb072bb272eee0f953da63dace1d4b08 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-04becdba | sha256-a5e2d41868df1b124071356cfe00b4ef88a908c84e55e8139d60bf5671f3fa4e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3efa6623 | sha256-ca0045837f7d81c9061b00ec80d70719f31d22d5a3c72b548eb8131657014ecc |
