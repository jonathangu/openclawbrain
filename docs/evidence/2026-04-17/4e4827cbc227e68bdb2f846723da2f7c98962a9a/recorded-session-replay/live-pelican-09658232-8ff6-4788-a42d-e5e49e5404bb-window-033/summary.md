# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033`
- winner mode: `graph_prior_only`
- trace hash: `sha256-694cf444538867e625d49591796eda7824a3f9914c6d50782ffa8d2751091f0e`
- fixture hash: `sha256-cdb2b18e3a901c8928c86a3e5d6789c9de0d594dce56653b0cb654624b8e744f`
- score hash: `sha256-2cf7ff5341488052638289ef1650500f81a31309eb2df653be3fd6dab9b3c977`
- bundle hash: `sha256-3083fc9fb9c04db3d8feb76f09c93e8491b84e34eb9e5ed92a517fcf653268f7`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5406b4db2619fd299a4dff36fb17ece03d149828bde2ae07870bc2e0cc31ba06 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4529368414f3e4a1d047973c4bc9667ab2d96a17e5cafc95290fbbb2355ebb27 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-642356a23c2fd0f9df7457c4280ceb0c9b4d599cd2c20e1e477e57414fbb1ab2 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-94dc37aab1cb6eb936944eadd5398ab89ed94d3eea04245847cb34d65d16a8b1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-08f68f96 | sha256-80be2b69182c39fcff4fd4bf5bd04f2a24cc52c911ddfdd28e66f8c371a7f20a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-08f68f96 | sha256-891becd2925cfd95f385ad757407d0c16e6e346dbb47f4ac5d0be4858caaf862 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-06b19271 | sha256-4949ebfd9d7f6bd0ab33a56a23d3933337c51cf500e2c0c519938be406d6a807 |
