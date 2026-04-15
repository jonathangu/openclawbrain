# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22c0c5cfe30f6528627aae6b3b1ce6c55137840c4388f7d03d5ba0c64043e114`
- fixture hash: `sha256-883333e2877ee56be18afd0bdb26f3a044eab5df448e40bf59cfd947e2e070a7`
- score hash: `sha256-32fe2a01e1760196b3cd69ac571de8fcb8a89d1c32eaa25df3d70f9475fd9b7d`
- bundle hash: `sha256-2836ef51cdea65b1aa9a2dba90f93b3d3f46bd14560cf4c4b31224bdaf3d7b2d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bfacda8f5501f5e4f01bbebcdaf7a5c0e18d211755bb5803d41f576de0d46bba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2bd4dc520977e6d6152a39982d606c277eac15bbac02628d612f02de92cfcfff |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-706d16cdbb5ff230d49a32d81ad48a70ecfc98653b9da24ac191c15b23b45b16 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8e61e91edd9f2f96412d346508bd2581d9e0746db4cd83f0ed7e463fdd55d1f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-97ff9813 | sha256-2f7a7c4ea9c36122636bcb383f75b5e39e7e15627352e4e264cdf2dfe64980ed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-97ff9813 | sha256-c6a1f5db075c7ff0130f6504f4441d8bc5ab33ba599a8ae49d65062ee87caa0b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-97ff9813 | sha256-2f7a7c4ea9c36122636bcb383f75b5e39e7e15627352e4e264cdf2dfe64980ed |
