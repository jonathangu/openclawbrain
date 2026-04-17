# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dd29f792fbfbc606fc0ae81485babcea7a498bf8f85b66de2333918434925117`
- fixture hash: `sha256-a84a33537b8d24e458443c5c6b1cbd9d02b490a8b56c8f49f8509184e51ddc87`
- score hash: `sha256-b52617649538c7dbbd4d83dd031116402d0b267ea977ef35db1343abfcc549ad`
- bundle hash: `sha256-2cdfe668626a4ff169a8be9d7cd79d1e5a160bdc48b2ff4a79a69ad3e3688115`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6181c2140140f5786a710721c2d0cc92976577da480a328a542e8b790bc4990 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cc620f21afd8cc03d72e0066e59db01a471dc84ed3dbd3674c529df7c1f9381 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c9686fdb0b8702b5803f091259e3eeadf5a09a278a88098f51e90e87466ced9d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8d500fdfbdeec654de0c23b240b72e85c0f28aafdf8fb7cada657a93bd74d2ae |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a35c55a | sha256-aa68d3c59f7b5be34a20bfb1a170d3ca32d1e08482c8c56364d9098552ab8388 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a35c55a | sha256-83bdd889d43e5d28adbc123ad0702b79ead9183e93f5c75ba459ced767adfe92 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-77351c87 | sha256-7040dfe505105ff858c23f41c59f9429551655d3510a9c1dccbcbb64f069f288 |
