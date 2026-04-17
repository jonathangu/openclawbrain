# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304b5ee53cc148670256892da800bf0d31f07b699447be9e8eaaeff5a3c2cab5`
- fixture hash: `sha256-60dc2f86ac1ee754f931ba95c5a33382b613c3b1b0a7e2c96deb303d2eccd093`
- score hash: `sha256-352989a25c71b795b1f51c6d618dda39799b70290d87ccce583547c9672b22b2`
- bundle hash: `sha256-1fd6fa8cd43cac5a24aa843fa40e25091a5b10e8e416dc04d6cc6524ee2495a7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81d1a7582801981771b9bc27a32c83725b8a8a67e2715cd65f17099531df2d18 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-11bbe136447f74f4bb9fd81070fc43e763810da79115797e5f2fe05bfe108734 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c21e142c7273deef06d9c96558ecb1fb1ffe3673e1b77c880a61a31fc7b1408c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3f2eec3cfe422e163280ceb3f91bd6c9b55198c79ceae81b522aa461f92094f6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-239d2299 | sha256-fb9ddb54248656d4de318580e92a638d0f3c2437820b5f89c7c86e221be72bf7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-239d2299 | sha256-71a3585783f1a359953c5d833b2a9857ba8aeda2be92902b23cd34cf537a2408 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9d6bfb52 | sha256-853a607e17b0e5f83ca33934fd8d0d36c3a27d25f021fbf0ce8969daee1e9dc8 |
