# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0daac77494acc6aca056ecf3e9f12fd58f33e4988f5a646f0ba5dd6ef080149`
- fixture hash: `sha256-267dbae4de2075656207ffdf48cdf822d6c7cd1996c42f8535ce786c53f3660f`
- score hash: `sha256-f0c326f7804fcdde2c49f4f2485e49443e8be0a2cf344298a275db8e50e63e51`
- bundle hash: `sha256-eee12890d73ceb4ad8fbd85dd372fb615a5b98564b17f36e86fba66df5881067`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7fed91035473e3fc2f9947814563b07b47451ce7ca5cc7e497e3f1c68f58c389 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-414f368cf3373910b6ed0d3b015158cf1fd7b5bb840fdb97a3d4a7ade8d9c1d0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2f7dfb6b71ecf6e67da5e779b424057faaea9315b95812f91ce0b515e62714f7 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-958f417e89f41365c9fdc9cf6ae7dd637cb33f1da3d870bd7ed3752548ae0d51 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2e9ed52d | sha256-600150e2d88fd3f7054d6c2c5e3ad27b0b55f465ad6bae19a6d11e0f62a336b3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2e9ed52d | sha256-c69e0821a53720f5dc8b1743b87fececd97463d70a572dc4ca977d10f4b32134 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b17b7fde | sha256-37c36caf5f93e4cc083298d653c17e706c360f35ea8e11832e8bce3c0a5513d0 |
