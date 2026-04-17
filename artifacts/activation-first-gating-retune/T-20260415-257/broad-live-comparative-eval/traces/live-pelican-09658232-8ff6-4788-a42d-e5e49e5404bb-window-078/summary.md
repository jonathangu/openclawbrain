# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8fecd38f3aa3470c67016a58c02da538613366240f311d73e765e2e999bfc5e1`
- fixture hash: `sha256-9a635fc4466dcd1f01d2e94228a353c7c6a97d36b77eaea2bf2676d0c4e0cb26`
- score hash: `sha256-e515f7486791fa05712eb3527f8ea5e4a78c6b1a1b2b78953009c18e7b95949e`
- bundle hash: `sha256-0968d587f685c609080ee9cb7118c07b4f69ea28119c2ab4667479ccf73910b7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32d500730de7d73f2a2bf38e8b78d2d6ad04a3a58dd8029622c951f7ddee70 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5c6cbbe5d125d2be3af19dd4e88858d1e4f821aafd1307ec6ebbe8512b3aadd5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-73ef49d187424f83f837ab771165ba4e4d13eeaf25b15e67f77667cde20c1a9c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6dcabdd32a3bd74bd760b93284b3b698fe0cc5a5bb284a020854fe9589131424 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f6e19cbe | sha256-a5241ecb4728772573d629a3c5e3b1768bb126da102d8e5865d8568747d1c625 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f6e19cbe | sha256-d6d815cf1a60a64714b52cda5b88eafa027fcd56a549bae6299311c155955477 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e386ca6d | sha256-cd96f4fac1d2a01d3af43ae44e61c1d25134e8839f5f522f97beb9e4e10e2d18 |
