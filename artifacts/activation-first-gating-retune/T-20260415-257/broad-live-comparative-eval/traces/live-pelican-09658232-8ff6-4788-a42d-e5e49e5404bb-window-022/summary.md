# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b973c64c8eac5a0b6deba25fbae9f31be4599e3d19192c7c9dd0b18e718f1e`
- fixture hash: `sha256-b932d5e627b7081f980ab111b252e205aa7e0185bfcd774e6388fb9e948098c1`
- score hash: `sha256-fcf3da4c6791ef33707cab1d3ef2a8fcc6af446ee318d712b620581fa0d9fbb7`
- bundle hash: `sha256-0c7122bf20c9455fe52a5cdaac5ea0cee65ac0e14225fcb956e64cd901cfd528`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd3c28bfccf2817f3d01d14dc16c97875abfde806e8cfbeff2d04b6e2a397e7b |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b63eeee8f20466828c6babd847d1f57d9dd420ca39fcc585ca8108fd458418ba |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6cb4df059b0bd955f9d948b58d5e9d37ff1e317d03e72b7bfe2f64e239b3fb2c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-00d5c19d876c002b4fec466a7e5d5defeb3452d90ca9977e01b2941ff2f3dbac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-58b86ac9 | sha256-793a5af6c7ec76bf43d2175393607f4960f75adfba02d287943e0a8c8fc88400 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-58b86ac9 | sha256-793a5af6c7ec76bf43d2175393607f4960f75adfba02d287943e0a8c8fc88400 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-2ead4ebc | sha256-d0f2bbcc94873bdade35c2d71aa947ef5c257f7d167af6dc08bd8c0283dce2fe |
