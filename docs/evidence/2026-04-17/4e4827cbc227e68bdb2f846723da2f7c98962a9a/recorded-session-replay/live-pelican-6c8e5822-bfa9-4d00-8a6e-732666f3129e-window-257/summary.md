# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cc22db3aaa15315761f798aaec1df1acf278bfe86338b981b1d314f80e60f459`
- fixture hash: `sha256-6250124575745297903131838786e09bce6bd0b2285afd782515714f7d74a408`
- score hash: `sha256-31e7228984db2fe435fdb5d0adc2d7aae7d34ee23d21b3a36bd2ba0fd7b974f9`
- bundle hash: `sha256-019197ef5db163c466d17b1e90a6395bffa11b5bb39489e44b6c591647d31be4`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-363a0ce15ddf219a167f700ba2217552de3446a99d080128478494cb795b929d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-1c09b69e0ac217db57f2dbdacb2a73041ccc869310b2942cb9a02f1dbb1e9e8f |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-2e34f2601afb18fba031c89dfa550aba7da2a0cbc8d27f731869cff4a7eaf51b |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-628e0932c069943c39b3c87852e7f0b0fff6419c61dff7e5233bb7bfb29754ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-599b7977 | sha256-3f859647bf97c01fdd579e90d45d5442b29e4ff9840f02a5b94418ee7c188e14 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-599b7977 | sha256-9fb1099193accd40861afd4a3d3f5ac90d338cd6e3b96c1e3bb12ece9ae317b0 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-d335ae36 | sha256-9041154e84b413f96de04ff3eeabe40e6828729514d373b691c63f604676d88f |
