# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd603b557f17857772a54a908d5f0ba5df5b9405e501fb3b65a61cd496b30680`
- fixture hash: `sha256-a72ba01cb2a77634727f52aed3858de560e72d44f77e52442f91249de387c84b`
- score hash: `sha256-ffac7ec6eaeff28fe2fe5bc58da48c03dbda267cfa08f4d6d95c790d94346a7f`
- bundle hash: `sha256-0679d275886cce2aeb00ec6ae03eb26becdb6ecea95c141bb3516ca9ba03688e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf4126f9f6708337f7f0c62a2db19e988397589b870e571ff16dc3ae73782dd0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d9b00b369629204738da81b1a5475856e1a4de08fd361b460ef315ea45922818 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9a940cbe0f2ff691698503af61733a7cc5ae056eb0738912b47aeca6eaa350d8 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-2adef09b0aa9a250706e15cceb0ba9ef05c3cb987891efb6543f65d7ebf7d7f8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e0f6ea20 | sha256-7d220dc3b96e83a4aed0645bcb5f2a7c2bc80231d7d6134177dfb2a81b94de4a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e0f6ea20 | sha256-074dce3679ada82f4baad76335e2aca417f0826b15167744a6fedc2150aa3143 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a0e24153 | sha256-07ad1bd4db00aa7672ac21e31304414eb876d9034f9e79311ddc70ee6bac587d |
