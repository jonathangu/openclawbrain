# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f10c00dc4efb180b5273900a9e561d1c614344a77050359aaa2d54aa27cc20d2`
- fixture hash: `sha256-43065829df1e95ca79dff07d99e5773679b5561b6bbdd3945d317201ab2cca51`
- score hash: `sha256-f90cbdf43130edd4919e1bf0617b717871af0ac3a6aecfd130b112605bc269b8`
- bundle hash: `sha256-de349710368ffe62ba0656178b35c3cbcf25f2578dc186d5e5651657c3481fcd`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f1460fd13a644dccb389d5e4bb97bb20a28fa61d221da193a36a1bd2b7379c0d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-96f8aa55402b169bc0e5f3a5438ed58619f2ae3be4414dd785002e2802119bf8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7cf13643668032d692e29c69c9c16065c8ba4a28f669974a019d6ea30848a5c1 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-6e33d85932860206d572052b41b095f7b45a08ed6b212a1762e8a69e51206a89 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5ac6abee | sha256-801df4b470c0e445d7f6f62e6d1bf5e82f323c2f692a06c78b37e85988fbde42 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5ac6abee | sha256-801df4b470c0e445d7f6f62e6d1bf5e82f323c2f692a06c78b37e85988fbde42 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-5ac6abee | sha256-801df4b470c0e445d7f6f62e6d1bf5e82f323c2f692a06c78b37e85988fbde42 |
