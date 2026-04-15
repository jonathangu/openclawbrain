# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68581f69a97780aac278954522193e99993d4befdc39acceb8ff881974cc0178`
- fixture hash: `sha256-d2931cc864933b7e6af27eb1382872e22dbe9358020b6cefacd8fc78d2489792`
- score hash: `sha256-eceb8f31eb039ecfa5724a75662e49686b7e0333971d0c5f5bf8b1b5210217c5`
- bundle hash: `sha256-c78791f9d9aa2c280188b5d6907246d3eeb25c199f3c5abe6b776c4b80526972`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-181208f7b843fa2c39286593bf1b96c7f44d97e1cb317cd9b55efb3be3bcccb4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-78050c2291717f11f11b282af1c70e3b29bd8d0cc780787a10d8890a72ea8722 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d46d0c83551e105713087019ccd54e9465ad5260247a8d7866ccb6ef8885cc9c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-75151d9f676635c87066a52bc3db42450e90e91a7ef73a37b8d1b767056c6d45 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-58417099 | sha256-e8f88d1b3d3a66d5ee89046cd8c5069e410ee06cfb7148b7ef5ce8b9eea0d1bf |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-58417099 | sha256-ee41fe758c3f2ba3a511a44a9e2210434a48e73b07c964b1fc4ed4ffddf8e89d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-58417099 | sha256-e8f88d1b3d3a66d5ee89046cd8c5069e410ee06cfb7148b7ef5ce8b9eea0d1bf |
