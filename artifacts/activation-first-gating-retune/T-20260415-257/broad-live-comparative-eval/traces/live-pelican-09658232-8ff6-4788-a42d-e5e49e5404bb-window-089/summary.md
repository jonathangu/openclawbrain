# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7a2236f637704fe149867dcf144d671dc7a13fa94e04f98252bb7a94efde6a70`
- fixture hash: `sha256-3dd95fcccf0fb105acb53dbd74c41b44d30300251f8ca1b0c6b6f7ee328de982`
- score hash: `sha256-4954c609fdbc2be259b3df762412b21e146a997d738b6afb7c540ec71509cb4d`
- bundle hash: `sha256-7678c4f2688ca904a4a23267ed6a50d9ef8797e8c3aadd97d02d021463aab340`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ef52e9f6d08b86a0755671620744d8fa71177a56d88b43c65d023da00ed4b3db |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cab6e0f97f428f6d68aec9f8da72ed0890d1a7d7b41056b621dacf6872d965cc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f027f9c08dc98a066bd3de6891b8ff24cdb2e8c980fea40c8893fa1bac4b546b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2497e7771c6d1e4ed7831f281b58a46b845ba3bc64ea2dda6224748fc8e1dbd0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ba9173b | sha256-ebbc90ee4177e196121396b0ddf8c7d32e80bcf6cb9034f2761e052725b843bc |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ba9173b | sha256-94f5f9dbb0c40438577738fc6af8589d6d22ce595bf5997245b0f02388fbb222 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b8e114a6 | sha256-fcd76fe413c07e940ada77ec9560b91b981c637b976b942300d21636013a7a63 |
