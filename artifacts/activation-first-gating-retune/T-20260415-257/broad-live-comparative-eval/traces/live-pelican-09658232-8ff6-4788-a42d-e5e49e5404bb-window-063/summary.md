# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bb64d18c7380c29a36adda7f18b9d94028bd2ec79c3f043249c311ff96079b77`
- fixture hash: `sha256-4d9c945d16c80ffc64625c9921a10c1aa73d0e2d0d7dc96750c287fa87ef0a3c`
- score hash: `sha256-f81589c9d87a56dbbb6b3b6f2ed0cb46020176597eccb58ba4057e87f91efc25`
- bundle hash: `sha256-3e69cd6075363892060cb91ef5514f29759c0598e089d9c2ecb3d925135c6dc7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2057eeb74350d5472d7a207a6cd23d83fdfc1cbff7a9da70502d2c9709cf85fb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9a5b56594b318dbd553d022c2825b0c35a4d930efc710d22c605db873ec32e50 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c12b710e7d147f14dbcc8130ad098a7ec3af8e8fbd98a88acd35ccb12848f50 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ce54abb525671844f2947313535e6ef2271e4ea4924d616a2d5347584f1abb65 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c6721539 | sha256-6ba9780280e11225bb358a7dcdc10d78f7e0694514eb0a4f25dc4c953a82f9f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c6721539 | sha256-d993ceeaaaa84a9f0c4c2b77ce6451ebadd6896855d2e4a55b349d105a3ea978 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8c40d0f8 | sha256-868396fb9692552fc3a3c0875d0dd3f34d45012dce66787faa4ffdc662a869ac |
