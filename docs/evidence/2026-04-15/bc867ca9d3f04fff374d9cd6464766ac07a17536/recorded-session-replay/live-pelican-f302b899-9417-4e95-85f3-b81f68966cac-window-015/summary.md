# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06c88cbd7b40857f6269dd03d5e04022f7a27c8c5e2a225bc79b2768cb90fdfd`
- fixture hash: `sha256-6d62bb5ab6456b9eec73e20f3d1a35ffc14e9452a4f4442f3b56ae134f63d27e`
- score hash: `sha256-bcb2bfd75bdbb4de45ac95233633bbe6082b6c652fc7e8e044974c2bcbb876bd`
- bundle hash: `sha256-caa83e3dc72ab28bd45bd199fa20bce29c4fe76670cde250ae6a0a7e304296a0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cd163af8984e87c72885a17249c9a84973c54f74e5363d963d16ae86c9b4e43 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-202cb0143e364ea5845b25517934f92518da8445cfc49f879747b59df5881fff |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0943df1fdd2cac6b152c3224c3843fd98f44c5d2f3b725cec51833469dcf81b9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-74e864eb044974dc3d9c31956e0f1ef71bcc95de0ea7b4dc6b919c7e6d0cb9c3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1711fe5c | sha256-089967eec1b61cbabdcf4305cfc601cf73ee2fd5158f0dcab300b96be6b291a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1711fe5c | sha256-ae1903cb8d81df995c63000e296a9f678627dacf78982f52d180e25fc28b5ad6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1711fe5c | sha256-089967eec1b61cbabdcf4305cfc601cf73ee2fd5158f0dcab300b96be6b291a8 |
