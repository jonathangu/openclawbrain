# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6f8a152004f1762e8eed1ecc9dffc7029171fd911a8be7a9ecd27602349fd8ea`
- fixture hash: `sha256-fef2416147c50461e059554c89ffc13514d9838c25717ac0bb496af10eda074b`
- score hash: `sha256-0bcd37f3fd635a80e3b81f7e7599f87aa9eb7c250a4292b3a16d04dcc576a0c4`
- bundle hash: `sha256-d6f78807309aaf2008669f1ec5bd90e9f0fd17b01fc7eca396d81d670cdc94ff`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-510726e0c2b1103191bca21eb76122edbdd44953bed132c7c6febf953ec52703 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caf423dcd31db9c42cfdcfec0d125008b0577ae3795dcab5c18fbd7ad40d22fd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-db8358b14c145fff43eaa3276266f4aad1441ab7c87d6bcd07ad7b00b84ed301 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-51f7b282dd3e8bb1e5863ba24cb23b2f08ebc0d5f184a5570c5bbe0cc9c0574c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0b88483f | sha256-b3707d6e415b09d9165951470014dfa955d36c25ef4e694acf35aa287cf39190 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0b88483f | sha256-4b5ce121b87d91141e31b8dc1520df6037297faa70a26518655ab771772d7986 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ab39395c | sha256-91d3a1dfe2d5226d9c0fc1c7d2d90c3be80ec7ed18ae7c77c58c3e3ba17fbb0b |
