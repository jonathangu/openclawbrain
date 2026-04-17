# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-47e2bbb0e3231912cba4ca695ee40afa634c254d39846db31c816e34ed5fce09`
- fixture hash: `sha256-3d564635ad5110cdb42d334240ce415d162e817662b201fa3f2e0f2bf21b9556`
- score hash: `sha256-ec6e8b33b7663fdd6ca76ca95d92144bff992cd968664c31de8e1937327e0cdd`
- bundle hash: `sha256-25dc4147caea6e479bbc196135df3a69a1923f78addf3be5e4b4241b3ac73d4d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-caaab5b4f02580bfcff0412720641649b1a6015360a0de6e547e4cce40c83035 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b7b03d883c23f0df516c456a81ad3db2066fedd223c91f9b062ec102760f21ce |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10a20c11abd4c93779edd4b122c6eebe9efb0c0754fc268b53ebcc401b49822e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ca0d3143d2bf2dbebeec8d94d54725f0811a1546ac8de98059ec9e7b0c2377aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4b49676d | sha256-99f168b65bfe697d52d2a67ef505d1edd5727effc700f53c2071bc258dbacf14 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4b49676d | sha256-174bd1882e674bdbb601c20affe337250129ceee8e7db059ab3c8f13afae4067 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-17212974 | sha256-cdcf2b97f7f5fe2bb75382b146d5694508feea511629c6ce37c43dcd7a5f2c4c |
