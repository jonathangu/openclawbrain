# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a725d997b61c014cfa09b0556fbe7457ad1066c9e82be586f6e632957134b68e`
- fixture hash: `sha256-0175601a40ce3b110cb977baef20750f4bd146a6b251d6933e92137c6f93984a`
- score hash: `sha256-40aff0ead9ddcfe635caf76194154c8998437e6e55fe60ada86b870f3a015c08`
- bundle hash: `sha256-11c1b99b82758199d828560cdc98a943c8219f49113eb6cdf7ea2a29720ba65c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5b1c6095a0bb639e8d10640852b232aba3f9944783712101783106fcf8add6d4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cbbd87f072a4d8536aa3fca8ca8a33c67c15724e1ffa1ed29d9c2414949220fb |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e67d217cc52bd33abc4349e5a52eec621e1fd15a0891aab1f56c8ee55fab7d6e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f57c31dd5bf046f94f555fa44ea2efcfcc4e52efc58b60955c138a77adbf483c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aed4637c | sha256-e668ee10783c806b72263fccb6563ca4f812ee0d7e3cd74c83c3fb0a9154e94b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aed4637c | sha256-b1a31531daade234e4eeca8d95a1a5b15c91f36675dbfcb22351b9f34e27282c |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-25f278a5 | sha256-2a68edd48f683cba20a4bf00e894111071e6a515dd86bc22c5c28a96826e7495 |
