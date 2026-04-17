# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7a2236f637704fe149867dcf144d671dc7a13fa94e04f98252bb7a94efde6a70`
- fixture hash: `sha256-3dd95fcccf0fb105acb53dbd74c41b44d30300251f8ca1b0c6b6f7ee328de982`
- score hash: `sha256-3550457154f11c3571390c1c8201ae4aaa225f4c6542a192cc01669ce0fc8d4c`
- bundle hash: `sha256-97bbeeea61482e4d9ad64b306334bbae21e8046bbafbc93e2c7123be32264963`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ef52e9f6d08b86a0755671620744d8fa71177a56d88b43c65d023da00ed4b3db |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3cf5cb767de069806bb35e7158f425a526458cb37757a9cb1a2dc8937a663cc7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6a5221c004097390ffc6f0d31517d443351d05455290bd770399cd856feb297f |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c0ab91d1d4528f4858b6f115bfde397ad353a1dc56a41813937b241dc78e88e2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f289f732 | sha256-e8022daa135faab4a251b99d33f74cad9f03fd3fe1beebc1a9fa8cace4ec1a5d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f289f732 | sha256-1500dbbe9f4b13c10e42eb5e40d97ec35212ffe5bf1344b5dbd02ba7e88bfcc1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8fc1f49d | sha256-78610c27e2818028520861172ed50797d829fbd22f33d234047082a0220fdc02 |
