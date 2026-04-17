# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183`
- winner mode: `graph_prior_only`
- trace hash: `sha256-203ac39480367005fd42cf7825311a0cbe85dd80f56721c12d00a8ea3f270b1f`
- fixture hash: `sha256-f1b7e7068a4652fbad5d085cdb0c1a635468b0ae89cc507258e65b4da9413c08`
- score hash: `sha256-7b4a2c0469db24f5fb9a3aaeb286167abc18f3a4af8f73c19754bf29fa2ba2e1`
- bundle hash: `sha256-13a39fcbea4255e354bd1af53eec21820bd6d273d791e5c542c3f081e3195fe9`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-08044093faa209a549cf6cbe79d77a3fd872d3cdde2c86b5886da5044f650477 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-29a13fa9d80d532f3bd022e83f298ee0260d3dad2b3ae2a073e5d822da14507f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6c2b40adb524b8e86a2dbda8adbc013ced2c716c5e0e324efaa809cefb507d18 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-b6644a58dc40fe3e43bab4ece4fd5ddcb87dbee8a44040538e439673830de50b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-627267c3 | sha256-d7123f8108ac7c2c3cf3eb7f0378274f9ac7b32b132867bb6858cc23e2929dc2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-627267c3 | sha256-2c53f21b5877eb0ded00d041e1c8817f9036236c691e9a6ae9740af0e57d284e |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ce5b27ac | sha256-fa3fba03d338f9441b52477fabe4ea5938ea47c455572ba15725b9556bd73d9f |
