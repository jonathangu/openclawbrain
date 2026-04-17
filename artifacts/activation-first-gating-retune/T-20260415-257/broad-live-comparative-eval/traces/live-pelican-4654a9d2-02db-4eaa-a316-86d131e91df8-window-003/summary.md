# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003`
- winner mode: `vector_only`
- trace hash: `sha256-871a8d0a2f1d4e43acf8de9d8e6956ae4d1ca9dd0a419c5265c96970bba52722`
- fixture hash: `sha256-219199343b7c6d3ad1312b7304ed4e0c3741109cf5c94240ae657c56e05e2f48`
- score hash: `sha256-e41fe28f8723b22ff64031f6a69e59f777b06a6dd0d08f85188fd41da459fc2d`
- bundle hash: `sha256-08d9c6bcff0e166cfa7cd74e5e840c95e01886328b805cdc9c02d7565955667e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 70 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/8
- phrase hit rate: 0.125

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b917e276e14b121cc269645e14a5fdafe3dcdf3d48a758ee09ff6c7e3bf5cdd4 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-2a62dbf7576d9193593f4c04a24362aa1866ec3adb4ea860820b2d569d2e3684 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fb2e3142ebb07ad5e31a472c58d354dcccb42f344edaf1284ea894e0ddf6020c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e2291f8bcc3d1829afba0840e61183f1e77cfe9fafc93bff92316d891765cb9c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-a7cae59c | sha256-7830090cdf6d5e516d897ccfc8490c8a105a1adb8db3e6bb90ccc1bd89a9c9ef |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a7cae59c | sha256-c5946ceb8ee42acde867774ab066807ec3b17ea8faf77be820bb052adbcc72d0 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-9c97df11 | sha256-27d1fd3d78816dc3abf87d67b3c7ad3155956a7732d9d429a08cbf4384fefc3d |
