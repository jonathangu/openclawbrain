# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f60100eb1742bfc6c299af2f5afe9b6c211473ff986af1ecb211b198ec2ac6e1`
- fixture hash: `sha256-7060077aa89ea2d2ed121c14a4166c1764801c149a1d2df1467761d22c2169ae`
- score hash: `sha256-3dc629451034c37b6e55ff4a8168b6e1382e0e7504857763433baacd53ff8693`
- bundle hash: `sha256-c79264ca94119c37933c99350e16d95cde93e091e7888d455a82c165703251d9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1af67f3f0f2a5d2c63ece4b570453604d2bc85441d7219830f849b19b9d0d604 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-42b7c960ccc2b406208a85c4eed4d55e47cf26d1f62ea07ef6db9619755c140c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-744372613bacafc58d525d58204b67a6f1e3d86f368d3a39fcc99bc3d807206e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c06c75b34d6e8678f39f92d8cc664cf85a896ccfa5dd9479eb601e0905f6bf5b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a16defa6 | sha256-e46e3b62d45b30a0a8275d60dfe15d793ee38c7aacb1eb2a8a28f84f8919228f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a16defa6 | sha256-590fef5931bfbe86c8f830cf6d2f2cfa40b3403f7cdc45a154ed919711d91cbd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-244d204d | sha256-01d4c1406c142a154a5210f8e18f703f3f0545558d98468ede2b2990b3b3522f |
