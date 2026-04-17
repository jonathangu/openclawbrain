# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22f847bbdb3bff7bf5823cbe39964b330b3ee1ba23484549f7f4546fac1981a9`
- fixture hash: `sha256-a7f2ea82d1ad7a3badc44ebc7ebcd547c985d36abe3fcd06170981ec576de057`
- score hash: `sha256-49eed6de580e32bb7e6f3e9ec7a1f1e29aa4267416c5d41594ff43eb175dbd75`
- bundle hash: `sha256-2f8a2df39065c53abfd1997040995651e9778c4ccd51d2c993e8d6460e90d934`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9025e023a7eb98100239409dc6df273a8fbdc8529118429bd0cb2b4995877ef2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-30d76dfa880a6f044b32e886e53b7022d0584664c686ed91878dc211e0f99989 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-340d125db94a65f6623cb246bc221c31f56c41aa86bba9cd85f3baa6ceb5e55e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5814ef7f39789530edeb46bcb9187f0b309bd543dc70862d3eb2c072a7d1fe56 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-599634b5 | sha256-c5dcf44a90d61cc57bd241db9e43d76636cbd45956409e9bbcfb7f457fd59e81 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-599634b5 | sha256-6b50dda5161e4fd526a87f8a825614f3e5367c9c84f5c83cb30c01c0b170c444 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-05a05282 | sha256-ad0ed097768d3555512b47e3f5b5398b90312ae4c414efa149e56710668941f8 |
