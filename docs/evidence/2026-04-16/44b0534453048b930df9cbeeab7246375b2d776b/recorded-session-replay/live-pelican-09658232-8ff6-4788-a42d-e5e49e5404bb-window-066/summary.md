# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ac1c0fab25c05525176cdfff2149e8d15cf9da0d9dd3e3ff8d1e6b40aadd074`
- fixture hash: `sha256-99a9dda4d1e27d20e5b5802fe99ae2cd9ee98cd875422b1ef45282c42f60a797`
- score hash: `sha256-f37df52afd945ce813c167e17078e3b5e141f55323cb43fea0ba1dc538c42547`
- bundle hash: `sha256-7db04a0a48c7a8820e4b41e5ce960190a4695d54b2bba7fff2d59301ae2ac1d6`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c3402e859f2552a40a7f253ef60215bf90d6f117858139b3ed26992a03a4545a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9dbd6bd938789fad50d8b13dacd1899a9fd29122ce3685f19b1bbc183c511e62 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-58809f4635eb501fdcc4520f0653e3620baeb94a95160fcd2d6e904067384aa9 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-71ef9586b1577b98793e9f55a334ff6e2e7a31a10f4c7724e0c314c7600d5f36 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cc986d94 | sha256-ffd047563146357cbe0fda4bb9eb6189e6fcd154414424635d99c54fee9bd2ea |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cc986d94 | sha256-417e87a6293e00a6e1fcc7f64da6190a395308779020f1da3398b001d5d47d79 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-1cdbaa5d | sha256-8847ad479fc28c1fb52ad0724ab048d90849a6fb599f56e0fd08f5949cbdd279 |
