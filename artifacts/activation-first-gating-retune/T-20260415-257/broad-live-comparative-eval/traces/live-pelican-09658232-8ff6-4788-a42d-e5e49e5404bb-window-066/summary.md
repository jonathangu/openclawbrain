# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ac1c0fab25c05525176cdfff2149e8d15cf9da0d9dd3e3ff8d1e6b40aadd074`
- fixture hash: `sha256-99a9dda4d1e27d20e5b5802fe99ae2cd9ee98cd875422b1ef45282c42f60a797`
- score hash: `sha256-8731ecfbef425eeada8352d290e283a20108e01437fd90105ad2d88c5bb25e0e`
- bundle hash: `sha256-f72b378dc25bbe139a3b7f251acd6bda7eaad1059ca9500f532a03b007363260`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-542ceacbb02f7d1851a8e974f5c261646a42b22cdc5aea08b88b8ec0e05b983c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fbee6086ed485c968d674873f5f2891e178fa4d690a915f7896076014031d6cb |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-244e7e73d1108d86d63f1c8c8a9e2ea39a07caaeda1d5fbad3fd54583f32921d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f5b78d9d | sha256-8c28cdf8a9f01d6fde46fd6b8c959938168fb8c3987a41fdde463e3659ef462d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f5b78d9d | sha256-e2ad5ccba532e87392bbd35e129b2e28534955729f1dbc5f8d8b3418a059f9b1 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-45faca66 | sha256-097317ceb0b236ad47a5cf1b1438583bb1b7cc2b4ec7896e4375a76d317beca5 |
