# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9de933020cc2b0d03a4b1b4f5bf51c7bca0c4ae8de78af7a2cf6d4b86ac284d4`
- fixture hash: `sha256-2e5835aa933cf2df6faf2714837c2953d1866a5094413604d0ec3e648b5257c4`
- score hash: `sha256-ddf4cb132d818010de58019ceabaf1134a1253679117c0e56a5a7c0033860247`
- bundle hash: `sha256-7db2c84bc6223ecc0f64f7765f24c85c3b6f20d447da14c32903705524232dc6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-13242c9a12ffb4d788c2f14891b978d17c5b819a44b8fb4dd405e1c1b50322e8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1278ea1dd216c54af5d8c51d31ae808b7fc97f47f3a90d79f7a84fb06fb16110 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-501e53da971933478df1d86baa73cf7559291b8e9d3a3c09e2e3a8a69cd18604 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3daeffc46371a9778b7911325da998bcb6961977c1a69afdcde8e8db1996b2cd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4f3ece5d | sha256-80b98eed4557b0c744c617f227da71fc187db5f3a569579435fcedb454b66cf3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4f3ece5d | sha256-e9a42b6ce69055732918ab43ff0882827118e9d55a38c801880361151be77fd6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0473bf90 | sha256-15c4eb19ef202350b46eee0838dd19376f35fdc01a3b5227a813b04413ede5b4 |
