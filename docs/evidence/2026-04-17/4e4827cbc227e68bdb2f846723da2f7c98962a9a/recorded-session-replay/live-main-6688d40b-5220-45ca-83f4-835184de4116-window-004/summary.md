# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2210bd8aa54ff55f81e90c13af23591578b0c820206054d3d91e01211b88bae7`
- fixture hash: `sha256-a562aa7a1ac863aa823f236bdbc816afd7b8d62760a47e5474f699f78bdac5e9`
- score hash: `sha256-55418e0526f42d75543044953a82b15ed34b953b4567740e83f33b79ee0efb25`
- bundle hash: `sha256-a2f48e320348d3827aad27f736148bd045b6652f337e8e25b2125efa045dcfbe`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cbbf8eff09f23d982b9af94fdc9d383c8e6e748daa65afe086a31e073a634311 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3e3ba88f2f980ddcbbceb4a7141e6fb8984e01496c4af74174d3424c72d8d0c0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7d9013e6ee96f67f4934a404f557d8c7435739b3468b2f8561803b8b770b0b44 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c9041c11fcaa926a177e9e157ca20d9b036bc026398bfd1aa0049650a46e212b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a197a294 | sha256-d9dacac3d452bb558d1f907ef2b72e5696ca4c44cace1fef617f7553898fe355 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a197a294 | sha256-c35c880002b05afb30d8644a56a418d89fff25e6fe7199f77bcd4d6212769c07 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9e9dd8c9 | sha256-0db87162c85cd7a1086bcfb85a5057f852f87c9d981209f960647b85b904d789 |
